"""随访问答和流式事件展示。"""

import json

import requests
import streamlit as st

from utils import api_client
from utils.ui_session import draft_has_changes, get_view, show_request_error
from utils.ui_components import show_risk_notice, task_name


# 显示本轮后端检索的文件名与逐字原文
def show_chat_sources(sources):
    if sources is None:
        return  # 普通个人信息查询和旧对话都没有 sources 字段。
    if not sources:
        st.caption("知识库未返回可核对的原文。")
        return
    with st.expander(f"查看本轮检索原文（{len(sources)} 段）"):
        for entry in sources:
            st.caption(entry["source"])  # 文件名由后端取自实际命中片段。
            st.text(entry["excerpt"])  # 按原文呈现，保留换行和标点。


# 发送当前范围内的对话
def show_chat(token, mode, task_id=None, patient_id=None, access_scope=None):
    # 同一个任务的查询和采集也使用不同上下文，切换时清理旧消息。
    scope = (st.session_state["user"]["user_id"], mode, patient_id, task_id, access_scope)
    if st.session_state.get("chat_scope") != scope:
        st.session_state["message"] = []
        st.session_state["chat_scope"] = scope
        st.session_state.pop("scope_risk", None)
    placeholder = "询问随访安排、医患信息或健康知识"
    if mode == "collect":
        placeholder = "说说这次的血压、服药情况，或最近哪里不舒服……"
    elif st.session_state.get("scope_risk"):
        show_risk_notice(st.session_state["scope_risk"])
    # ==================== 用历史数据重新画出聊天消息 ====================
    # message = {"role":"assistant","content":"你好"}。
    for message in st.session_state.get("message", []):
        # chat_message 返回消息容器，role="user" 和 "assistant" 决定身份样式/图标。
        # with 将这条消息的正文放进对应容器，而不是自动去调用模型。
        with st.chat_message(message["role"]):
            # 重新画历史只读取这个列表，不会再次向后端发送历史里的每条问题。
            st.markdown(message["content"])
            show_chat_sources(message.get("sources"))  # 重画已完成回答时复用原文，不再发起检索。

    # ==================== 接收这一轮新问题 ====================
    chat_disabled = mode == "collect" and draft_has_changes()
    if chat_disabled:
        st.caption("有手工修改待保存。你仍可打开侧栏「问助手」查询。")
    risk_box = st.container()  # 单独放置规则提醒，避免模型 reset/error 清除患者需要看到的信息。
    prompt = st.chat_input(placeholder, max_chars=4000, disabled=chat_disabled, key="chat_prompt")
    if prompt and prompt.strip():
        # chat_message(...).write(prompt) 是先创建用户消息容器，再在其中写入文字。
        st.chat_message("user").write(prompt)
        # append 把一条字典追加到 list 末尾，返回 None；目的是供下一轮重跑时重画。
        # 这里只改内存列表，没有 INSERT 到 medical.db.chat_messages。
        st.session_state["message"].append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            # st.empty() 返回可更新的页面占位对象，不是空字符串。反复更新同一个对象，会替换它原来的内容。
            # 错误和回答使用不同占位，以便在断流时移除未完成的回答。
            status_box = st.empty()
            answer_box = st.empty()
            # answer: str，当前累计回答，例如 "家庭" → "家庭血压" → "家庭血压测量..."。
            answer = ""
            sources = None  # 每一轮重新接收该回答的检索结果，不能沿用上一条消息的原文。
            # completed: bool，是否收到后端的完整 answer 事件；仅有 delta 不代表完成。
            completed = False
            # failed: bool，是否发生网络错误、格式错误或后端发来的 error 事件。
            failed = False
            draft_updated = False  # 收到已保存的草稿事件后，结束回答再重新加载页面。

            try:
                # method="POST"，path="/chat"，token 是登录凭证，data 是问题字典。
                # 发送当前页面明确绑定的模式和范围，普通问答与草稿采集各自独立。
                chat_data = {"mode": mode, "message": prompt}
                if task_id:
                    chat_data["task_id"] = task_id
                if patient_id:
                    chat_data["patient_id"] = patient_id
                # user_id/role 不从这里传，后端从 token 读取身份，并再次校验任务归属。
                # stream=True 是 HTTP 读取模式：边收到回复边处理，不等所有回答结束。
                # 模型流式生成由 ReactAgent 执行；单独设置 stream=True 不会让模型自动流式化。
                # with 保证回答结束或中断后关闭 HTTP 连接。
                with api_client.request_api("POST", "/chat", token=token, data=chat_data, stream=True) as response:
                    # response: requests.Response，支持 with，退出时会执行 close()。
                    # 本接口正文是 NDJSON：每一行都是一个独立 JSON 对象。
                    # 与登录接口不同，不能直接 response.json() 解析整段多行事件流。
                    # iter_lines() 返回可迭代对象；for 每次取出一行 bytes，且去掉行尾换行。
                    # chunk_size=1 控制底层读取块大小，并不表示“一轮循环只得到一个汉字”。
                    # UTF-8 汉字通常占多个字节；iter_lines 会先拼出完整的一行。
                    for line in response.iter_lines(chunk_size=1):
                        if not line:
                            # 空白行 bytes 为 b""；continue 跳过本次，继续等下一行。
                            continue
                        # 两步转换：bytes --decode("utf-8")--> str --json.loads(...)--> dict。
                        # 例如后端发送一行 {"type":"delta","content":"你好"}，
                        # 转换后可以用 event["type"] 取 "delta"，用 event["content"] 取 "你好"。
                        event = json.loads(line.decode("utf-8"))
                        event_type = event["type"]
                        content = event["content"]

                        # sources 的正文是原文列表，draft/risk 是字典，其余是文字。
                        # 这几个字符串是前后端约定的事件名，不是 Streamlit 自动规定的。
                        # 一轮可能有多次工具调用，因此不保证每种事件都只出现一次。
                        if event_type == "status":
                            # 内部工具进度不展示给患者或医生。
                            continue
                        elif event_type == "reset":
                            # Agent 准备调用工具时，之前流出的文字可能是临时过渡语。
                            # reset 清掉这一轮的临时回答，不删除历史消息列表。
                            answer = ""
                            answer_box.empty()
                        elif event_type == "delta":
                            # delta = 新增片段；例如 answer="你好"，content="，请问"，
                            # answer += content 后得到 "你好，请问"。
                            status_box.empty()
                            answer += content
                            # “▌”只是人工添加的生成中光标，不存进 answer 变量和聊天历史。
                            answer_box.markdown(answer + " ▌")
                        elif event_type == "answer":
                            # answer 事件的 content 是完整最终回答，不是又一个新增片段。
                            # 所以这里用 = 覆盖，不能 +=，否则前面流过的文字会重复一遍。
                            answer = content
                            completed = True
                        elif event_type == "sources":
                            sources = content  # 和完整 answer 一起展示，出错时丢弃半成品。
                        elif event_type == "draft":
                            # 事件只提示后端已保存；页面重跑后再从草稿接口取最新内容与版本。
                            if mode == "collect" and content["task_id"] == task_id:
                                draft_updated = True
                        elif event_type == "risk":
                            content = dict(content)
                            content["task_id"] = task_id
                            st.session_state["scope_risk"] = content
                            if mode == "collect":
                                st.session_state["draft_risk_event"] = content
                            with risk_box:
                                show_risk_notice(content)  # 即使没有关联任务也及时展示就医提示。
                        elif event_type == "error":
                            # HTTP 已经返回 200 开始传流后，模型仍可能失败。
                            # 因此即使 HTTP 状态正常，也必须检查流中的 error 事件。
                            status_box.error(content)
                            failed = True
                            # break 退出 for 循环，随后离开 with，关闭响应连接。
                            break
            except requests.RequestException as error:
                # 包括发出请求时的错误，以及读取过程中断线、超时等通信错误。
                show_request_error(error)
                failed = True
            except (ValueError, KeyError):
                # ValueError：JSON 无法解析等；KeyError：缺少 type/content 等约定字段。
                st.error("回答数据格式异常，请重试。")
                failed = True

            # 只有收到完整结束事件，才把回答加入历史；断流不保存半句话。
            # 这三个条件必须同时满足：收到 answer、answer 非空、全过程没有失败。
            if completed and answer and not failed:
                status_box.empty()
                # 最后再显示一次完整文本，移除生成时添加的“▌”。
                answer_box.markdown(answer)
                # 这里只追加一条完整助手消息，不是每收到一个 delta 就追加一条。
                message = {"role": "assistant", "content": answer}
                if sources is not None:
                    message["sources"] = sources
                    show_chat_sources(sources)
                st.session_state["message"].append(message)
                if draft_updated:
                    st.rerun()  # 新一轮只重画历史，不重复发送这条消息。
            else:
                # 发生错误时删除屏幕上的半成品，但保留错误提示和用户已经发送的问题。
                answer_box.empty()
                if not failed:
                    # 没有明确报错但连接结束了，仍可能缺少最终 answer，不能当作成功。
                    status_box.error("未收到完整回答，请重试。")


def show_assistant(token, doctor, current_task, detail):
    st.title("问助手")
    scope_label = "我的随访资料"
    if doctor:
        scope_label = "我的患者和随访安排"
    if current_task:
        scope_label = task_name(current_task) + " · " + current_task["followup_date"]
        if doctor:
            scope_label = (current_task.get("patient_name") or "当前患者") + " · " + scope_label
    elif detail:
        scope_label = detail["patient"]["patient_name"] + "的资料"
    st.caption("查询范围　" + scope_label + "　｜　可检索健康知识库")
    if current_task or detail:
        if st.button("返回本次随访" if current_task else "返回患者档案", key="resume_context"):
            st.session_state["workspace"] = "task" if current_task else "patient"
            st.rerun()
    if current_task:
        show_chat(token, "query", task_id=current_task["task_id"], patient_id=current_task.get("patient_id"))
    elif detail:
        # 任务改派或主管关系变化时，当前患者编号可能没变，旧问答却不再全部可见。
        access_scope = []
        for task in detail["tasks"]:
            access_scope.append(("任务", task["task_id"]))
        for discharge in detail["discharges"]:
            access_scope.append(("出院", discharge["discharge_id"]))
        show_chat(token, "query", patient_id=detail["patient"]["patient_id"], access_scope=tuple(sorted(access_scope)))
    elif doctor:
        # 医生首页对话可能提及多位患者；任务改派后也需要清理不再属于他的历史。
        profiles = get_view(token, "/patients")
        tasks = get_view(token, "/tasks")
        if profiles is None or tasks is None:
            st.session_state.pop("chat_scope", None)
            st.stop()  # 暂不能核实权限，恢复后再展示对话。
        access_scope = []
        for patient in profiles:
            access_scope.append(("患者", patient["patient_id"], bool(patient.get("is_primary"))))
        for task in tasks:
            access_scope.append(("任务", task["task_id"], task.get("patient_id") or ""))
        show_chat(token, "query", access_scope=tuple(sorted(access_scope)))
    else:
        show_chat(token, "query")
