# create_agent 将模型、工具和中间件装成可多轮运行的图。
from langchain.agents import create_agent
from uuid import uuid4
from threading import Lock  # 同一轮多个工具可并行，共享提议要按次更新。
# InMemorySaver 按 thread_id 在当前进程保存消息和工具调用历史。
from langgraph.checkpoint.memory import InMemorySaver
# 工厂根据 rag.yml 创建模型客户端，真正的请求发生在执行时。
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
# 实际能调用哪些工具要等下方按角色组装 medical_tools 时才确定。
from agent.tools.agent_tools import (
    rag_search, get_my_followup_tasks, get_my_discharges,
    get_my_recent_records, fill_context_for_report, propose_followup_draft,
    get_related_people, get_current_report,
)
# 中间件分别观测工具调用、记录模型执行和补充当前任务提示。
from agent.tools.middle_ware import monitor_tool, log_before_model, report_prompt_switch
# 框架区分模型片段、完整模型消息和工具结果，前端展示方式也不同。
from langchain_core.messages import AIMessage, ToolMessage, AIMessageChunk
# 用框架现成的次数限制避免一轮聊天无限重复调用工具。
from langchain.agents.middleware import ToolCallLimitMiddleware


class ReactAgent:
    def __init__(self, user_id: str, role: str, task_id: str | None = None,
                 patient_id: str | None = None, mode: str = "query"):
        """创建当前身份和任务的 Agent；不提交任何业务记录。

        输入示例：user_id="U001"、role="patient"、task_id="F001"。
        task_id=None 保留通用查询；有编号时，工具只能读取该授权任务范围。
        构造函数不返回业务数据，创建结果是具有独立记忆的 ReactAgent 对象。
        模型或框架配置异常由外层 chat_session 捕获并转为错误事件。
        """
        # 创建对象时生成一次，后续提问继续使用它
        self.thread_id = str(uuid4())  # uuid4()：生成一个随机的唯一标识，用来区分会话。 eg: 0517a8d6-693e-4a4d-a8d3-3f5ff53cf4f8
        self.user_id = str(user_id)  # 是这个用户在用 agent
        self.role = role  # 字符串：patient 表示患者，doctor 表示医生。
        self.task_id = task_id  # str 或 None：由后端校验归属的任务编号，不从聊天文字提取。
        self.patient_id = patient_id  # 档案范围由 HTTP 接口核对，工具使用同一组条件。

        # 身份由后端查登录凭证得到，模型只能选择查询工具，不能指定其他用户。
        medical_tools = [
            rag_search, get_my_followup_tasks, get_my_discharges,
            get_my_recent_records, get_related_people,
        ]
        if task_id is not None:
            medical_tools.append(get_current_report)  # 复用该角色在报告页能看到的内容。
        # 报告文字整理只提供给医生；患者采集提议只在指定任务时开放。
        if role == "doctor":
            # 医生可让聊天助手整理文字；保存正式报告由专用 HTTP 接口处理。
            medical_tools.append(fill_context_for_report)
        if role == "patient" and task_id is not None and mode == "collect":
            medical_tools.append(propose_followup_draft)  # 只在患者任务中开放采集提议。

        # 创建一次，后续同一登录和同一任务复用它的内存检查点。
        self.agent = create_agent(
            model=chat_model,
            # 初始提示来自文件，动态提示会再附上本次持久采集草稿。
            system_prompt=load_system_prompts(),
            # 未注册的工具无法被此 Agent 调用，例如患者不能调用医生文字整理工具。
            tools=medical_tools,
            middleware=[monitor_tool, log_before_model, report_prompt_switch,
                        ToolCallLimitMiddleware(
                            # 本轮最多七次工具调用；这不是整个账号的总次数。
                            run_limit=7,
                            # 超过上限让本轮失败，由 chat_session 返回可重试事件。
                            exit_behavior="error",
                        ),
                        ],
            checkpointer=InMemorySaver(),  # InMemorySaver：把 Agent 的检查点保存在当前 Python 进程的内存中。
        )

    def execute_stream(self, query: str, draft=None):
        """输入一条问题文本，逐个输出页面约定的聊天事件。

        例如 query="这次随访什么时候到期？"，输出 status、delta、answer 等字典。
        user_id、role、task_id 使用创建对象时保存的服务端值，不从问题中提取。
        draft 是后端本轮读取的最新采集草稿（或 None）；工具只能提议，不直接写库。
        本方法会调用模型和工具、更新内存；正常完成后把草稿提议交给会话层保存。
        推理或工具异常交给 chat_session 处理，未完成的回答不能作为正常结果保存。
        """
        # 这一轮只传最新问题，框架按 thread_id 取回历史，无需前端重发。
        input_dict = {
            "messages": [
                {"role": "user", "content": query}
            ]
        }

        # 将工具内部名称翻译成页面状态文案，例如 rag_search 显示“检索高血压知识库”。
        tool_names = {
            "get_my_followup_tasks": "查询随访任务",
            "get_my_discharges": "查询出院信息",
            "get_my_recent_records": "查询近期血压记录",
            "get_related_people": "查询医患资料",
            "get_current_report": "读取本次随访进度和报告",
            "fill_context_for_report": "整理报告草稿",
            "rag_search": "检索高血压知识库",
            "propose_followup_draft": "整理本次随访信息",
        }

        # set 保存已展示过的调用编号，防止多次读取同一状态时重复刷屏。
        shown_tool_calls = set()
        # 直到拿到完整助手消息才在这里保留文本，delta 只负责即时显示。
        final_answer = ""
        # 本轮共用这个字典；工具将提议写在内存中，身份和原话均由服务端提供。
        context = {"report": False, "user_id": self.user_id, "role": self.role,
                   "task_id": self.task_id, "patient_id": self.patient_id,
                   "draft": draft, "patient_message": query,
                   "draft_lock": Lock()}

        # yield 先发一条状态，然后继续运行，前端无需等整个函数返回。
        yield {
            "type": "status",
            "content": "正在处理你的问题……",
        }

        # 框架每次返回 (mode, data)，由 mode 区分消息片段与完整状态。
        for mode, data in self.agent.stream(
                input_dict,
                # values 每一步执行完后的完整状态（比如大模型决定调用某个工具，或者工具返回了结果）
                stream_mode=["messages", "values"],  # messages一个字一个字蹦出来的模型输出（Token 流）
                # 指定 v1 的返回形状，与下方 mode, data 的解包方式保持一致。
                version="v1",
                context=context,  # 模型的工具参数中不开放身份、任务范围或已存草稿。
                config={
                    "configurable": {
                        # 每次调用传同一个 thread_id 才会取回这次任务上轮的消息。
                        "thread_id": self.thread_id,
                    },
                    # 最大图步数限制模型和工具之间反复往返。
                    "recursion_limit": 20,
                },
        ):
            # 第一种数据：模型正在生成的文字片段。
            if mode == "messages":
                # metadata 标明产出片段的节点，过滤掉非模型节点的数据。
                message_chunk, metadata = data
                # 只接收 Agent 的模型节点生成的内容。
                if metadata.get("langgraph_node") != "model":
                    continue
                # 工具消息也可能出现在流中，只有 AIMessageChunk 是要显示的模型片段。
                if not isinstance(message_chunk, AIMessageChunk):
                    continue
                # 某些片段只携带工具参数；文本为空时不生成空的 delta。
                text = message_chunk.text
                if text:
                    # 不等完整答案生成，马上交给页面。
                    yield {
                        "type": "delta",
                        "content": text,
                    }

            # 第二种数据：执行步骤后的完整状态。
            elif mode == "values":
                # values 含当前全部消息，[-1] 只观察最新一步。
                message = data["messages"][-1]
                if isinstance(message, AIMessage):
                    # AIMessage 有工具调用时，本轮还没有确定最终答案。
                    if message.tool_calls:
                        for tool_call in message.tool_calls:
                            # id 标识一次具体调用，同一个工具本轮可有不同的调用编号。
                            call_id = tool_call["id"]
                            if call_id in shown_tool_calls:
                                continue
                            # 先记住这个编号，下一次 values 再出现时跳过。
                            shown_tool_calls.add(call_id)
                            # 本轮要调用工具，前面的文字可能只是临时说明。
                            final_answer = ""
                            yield {
                                "type": "reset",
                                "content": "",  # # 清空前端已显示的字
                            }
                            # 例如 name 为 get_my_recent_records，页面应显示中文进度。
                            name = tool_call["name"]
                            label = tool_names.get(name, name)  # eg:正在 "查询使用记录"
                            yield {
                                "type": "status",
                                "content": f"正在{label}……",
                            }
                    elif message.text:
                        # 保留完整答案，供最后确认和保存。
                        final_answer = message.text
                # ToolMessage 表示数据已经返回模型，下一步仍由模型整理文字。
                elif isinstance(message, ToolMessage):
                    yield {
                        "type": "status",
                        "content": "工具已返回结果，正在继续处理……",
                    }

        # 有完整回答且确实提出字段变化，才把候选草稿交给会话层。
        if final_answer and "draft_proposal" in context:
            # 内部事件被 chat_session 拦截；只有保存成功的 draft 事件才发给页面。
            yield {"type": "draft_proposal", "content": context["draft_proposal"]}
        # 正常执行结束后，交出最终完整答案。
        # 如果没有答案，则交出空字符串，由页面显示提示。
        yield {
            "type": "answer",
            "content": final_answer,
        }


# 请通过 backend.py 的登录接口使用 Agent，避免调试入口绕过身份验证。
