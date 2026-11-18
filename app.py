"""Streamlit 主页面：登录、进入随访任务、工具状态和 Agent 的流式回答。"""

# ==================== 阅读入口：这个文件怎样变成网页 ====================
# 1. 运行 streamlit run app.py 后，是 Streamlit 服务端执行这里的 Python。
#    浏览器负责显示它生成的页面；浏览器不会直接执行这里的 Python / SQL。
#    网页端口：Streamlit 默认使用 8501，FastAPI 通常在另一个进程的 8000 端口。
# 3. 点击按钮、提交表单、发送聊天消息时，Streamlit 通常会从头重新执行本文件。
#    这叫“脚本重跑”，不是每次都重启整个 Python 服务。
#    表单内部的输入先一起收集，提交表单时再交给本轮脚本处理。
# 4. 普通变量在新一轮执行时重新赋值；st.session_state 保存当前会话需要保留的值。
#    它实际保存在 Streamlit 服务端，按浏览器连接的会话区分，
#    不是浏览器 localStorage，也不是 medical.db。浏览器刷新/重连或服务重启时，这些值就不在了
#
# ==================== 从页面到数据库：app.py 为什么没有 medical.db ====================
# 本文件不连接 SQLite。它通过 request_api() 请求 backend.py 提供的 HTTP 接口：
#   页面输入账号密码
#     → request_api("POST", "/auth/login", data={...})
#     → backend.py 的 login(data)
#     → utils/user_store.py 核对姓名、密码和选定角色
#     → get_connection() 打开 DB_PATH 指向的 data/medical.db。
#   后端返回登录凭证和账号信息 → 页面保存到 st.session_state → 重跑显示聊天页。
# medical.db 不仅用于登录：注册写 users，登录写 sessions，鉴权查 sessions/users，
# 退出删除该 sessions 记录，Agent 的医疗工具还会查询任务、出院、随访记录等表。
#
# ==================== 从问题到回答：两种查询来源 ====================
#   页面 POST /chat → 后端根据 token 查身份 → Agent 根据问题选择工具：
#     “我的随访什么时候到期” → 医疗查询工具 → followup_store.py → medical.db。
#     “家庭血压应该怎么测”   → rag_search → RagService → Chroma 向量库。
#   工具结果交给模型整理 → 后端逐行返回 JSON 事件 → 本文件更新状态/回答。
#   注意：这里不是根据关键词直接选择工具，选择动作在后端 Agent 中完成。
#   患者先通过 GET /tasks 查看任务，再通过 GET /tasks/{task_id} 进入指定任务。
#   首页查询只传 mode=query；打开患者或任务后可按相应范围查询。
#   进入可编辑随访才启用 mode=collect，并携带 task_id，后端重新检查归属。
#   页面选择任务不会改变任务状态，也不会提交任何随访记录。
#
# ==================== 向量库什么时候同步 ====================
# 入口在 agent/tools/chat_session.py 的 create_chat_agent(user_id, role, task_id)。
# 后端收到聊天请求，要为会话创建 Agent 时，会检查进程内 knowledge_ready：
#   False → rag.vector_store.load_document() → 成功后改为 True。
# 实际同步函数在 rag/vector_store.py 的 VectorStoreService.load_document()。
# 它按 config/chroma.yml 扫描 data/ 中允许的 TXT/PDF/DOCX，比较文件指纹，
# 把新增/变化的文件切块并向量化，跳过未变化的文件，清理已删除来源的旧块。
# 向量库放在 chroma_db/，不是把 medical.db 的患者记录整体拿去向量化。
# 同一后端进程成功同步后，其他会话首次聊天也会跳过这次同步；同步失败可重试。
# 后端重启后的第一次聊天或报告生成会检查同步；登录时只读取账号和会话。
# 修改知识资料后重启后端，下一次聊天或生成报告时重新检查；前端重跑只刷新显示。
# fastapi dev 的自动重载如果重启了后端进程，也会把这个内存标记重新置为 False。
# 每次真正执行 RAG 检索时，仍会将本次查询文本向量化，这与“资料同步”是两件事。
#
# 下面的例子只用来解释类型，示例令牌不是可用的真实登录凭证。

# NDJSON 每一行都是一条聊天事件；解析后可按 type 更新页面。
import json
# 用 requests 的异常类区分 HTTP 失败和服务无法连接。
import requests
import streamlit as st
# 所有业务读写从这个入口走 HTTP，页面不持有数据库连接。
from utils.api_client import request_api


# 这些名称只用于页面；传给后端时仍使用左边的固定字段名。
DRAFT_TEXT_FIELDS = {
    "systolic_bp": "高压（收缩压，mmHg）",
    "diastolic_bp": "低压（舒张压，mmHg）",
    "measured_at": "什么时候测的？（可保留原话）",
    "symptoms": "最近有没有哪里不舒服？",
    "missed_dose_reason": "如果漏服了，是什么原因？",
    "adverse_effects": "吃药后有没有不舒服？",
}
# 下拉选项保留中文供阅读，提交前会转换成服务端规定的枚举值。
DRAFT_CHOICES = {
    "bp_measurement_status": {"未说明": None, "测过了": "measured", "还没测": "not_measured"},
    "medication_adherence": {
        "未说明": None, "按时服药": "regular", "有漏服": "missed", "没有开药": "not_prescribed",
    },
}


def clear_draft_state():
    """清除草稿页面状态，返回 None；只清内存，不删除后端已保存草稿。"""
    # list() 先复制键名，避免一边遍历字典一边删除导致错误。
    for key in list(st.session_state):
        if key.startswith("draft_"):
            st.session_state.pop(key, None)  # 退出或切换任务时连输入框一起清理。


def draft_display_values(data):
    """输入草稿字段字典，返回输入框显示值；例如 systolic_bp=None 变成空字符串。

    只转换显示格式，不请求网络、不补全缺失值、不改变业务数据。
    """
    values = {}  # 普通字典，例如 {"systolic_bp": "138", "symptoms": ""}。
    for field in DRAFT_TEXT_FIELDS:
        value = data.get(field)  # None 表示没有提供，不当成 0 或“没有症状”。
        # 文本框用空串表示尚未填写，需要与数值 0 区分。
        values[field] = ""
        if value is not None:
            values[field] = str(value)
    # 同一套选项映射既用于显示，又在保存时反向转换成枚举。
    for field, choices in DRAFT_CHOICES.items():
        values[field] = "未说明"  # 空值的唯一默认选项。
        for label, value in choices.items():
            if value == data.get(field):
                values[field] = label  # 例如 regular 对应“按时服药”。
    # 例如 {"systolic_bp":"138","medication_adherence":"按时服药"}。
    return values


def draft_has_changes():
    """比较页面输入和加载时的草稿，返回 bool；只读页面内存，不访问后端。"""
    base = st.session_state.get("draft_base")  # base 是带 revision 的完整接口返回值。
    if base is None:
        return False  # 尚未进入草稿页面，没有可以丢失的修改。
    # 先统一为页面显示的字符串，再比较；否则整数 138 和字符串“138”会被误判。
    values = draft_display_values(base["data"])
    edits = st.session_state.get("draft_edits", {})  # 请求失败后控件可能消失，备份仍在。
    for field, value in values.items():
        # 优先使用现有控件值；控件暂时消失时找回备份，最后才采用已保存值。
        current = st.session_state.get("draft_field_" + field, edits.get(field, value))
        if current != value:
            return True  # 包括患者主动清空已有内容的情况。
    return False


def remember_draft_edits():
    """在请求前备份草稿输入，返回 None；只写页面内存，不保存到数据库。

    请求失败可能让本轮页面提前停止，Streamlit 会清理未渲染控件的键。
    把输入另存普通字典，恢复连接后仍能找回尚未点击保存的文字。
    """
    # base 是首次加载或上次保存成功时的快照，没有它就不知道该与哪版比较。
    base = st.session_state.get("draft_base")
    if base is None:
        return  # 尚未加载过草稿，不创建替代数据。
    # 备份中只放当前任务的表单值，身份和提交版本分别保存。
    edits = st.session_state.get("draft_edits", {})
    for field in draft_display_values(base["data"]):
        # 例如 symptoms 对应 draft_field_symptoms，与下方 text_input 的 key 相同。
        key = "draft_field_" + field
        if key in st.session_state:
            edits[field] = st.session_state[key]  # 保留本次浏览器送来的最新输入，包括主动清空。
    st.session_state["draft_edits"] = edits  # 普通字典不会随控件消失被 Streamlit 清理。


def load_draft_fields(draft):
    """把接口草稿放进页面内存，返回 None；仅在本轮输入框创建前调用。

    输入包含 task_id/revision/data；此函数不写数据库，明确重新加载时会替换本地修改。
    """
    st.session_state["draft_base"] = draft  # 保存版本，之后 PUT 用它防止旧页面覆盖新数据。
    # 持久字段转换成可用于 Streamlit 控件的值，然后再恢复输入。
    values = draft_display_values(draft["data"])
    st.session_state["draft_edits"] = dict(values)  # 明确重新加载时同步替换备份，不能恢复被放弃的旧值。
    for field, value in values.items():
        st.session_state["draft_field_" + field] = value  # 控件 key 保持稳定，普通重跑不重置输入。


def show_risk_notice(content):
    """展示规则提醒，输入含 risk 和 notification_saved 的字典，返回 None。

    例如 risk.level="high" 时显示就医提示；不调用模型、不发请求、不修改病历。
    提醒独立于模型回答，模型失败或清除临时回答时仍须保留。
    """
    risk = content.get("risk", {})  # 缺少规则结果时不推断为安全或低风险。
    message = risk.get("patient_message", "")  # 文案由服务端规则决定，页面不判断血压。
    if risk.get("level") == "high" and message:
        st.error(message)  # 明确红旗提示使用醒目的样式。
    elif risk.get("level") == "attention" and message:
        st.warning(message)  # 需要关注但未命中红旗的规则提示。
    if risk.get("level") in ("high", "attention"):
        # 多条规则可能同时命中，逐条显示，方便患者理解为何收到提醒。
        for reason in risk.get("reasons", []):
            st.write("触发原因：" + reason)  # 保留可解释的规则依据，不展示推理过程。
        # 明确的 False 表示这次写库失败；未传这个标记时不能推断通知投递结果。
        if content.get("notification_reason") == "no_task":
            st.warning("当前没有关联随访任务，本次未发送医生站内提醒。请直接联系医生。")
        elif content.get("notification_saved") is False:
            st.warning("医生站内提醒尚未保存成功，请直接联系医生；不要等待助手回答。")
        elif content.get("notification_saved") is True:
            st.caption("负责医生的站内提醒已保存。")


def show_cached_task_risk(task_id):
    """读取当前任务已收到的规则结果并展示，返回 None；不请求网络、不重新判定风险。

    用于任务归属已确认但草稿暂时读取失败的情况；另一任务的缓存不展示。
    身份或权限失效时由调用处先清缓存，不能拿缓存替代授权。
    """
    displayed = []  # 同一风险只展示一次，但通知保存失败提示仍需保留。
    # 一个缓存来自流式事件，另一个来自读取持久草稿，两者可能描述同一提醒。
    for key in ("draft_risk_event", "draft_loaded_risk"):
        content = st.session_state.get(key)
        if content is None or content.get("task_id") != task_id:
            continue  # 每份缓存明确绑定任务，不依赖页面当前显示的位置。
        risk = content.get("risk", {})
        if risk not in displayed:
            show_risk_notice(content)
            # 记录本轮已经呈现过的规则结果，防止后续缓存重复展示。
            displayed.append(risk)


def show_failed_draft_risk(error, task_id):
    """从草稿保存失败的响应提取风险提醒，返回 None；只保存页面内存，不写业务库。

    输入为 PUT 异常和已授权任务；错误正文可含 risk、notification_saved=False。
    无响应、没有风险字段或权限失效时不推断新风险，也不缓存错误正文中的患者信息。
    """
    response = error.response
    if response is None or response.status_code not in (409, 422, 503):
        return  # 401/403/404 必须走权限处理，不能据此恢复旧提醒。
    if st.session_state.get("user", {}).get("role") != "patient":
        return  # 仅患者任务草稿保存调用这里，不把它扩展成所有接口的通用缓存。
    if st.session_state.get("selected_task", {}).get("task_id") != task_id:
        return  # 保存失败期间任务已切换时，不能混入另一项任务的信息。
    try:
        # 后端可能同时返回 detail 和 risk，保存失败也有独立的风险信息。
        payload = response.json()
    except ValueError:
        return  # HTML 等非 JSON 错误没有可用的规则结果。
    if not isinstance(payload, dict) or not isinstance(payload.get("risk"), dict):
        return  # 兼容原有仅含 detail 的错误，不凭错误类型生成就医结论。
    if payload["risk"].get("level") != "high":
        return
    # 失败提示绑定 task_id，切换任务后不会误显示上一位患者的提醒。
    content = {"task_id": task_id, "risk": payload["risk"], "notification_saved": False}
    st.session_state["draft_risk_event"] = content  # 写库失败仍保留服务端已经检查出的危险信号。
    show_risk_notice(content)  # 当轮立即显示，不能等下一次 GET 或模型回答。


def show_patient_submit(token, task_id, latest, base):
    """让患者确认并提交已保存的草稿，返回 None；仅点击按钮时发送 POST。

    输入为登录凭证、任务编号和草稿接口结果；提交体例如
    {"expected_revision": 2, "confirmed": True}，不发送未保存的表单或指定风险。
    缺项、版本变化或页面有改动时不能提交；失败保留输入，不自动重试。
    """
    values = {}  # 当前显示内容也是确认依据；编辑后必须重新确认。
    for field in draft_display_values(base["data"]):
        # 将当前可见的每个输入都纳入核对，改一个数字也应取消既有勾选。
        values[field] = st.session_state.get("draft_field_" + field)
    signature = {"task_id": task_id, "revision": base["revision"],
                 "server_revision": latest["revision"], "editable": latest["editable"],
                 "values": values}  # 普通字典比较，明确区分服务器版本和当前显示。
    # 比较本次核对依据与上一次；内容、版本、可编辑状态发生变化都算新确认。
    if st.session_state.get("draft_confirm_signature") != signature:
        st.session_state["draft_confirm"] = False  # 内容或版本变动使以前的确认失效。
        st.session_state["draft_confirm_signature"] = signature
    # pop 读取后删除，网络错误引发的重置只需处理一次。
    if st.session_state.pop("draft_reset_confirmation", False):
        st.session_state["draft_confirm"] = False  # 提交失败后，下一轮先取消旧确认。
    # 完整性只是第一关，下一行还会检查未保存改动和版本过期。
    can_submit = latest["editable"] and not latest["missing_fields"]
    if draft_has_changes() or latest["revision"] != base["revision"]:
        can_submit = False  # 页面不把尚未保存的内容说成已经提交。
    if not can_submit:
        st.session_state["draft_confirm"] = False  # 禁用时不保留可能过期的勾选。
    st.divider()
    st.write("确认本次随访信息")
    confirmed = st.checkbox("我已核对上面的信息，确认提交本次随访", key="draft_confirm",
                            disabled=not can_submit)  # 默认不勾选，不代替患者确认。
    # 只有患者主动勾选并点击按钮才发出 POST；普通页面重跑不会提交。
    if st.button("确认提交随访", key="draft_submit", type="primary", disabled=not can_submit or not confirmed):
        try:
            result = request_api("POST", f"/tasks/{task_id}/submit", token=token, data={
                "expected_revision": base["revision"], "confirmed": True,
            }).json()  # 服务端独立检查归属、版本、缺项和重复请求。
        except requests.RequestException as error:
            # 保留输入，但下一轮要求患者重新核对，避免误用之前的确认。
            st.session_state["draft_reset_confirmation"] = True
            show_request_error(error)  # 输入保持原样，不把错误当成成功。
            st.warning("请重新加载确认提交结果；连接中断时，后端可能已经保存，请勿反复点击。")
        else:
            st.session_state["draft_submit_notice"] = result  # 下一轮展示成功并读取实时状态。
            st.session_state["draft_reload"] = True
            # 接口已接收正式记录，下一轮读取实际处理进度，不让患者再次触发模型。
            st.rerun()


def display_report(content):
    """展示服务端允许当前账号阅读的报告字典；直接展开引用可查看生成时的原文。"""
    if content.get("legacy_report"):
        st.write(content["summary"])  # 旧报告以 Markdown 保存，保留标题和原有段落。
    else:
        st.text(content["summary"])  # 保留每个确认事实之间的换行，患者原话中的符号按原文显示。
    # 新旧规则使用过不同的英文代码，页面分别翻译，保留历史语义。
    levels = {"high": "已触发升级", "unassessed": "待评估", "no_alert": "未触发预设告警",
              "no_flag": "未标记异常", "review": "需要复核"}
    level = content.get("risk_level", "unassessed")
    st.write("规则风险", levels.get(level, level))
    # 没有关注项时列表为空，页面不会自行补写新的风险判断。
    for point in content.get("abnormal_points", []):
        st.write("需关注", point)
    for recommendation in content.get("recommendations", []):
        st.write("参考建议", recommendation)  # 文字末尾的〔1〕对应第一条引用。
    # enumerate 从 1 编号，使标题与建议正文的〔1〕、〔2〕一一对应。
    for number, source in enumerate(content.get("sources", []), 1):
        with st.expander(f"〔{number}〕{source['source']} · 查看原文"):
            st.text(source["excerpt"])  # 使用生成时保存的片段，即使知识库变化也能核对。
            st.caption("片段编号 " + source["chunk_id"])


def show_report_workflow(token, task_id, doctor=False):
    """展示一个任务的报告进度。GET 只读，按钮触发生成、审核或发布，失败允许重试。"""
    # 同一任务的 GET、生成和审核操作共用此前缀，例如 /tasks/F001/report。
    path = f"/tasks/{task_id}/report"
    try:
        view = request_api("GET", path, token=token).json()  # 每轮重新授权，页面不缓存报告正文。
        if st.button("刷新报告状态", key="refresh_report"):
            st.rerun()
        # 记录已提交而任务仍在 submitted，说明草稿生成尚未完成。
        if view["task_status"] == "submitted":
            st.info("随访记录已保存，正在等待整理报告。")
            if doctor and st.button("生成报告草稿", key="generate_report"):
                with st.spinner("正在检索资料并生成报告……"):
                    # 只发送当前正式记录编号，即使重试多次也不能创建第二条随访记录。
                    request_api("POST", path + "/generate", token=token, data={"record_id": view["record_id"]})
                st.rerun()  # 成功后重新读取；页面重复刷新不会再次生成。
        # 患者和医生使用不同的页面控件，后端仍对每次 GET 与 POST 独立鉴权。
        if not doctor:
            if view.get("note"):
                st.write("医生意见", view["note"])  # 退回只显示补充要求；已发布可查看最终审核意见。
            elif view["task_status"] == "needs_info":
                st.info("医生请你补充本次随访信息，当前没有可查看的具体意见。")
            # 仅在接口确实返回 content 时显示正文。发布前患者响应里没有这个字段。
            if "content" in view:
                st.subheader("我的最终随访报告")
                st.write("本次负责医生", report_doctor_label(view.get("responsible_doctor"),
                                                        view.get("responsible_doctor_source")))
                if view.get("reviewer"):
                    st.caption("审核医生 · " + doctor_label(view["reviewer"]))
                if view.get("reviewed_at"):
                    st.caption("审核时间　" + view["reviewed_at"])
                display_report(view["content"])
            # 已通过但尚未发布时也走等待分支，医生仍需点击发布按钮。
            elif view["task_status"] == "in_review":
                st.info("报告正在等待医生审核或发布。")
            elif view["task_status"] == "completed":
                st.info("本次随访已结束，目前没有可查看的已发布报告。")
            return
        # 医生可以在报告旁核对同一个 record_id 的提交版本和原始读数。
        if view.get("record"):
            with st.expander("本次提交的原始记录", expanded=True):
                show_record(view["record"])  # 使用患者确认时的正式版本，中文展示原始信息。
        if "content" not in view:
            if view["task_status"] in EDITABLE_STATES:
                st.info("正在等待患者填写并确认本次随访信息。")
            return  # 模型失败时仍可看原始记录、重新生成。
        report_states = {"pending_review": "待审核报告", "approved": "待发布报告",
                         "returned": "已退回的报告", "published": "已发布报告"}
        st.subheader(report_states.get(view.get("status"), "随访报告"))
        if view.get("reviewer"):
            st.caption("审核医生　" + doctor_label(view["reviewer"]))
        display_report(view["content"])
        if view.get("note"):
            st.write("已保存的审核意见", view["note"])
        # expected_version 是打开此报告时读到的版本，不在页面自行加一。
        payload = {"report_id": view["report_id"], "expected_version": view["version"]}
        # key 包含账号、报告和版本。换患者或另一位医生登录时，不能沿用之前的审核输入。
        key = f"{st.session_state['user']['user_id']}_{view['report_id']}_{view['version']}"
        if view["status"] == "pending_review":
            # 四种动作中的前三种用于审核；发布只会在服务端返回 approved 后出现。
            options = {"通过": "approve", "修改后通过": "edit", "退回补充": "return"}
            choice = st.radio("审核操作", list(options), key="choice_" + key)
            payload["action"] = options[choice]  # 例如“退回补充”提交给 API 的 action 是 return。
            if choice == "修改后通过":
                # 编辑器初值取数据库内原摘要，医生可直接更正和补充本次情况。
                payload["summary"] = st.text_area("修改后的摘要", view["content"]["summary"],
                                                  max_chars=8000, key="summary_" + key)
                # 一行代表一个关注项；写回列表保留条目界限，空行会略过。
                points = st.text_area("需关注情况（每行一项）", "\n".join(view["content"]["abnormal_points"]),
                                      key="points_" + key)
                payload["abnormal_points"] = []
                for line in points.splitlines():
                    if line.strip():
                        payload["abnormal_points"].append(line.strip())
                payload["keep_sources"] = []  # 保留建议及其对应引用一起提交，不能改写知识库原文。
                for number, source in enumerate(view["content"]["sources"], 1):
                    if st.checkbox(f"保留建议〔{number}〕 {source['quote']}", value=True,
                                   key=f"keep_{key}_{number}"):
                        payload["keep_sources"].append(number)
            payload["note"] = st.text_area("审核意见（退回时请写明需要补充什么）",
                                           max_chars=2000, key="note_" + key)
            if st.button("确认审核", key="review_" + key, type="primary"):
                request_api("POST", path + "/review", token=token, data=payload)
                st.rerun()
        elif view["status"] == "approved":
            st.info("审核已通过，发布后患者即可查看。")
            if st.button("发布给患者", key="publish_" + key, type="primary"):
                payload["action"] = "publish"
                request_api("POST", path + "/review", token=token, data=payload)
                st.rerun()
    except requests.RequestException as error:
        show_request_error(error)  # 保留可修改的输入；409 时刷新版本，503 时可重新生成。


def show_patient_draft(token, task_id):
    """读取、展示并修改当前患者草稿，返回 None；通过 GET/PUT 读写 task_drafts。

    输入是登录凭证与任务编号。保存发送 expected_revision 和完整 data 字典。
    无权访问或连接失败时停止当前页面；冲突保留本地输入，不自动重试覆盖。
    旧记录含义不够明确时显示后端的 notes 提示，请患者核对，不自动替患者选择。
    """
    path = f"/tasks/{task_id}/draft"  # 例如 /tasks/F001/draft，任务归属仍由后端校验。
    try:
        latest = request_api("GET", path, token=token).json()  # 重进页面可恢复数据库里的草稿。
    except requests.RequestException as error:
        # 登录失效或归属改变都要丢弃缓存；暂时断线则允许恢复之前的草稿编辑。
        if error.response is not None and error.response.status_code in (401, 403, 404):
            clear_task_selection()  # 权限失效后不能继续展示旧草稿。
        else:
            show_cached_task_risk(task_id)  # 数据库暂不可读时，已收到的就医提醒仍然有效。
        show_request_error(error)
        if st.button("重新加载草稿", key="draft_retry"):
            st.rerun()
        st.stop()  # 请求失败不等于空草稿，也不开放可能覆盖旧数据的编辑。

    # latest 是本轮 GET 返回的版本，base 是用户开始编辑时的版本。
    base = st.session_state.get("draft_base")
    if base is None or base["task_id"] != task_id or st.session_state.pop("draft_reload", False):
        load_draft_fields(latest)  # 首次进入或明确保存成功后，采用服务器返回值。
    elif latest["revision"] != base["revision"] and not draft_has_changes():
        load_draft_fields(latest)  # 没有手工改动时，安全展示 Agent 或另一个页面保存的新版本。
    base = st.session_state["draft_base"]
    # 如果控件在上次失败时被移除，可从普通会话字典还原输入。
    for field, value in st.session_state.get("draft_edits", {}).items():
        key = "draft_field_" + field
        if key not in st.session_state:
            st.session_state[key] = value  # 仅恢复请求失败期间被清掉的控件，不覆盖本轮新输入。
    if latest.get("risk", {}).get("level") == "high":
        # 及时记录刚读到的高风险，以便下次网络断开时仍能呈现必要提醒。
        st.session_state["draft_loaded_risk"] = {"task_id": task_id, "risk": latest["risk"]}
    else:
        st.session_state.pop("draft_loaded_risk", None)  # 当前已读结果更新，历史提醒仍由 alerts 保存。
    show_risk_notice({"risk": latest.get("risk", {})})  # 规则结果不是模型回答，不会被聊天 reset 清空。
    for alert in latest.get("alerts", []):  # 读取持久通知，重新登录后仍能看到已触发提醒。
        st.warning("此前触发的提醒：" + alert["message"])
        st.caption("提醒时间：" + alert["created_at"])
    recent_risk = st.session_state.get("draft_risk_event")  # 通知保存失败时也保留刚收到的就医提示。
    # 保留刚才未保存成功的通知状态，即使内容与规则快照相同也要显示失败提示。
    if recent_risk and recent_risk.get("task_id") == task_id and (recent_risk.get("risk") != latest.get("risk") or recent_risk.get("notification_saved") is False):
        show_risk_notice(recent_risk)
    # pop 让提交回执只显示一次，刷新页面不会触发第二次提交。
    submitted = st.session_state.pop("draft_submit_notice", None)
    if submitted is not None:
        st.success("本次随访已提交，记录已保存。")  # 不把提交说成医生已经审核或发布报告。
    for note in latest.get("notes", []):  # 例如旧记录只写“已服药”，尚不能确定是否按时。
        st.info(note)  # 只展示核对提示，输入框仍保持未说明，直到患者自己确认。
    if st.session_state.pop("draft_saved_notice", False):
        st.success("草稿已保存，可继续补充或修改。")
    # 如果后端已经保存新版本，提示核对；当前页面的个人编辑先保留。
    if latest["revision"] != base["revision"]:
        st.warning("草稿已有更新，当前输入仍保留。请先核对，再放弃当前修改并重新加载，避免覆盖新内容。")
    if draft_has_changes():
        st.caption("当前有尚未保存的修改。可以先问助手，回到这里继续；采集和提交前请先保存。")
    # 从采集对话转到这张核对单时，显示同一版本，完整后可以直接核对提交。
    with st.expander("核对随访信息 · 手工修改或确认提交", expanded=not latest["missing_fields"] or draft_has_changes()):
        st.caption("还没说到的信息可以留空。修改后请先保存，再确认提交。")
        if st.button("放弃当前修改并重新加载", key="draft_discard"):
            load_draft_fields(latest)  # 用户已明确选择放弃，才替换输入及版本。
            st.rerun()
        if latest["missing_fields"]:
            # 缺项由服务器按“已测量”“有漏服”等条件计算，页面直接列给患者。
            st.write("已保存草稿还需补充：" + "、".join(latest["missing_fields"]))
        editable = latest["editable"]  # 即使控件不可编辑，后端保存接口仍要独立校验状态。
        if not editable:
            st.info("这项任务当前不能修改草稿，下面仅供查看。")
        st.selectbox("这次测过血压了吗？", list(DRAFT_CHOICES["bp_measurement_status"]),
                     key="draft_field_bp_measurement_status", disabled=not editable)
        for field, label in DRAFT_TEXT_FIELDS.items():
            # 每个文本框用字段名作 key，保存时可以精确取回相应输入。
            st.text_input(label, key="draft_field_" + field, disabled=not editable)
        st.selectbox("最近服药情况", list(DRAFT_CHOICES["medication_adherence"]),
                     key="draft_field_medication_adherence", disabled=not editable)
        if st.button("保存草稿", key="draft_save", disabled=not editable):
            data = {}  # 空输入转换为 None；接口 JSON 中对应 null。
            try:
                for field in DRAFT_TEXT_FIELDS:
                    # 去掉首尾空格再判断是否为空，患者可以有意清空之前填过的字段。
                    value = st.session_state["draft_field_" + field].strip()
                    data[field] = None
                    if value:
                        data[field] = value  # 时间、症状和原因保留文字，不自行推算日期或医学结论。
                        if field in ("systolic_bp", "diastolic_bp"):
                            data[field] = int(value)  # 无法转成整数时明确报错，不写成 0。
                for field, choices in DRAFT_CHOICES.items():
                    # 例如把页面选择“还没测”转换为 not_measured。
                    data[field] = choices[st.session_state["draft_field_" + field]]
            except ValueError:
                st.error("高压和低压请填整数；还不知道时留空即可。")
                return  # 输入仍保留在控件中，患者可以修改。
            try:
                # PUT 提交完整可编辑表单；与聊天只提议本轮提到的字段有所区别。
                request_api("PUT", path, token=token, data={
                    "expected_revision": base["revision"], "data": data,
                })  # 例如 revision=2；如果服务器已是 3，必须拒绝覆盖。
            except requests.RequestException as error:
                if error.response is not None and error.response.status_code in (401, 403, 404):
                    clear_task_selection()  # 写入阶段也可能撤权，立即清除旧草稿和风险缓存。
                    show_request_error(error)
                    st.stop()
                show_failed_draft_risk(error, task_id)  # 规则提醒独立于写库成功，不被保存失败吞掉。
                show_request_error(error)
                # 409 提醒用户重新核对版本，不能自动用过时输入覆盖服务器值。
                if error.response is not None and error.response.status_code == 409:
                    st.warning("未覆盖服务器草稿。当前输入仍保留，请核对后重新加载。")
            else:
                st.session_state["draft_reload"] = True  # 下一轮控件创建前再更新，避免修改已创建的控件。
                st.session_state["draft_saved_notice"] = True
                st.rerun()
        show_patient_submit(token, task_id, latest, base)  # 提交与保存草稿是两个明确的操作。



def clear_login():
    """清除页面登录信息。无输入、返回 None；不删除数据库账号或医疗记录。"""
    clear_draft_state()  # 清除草稿及输入框缓存，防止下一个账号看到旧内容。
    # user={...} 保存账号字典；token="示例令牌" 保存登录凭证字符串；message=[...] 保存聊天消息列表。
    # agent 是旧版页面留下的键。当前版本的 Agent 对象在后端，本页面不再创建它。
    # 退出或登录过期时，一起清掉身份、令牌和页面里的聊天历史。
    for key in ("user", "token", "message", "agent", "selected_task", "selected_patient",
                "workspace", "chat_scope", "saved_edits", "scope_risk", "patient_filter", "patient_search",
                "doctor_section", "confirm_logout", "logout_requested"):
        # key 是每轮循环取出的 str，例如第一次为 "user"。
        # pop(key, None) = 删除并返回这个键的旧值；键不存在就返回 None，不报错。
        # 这里没有接收 pop 的返回值，因为我们只想删除，不需要旧内容。
        st.session_state.pop(key, None)

    # 这个函数不会删除数据库账号，也不会使后端 token 自动失效。
    # 真正“退出登录”还要调用 POST /auth/logout，见下面侧栏按钮的处理顺序。


def show_request_error(error):
    """将 requests 异常转成中文页面提示；401 清理登录状态，返回 None，不发送新请求。"""
    # 【输入】error 是 requests.RequestException 或其子类的异常对象，不是字符串。
    # 例如 HTTPError 表示后端返回 401/409/503 等错误；ConnectionError 表示连接失败。
    # 使用例子：except requests.RequestException as error: show_request_error(error)。
    # 【输出】返回 None；副作用是在页面显示错误，必要时清掉当前登录状态。
    # error.response：可能是 requests.Response，也可能为 None。
    # Response 可以理解为“HTTP 回复对象”，包含状态码、响应头、正文等。
    # HTTPError 有 response 属性；连接不上服务时，这个属性通常是 None。
    response = error.response
    if response is None:
        # 没拿到 HTTP 回复，就没有后端提供的 detail，只能显示本地准备的提示。
        # st.error() 创建红色错误提示；这里不接收它返回的页面元素对象。
        st.error("暂时无法连接后端，请确认 FastAPI 已启动后重试。")
        # return 只结束 show_request_error，不等于停止整个 app.py。
        return

    # 后端把可展示的错误写在 detail 中，例如“用户名已存在”。
    try:
        # response.json() 将回复正文解析成 Python 值，本项目的错误正文是 dict：
        # {"detail": "用户名已存在。"}。json() 不是重新发送 HTTP 请求。
        # dict.get("detail", 默认值) 取出提示文字；没这个键时使用默认值。
        detail = response.json().get("detail", "请求失败，请稍后重试。")
    except ValueError:
        # 后端异常时可能返回 HTML/普通文本，无法作为 JSON 解析，进入这里。
        detail = "后端返回了异常内容，请检查后端日志。"
    if not isinstance(detail, str):
        # isinstance(x, str) 检查 x 是否为字符串。
        # FastAPI 的 422 校验错误可能返回 detail=[{...}, ...]，即 list，
        # 不直接把内部校验结构堆给用户看，而是换成容易理解的提示。
        detail = "输入格式不正确，请检查填写的内容、数值和长度。"
    st.error(detail)

    # 登录过期后只清理本地状态，下一次脚本重跑便会回到登录页。
    # status_code 是 int：401 表示身份校验失败，409 表示冲突，503 表示服务不可用。
    # 这里不自动调用 st.rerun()，让用户先看到错误；下一次脚本重跑时显示登录区域。
    if response.status_code == 401:
        clear_login()


def clear_task_selection():
    """返回任务列表前清理当前选择。无输入、返回 None；只修改页面会话内存。"""
    clear_draft_state()  # 切换任务不沿用上一项任务的草稿和版本。
    # selected_task 是任务字典，例如 {"task_id": "F001", "task_status": "pending", ...}。
    st.session_state.pop("selected_task", None)
    # 切换任务不沿用屏幕上的旧对话，避免把前一项任务的问题显示到新任务下。
    st.session_state["message"] = []
    st.session_state.pop("chat_scope", None)
    st.session_state.pop("scope_risk", None)


def show_chat(token, mode, task_id=None, patient_id=None, access_scope=None):
    """按服务端范围发送一轮对话，返回 None；只有 collect 可能触发后端保存草稿。"""
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
            # 两个占位让“正在检索...”和模型回答分开，避免每个小片段新增一个气泡。
            status_box = st.empty()
            answer_box = st.empty()
            # answer: str，当前累计回答，例如 "家庭" → "家庭血压" → "家庭血压测量..."。
            answer = ""
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
                with request_api("POST", "/chat", token=token, data=chat_data, stream=True) as response:
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

                        # event_type 是 str；draft/risk 事件正文是字典，其他事件正文是 str。
                        # 这几个字符串是前后端约定的事件名，不是 Streamlit 自动规定的。
                        # 一轮可能有多次工具调用，因此不保证每种事件都只出现一次。
                        # 工具状态与回答分别放置，例如“正在检索高血压知识库”。
                        if event_type == "status":
                            # 例：{"type":"status","content":"正在检索高血压知识库……"}。
                            # 只更新状态位置，不把状态文字拼进最终回答。
                            status_box.info(content)
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
                st.session_state["message"].append({"role": "assistant", "content": answer})
                if draft_updated:
                    st.session_state["draft_saved_notice"] = True
                    st.rerun()  # 新一轮只重画历史，不重复发送这条消息。
            else:
                # 发生错误时删除屏幕上的半成品，但保留错误提示和用户已经发送的问题。
                answer_box.empty()
                if not failed:
                    # 没有明确报错但连接结束了，仍可能缺少最终 answer，不能当作成功。
                    status_box.error("未收到完整回答，请重试。")

TASK_STATES = {
    "pending": "待随访", "in_progress": "进行中", "needs_info": "需补充",
    "submitted": "已提交", "in_review": "审核中", "completed": "已完成",
}
EDITABLE_STATES = ("pending", "in_progress", "needs_info")


def task_name(task):
    """输入接口任务，返回患者易读的标题；只转换约定的出院后天数。"""
    name = task.get("task_type") or "随访"
    if name.startswith("D") and name[1:].isdigit():
        return "出院后第 " + name[1:] + " 天随访"
    return name


def doctor_label(doctor):
    """输入医生公开资料或 None，返回姓名、科室和职称；不请求网络。"""
    if not doctor:
        return "暂未提供"
    parts = []
    for field in ("name", "department", "title"):
        if doctor.get(field):
            parts.append(doctor[field])
    return " · ".join(parts) or "暂未提供"


def task_doctor(task):
    """从当前任务摘要取负责医生，返回可显示的文字；不借用主管医生填空。"""
    return doctor_label({"name": task.get("doctor_name"), "department": task.get("doctor_department"),
                         "title": task.get("doctor_title")})


def report_doctor_label(doctor, source):
    """根据报告署名来源显示医生。缺少历史依据时返回说明，不猜测当次负责人。"""
    if source == "unverified":
        return "当次负责医生未核实"
    label = doctor_label(doctor)
    if source == "current_task":
        label += "（据当前任务分配）"
    return label


def show_record(record):
    """输入已授权的正式记录，用中文显示读数、症状和提交版本；不请求网络或写库。"""
    data = record.get("data") or {}
    high, low = record.get("systolic_bp"), record.get("diastolic_bp")
    if high is not None or low is not None:
        st.write(f"血压　{high if high is not None else '—'} / {low if low is not None else '—'} mmHg")
    else:
        measure = record.get("bp_measurement_status")
        st.write("血压　" + {"not_measured": "未测量", "measured": "已测量，未记录读数"}.get(measure, "未提供"))
    st.caption(f"提交时间　{record.get('submitted_at') or '未提供'}　·　提交版本 {record.get('revision') or '未提供'}")
    # 优先使用当前记录明确保存的值；缺失信息如实留空。
    for field, label in (("measured_at", "测量时间"), ("heart_rate", "心率"), ("symptoms", "症状"),
                         ("medication_adherence", "服药情况"), ("missed_dose_reason", "漏服原因"),
                         ("adverse_effects", "用药后不适"), ("risk_reason", "规则提示依据")):
        value = data.get(field)
        if value is None:
            value = record.get(field)
        if field == "medication_adherence" and value is not None:
            # 历史字段的 True/False 仅说明是否服过药，不能据此判断是否规律。
            labels = {"regular": "按时服药", "missed": "有漏服", "not_prescribed": "没有开药",
                      "True": "已服药，是否按时待核对", "False": "未服药，原因待核对"}
            value = labels.get(str(value), value)
        if value is not None and value != "":
            st.write(label, value)


def get_view(token, path):
    """用登录凭证读取 JSON，失败展示真实错误并返回 None；空列表表示读取成功。"""
    try:
        return request_api("GET", path, token=token).json()
    except requests.RequestException as error:
        if error.response is not None and error.response.status_code in (403, 404):
            if path.startswith("/tasks/"):
                task_id = path.split("/")[2]
                st.session_state.get("saved_edits", {}).pop(task_id, None)
            clear_task_selection()  # 清理失去授权的当前视图，其他已暂存任务仍需各自重新授权才能恢复。
            st.session_state.pop("selected_patient", None)
        show_request_error(error)
        if error.response is not None and error.response.status_code == 401:
            st.stop()  # 身份过期，本轮立即停止，不能继续绘制其他模块。
        return None


def go_home():
    """返回工作区首页，按任务保留尚未保存的表单内容；只改变页面会话。"""
    if draft_has_changes():
        remember_draft_edits()
        base = st.session_state["draft_base"]  # 保留开始编辑时的版本，恢复后仍能检查冲突。
        edits = st.session_state.setdefault("saved_edits", {})
        edits[base["task_id"]] = {"base": base, "edits": dict(st.session_state["draft_edits"])}
    clear_task_selection()
    st.session_state.pop("selected_patient", None)
    st.session_state["workspace"] = "home"
    st.rerun()


def open_task(token, task_id):
    """用户点选时通过 GET 重查归属；成功才绑定任务、恢复该任务的未保存修改。"""
    task = get_view(token, f"/tasks/{task_id}")
    if task is None:
        st.stop()
    clear_task_selection()
    st.session_state["selected_task"] = task
    cached = st.session_state.get("saved_edits", {}).pop(task_id, None)
    if cached and task["task_status"] in EDITABLE_STATES:
        st.session_state["draft_base"] = cached["base"]
        st.session_state["draft_edits"] = cached["edits"]
    st.session_state["workspace"] = "task"
    st.rerun()


def task_action(task):
    """按服务端状态和已保存草稿决定按钮文字；不修改状态或自动提交。"""
    status = task["task_status"]
    if status == "needs_info":
        return "补充随访信息"
    if status in ("submitted", "in_review"):
        return "查看审核进度"
    if status == "completed":
        if task.get("report_status") == "published":
            return "阅读报告"
        return "查看随访"
    if task.get("draft_needs_clarification") is not None or task.get("draft_revision") or status == "in_progress":
        return "继续随访"
    return "开始随访"


def task_priority(task):
    """给患者首页排序，先补充退回资料，再继续草稿或开始最近任务；只返回排序键。"""
    rank = 3
    if task["task_status"] == "needs_info":
        rank = 0
    elif task["task_status"] in EDITABLE_STATES:
        rank = 2
        if task.get("draft_needs_clarification") is not None or task.get("draft_revision") or task["task_status"] == "in_progress":
            rank = 1
    return rank, task.get("followup_date") or "", task["task_id"]


def task_card(token, task, primary=False, doctor=False):
    """显示一项随访与对应操作；用户点击后才请求任务详情。"""
    with st.container(border=True):
        description, action = st.columns([3, 1])
        with description:
            st.markdown("**" + task_name(task) + "**")
            st.caption(f"{task['followup_date']}　·　{TASK_STATES.get(task['task_status'], '状态待确认')}")
            if doctor:
                st.write(task.get("patient_name") or "随访任务")
                if task.get("report_status") == "approved":
                    st.caption("审核已通过，等待发布给患者。")
            elif task.get("report_status") == "published":
                st.write("本次负责医生　" + report_doctor_label(task.get("report_doctor"), task.get("report_doctor_source")))
            else:
                st.write("本次负责医生　" + task_doctor(task))
            if task["task_status"] == "needs_info":
                if task.get("review_note"):
                    st.write("医生请你补充", task["review_note"])
                else:
                    st.caption("本次随访需要补充，目前没有可查看的具体意见。")
            elif not doctor and task["task_status"] in ("submitted", "in_review"):
                if task.get("submitted_at"):
                    st.caption("提交时间　" + task["submitted_at"])
                if task["task_status"] == "submitted":
                    st.write("记录已保存，正在等待整理报告。")
                else:
                    st.write("报告已进入审核流程，发布后可以阅读。")
            elif task.get("draft_needs_clarification") is not None and task["task_status"] in EDITABLE_STATES:
                progress = "已保存，可继续填写"
                if task.get("draft_needs_clarification") == 0:
                    progress = "已保存，请核对并确认提交"
                st.caption(progress)
            if task["task_status"] == "completed" and task.get("report_date"):
                st.caption("报告日期　" + task["report_date"])
        with action:
            label = task_action(task)
            if doctor:
                label = "查看本次随访"
                if task.get("report_status") == "pending_review":
                    label = "审核报告"
                elif task.get("report_status") == "approved":
                    label = "前往发布"
                elif task["task_status"] == "submitted":
                    label = "查看生成进度"
            style = "secondary"
            if primary:
                style = "primary"
            if st.button(label, key="enter_task_" + task["task_id"], type=style):
                open_task(token, task["task_id"])


def show_patient_home(token):
    """读取本人档案与任务；首页突出当前动作，按日期另列已发布报告。"""
    st.title("我的随访")
    profiles = get_view(token, "/patients")
    if profiles:
        st.write("主管医生　" + doctor_label(profiles[0].get("primary_doctor")))
    elif profiles == []:
        st.caption("该账号尚未关联患者档案。")
    tasks = get_view(token, "/tasks")
    if tasks is None:
        if st.button("重新加载任务", key="reload_tasks"):
            st.rerun()
        return
    upcoming, reports, closed = [], [], []
    for task in tasks:
        if task["task_status"] == "completed" and task.get("report_status") == "published":
            reports.append(task)
        elif task["task_status"] == "completed":
            closed.append(task)  # 某些旧任务已结束，但目前没有可以阅读的已发布报告。
        else:
            upcoming.append(task)
    upcoming.sort(key=task_priority)
    st.subheader("下一步")
    if upcoming:
        task_card(token, upcoming[0], primary=True)
        if len(upcoming) > 1:
            with st.expander("其他随访安排", expanded=False):
                for task in upcoming[1:]:
                    task_card(token, task)
    elif tasks:
        st.info("当前随访均已完成，可以阅读下方的报告。")
    else:
        st.info("目前没有随访安排。你也可以从侧栏问助手。")
    st.subheader("历次随访报告")
    if not reports:
        st.caption("医生发布后，可在这里阅读报告。")
    # 按报告日期倒序；旧报告缺少日期时，使用本次随访日期。
    def report_order(task):
        """输入任务摘要，返回历史报告排序日期。"""
        return task.get("report_date") or task.get("followup_date") or ""
    for task in sorted(reports, key=report_order, reverse=True):
        task_card(token, task)
    if closed:
        with st.expander("其他已结束的随访"):
            for task in closed:
                task_card(token, task)


def open_patient(patient_id):
    """从医生名册选择一位患者；详情页会再次请求后端核对权限。"""
    clear_task_selection()
    st.session_state["selected_patient"] = patient_id
    st.session_state["workspace"] = "patient"
    st.rerun()


def show_physician_home(token):
    """显示授权名册、待办和风险通知；当前栏目独立读取，失败可以切换其他栏目。"""
    st.title("医生工作台")
    section = st.radio("工作台栏目", ["我的患者", "待办随访", "风险通知"], horizontal=True, key="doctor_section")
    if section == "待办随访":
        tasks = get_view(token, "/tasks")
        if tasks is None:
            return
        count = 0
        for task in tasks:
            if task["task_status"] in ("submitted", "in_review"):
                count += 1
                task_card(token, task, doctor=True)
        if not count:
            st.info("当前没有待处理的随访。")
        return
    if section == "风险通知":
        if st.button("刷新风险通知", key="reload_notifications"):
            st.rerun()
        notices = get_view(token, "/notifications")
        if notices == []:
            st.info("当前没有风险通知。")
        for notice in notices or []:
            with st.container(border=True):
                st.markdown("**" + notice["patient_name"] + "**")
                st.warning(notice["message"])
                st.caption(notice["created_at"])
                if st.button("查看相关随访", key="notice_" + str(notice["id"])):
                    open_task(token, notice["task_id"])
        return
    patients = get_view(token, "/patients")
    if patients is None:
        return
    search_col, filter_col = st.columns([1, 2])
    search = search_col.text_input("搜索患者姓名", key="patient_search").strip()
    categories = ["全部", "待我处理", "待患者填写", "已完成随访", "暂无任务"]
    category = filter_col.selectbox("按随访状态筛选", categories, key="patient_filter")
    flags = {"待我处理": "has_review", "待患者填写": "has_to_fill", "已完成随访": "has_completed"}
    count = 0
    for patient in patients:
        if search and search not in patient["patient_name"]:
            continue
        if category in flags and not patient[flags[category]]:
            continue
        if category == "暂无任务" and patient["task_count"]:
            continue
        count += 1
        with st.container(border=True):
            summary, status = st.columns([3, 2])
            # 姓名按钮用稳定患者编号作 key；同名不会打开对方的档案。
            if summary.button(patient["patient_name"], key="patient_" + patient["patient_id"],
                              help="打开患者档案", icon=":material/person:"):
                open_patient(patient["patient_id"])
            # 一位患者可由多位医生分次随访；说明当前登录医生为什么能查看此人。
            relation = "我负责其中的随访"
            if patient["is_primary"]:
                relation = "我是主管医生"
            summary.caption(relation + "　·　" + patient_summary(patient))
            states = []
            for title, flag in flags.items():
                if patient[flag]:
                    states.append(title)
            status.write(" · ".join(states) or "暂无任务")
            if patient.get("latest_followup_date"):
                status.caption("最近随访安排　" + patient["latest_followup_date"])
    if not count:
        st.info("当前没有符合条件的患者。")


def patient_summary(patient):
    """用档案已有字段显示性别、年龄和诊断；不推算未知资料。"""
    parts = []
    if patient.get("gender"):
        gender = str(patient["gender"])
        labels = {"male": "男", "female": "女", "unknown": "性别未记录"}
        parts.append(labels.get(gender, gender))
    if patient.get("age") is not None:
        parts.append(str(patient["age"]) + " 岁")
    if patient.get("diagnosis"):
        parts.append(patient["diagnosis"])
    return " · ".join(parts) or "基本资料暂未提供"


def show_discharge_medications(raw):
    """显示出院时记录的处方明细；输入可能为 JSON 列表、旧版文字或空值，不推算剂量。"""
    st.markdown("**出院用药**")
    if not raw:
        st.caption("未提供")
        return
    medications = raw
    if isinstance(raw, str):
        try:
            medications = json.loads(raw)  # 新格式通常是一组药品，每种药保存独立的规格和用法。
        except ValueError:
            st.write(raw)  # 旧资料可能是一段医嘱文字，直接显示原文。
            return
    if not isinstance(medications, list):
        st.write(raw)
        return
    if not medications:
        st.caption("未提供")
    for medicine in medications:
        if not isinstance(medicine, dict):
            st.write(medicine)
            continue
        st.write(medicine.get("name") or "药名未提供")
        parts = []
        for key, label in (("strength", "规格"), ("dose", "每次用量"), ("route", "途径"), ("frequency", "频次")):
            if medicine.get(key):
                parts.append(label + " " + str(medicine[key]))
        if medicine.get("quantity") is not None:
            parts.append("数量 " + str(medicine["quantity"]) + str(medicine.get("quantity_unit") or ""))
        if parts:
            st.caption("　·　".join(parts))
        if medicine.get("note"):
            st.caption(medicine["note"])


def show_patient_profile(token, detail):
    """展示已获授权的患者档案、出院信息、随访任务与正式记录；不写入数据。"""
    patient = detail["patient"]
    st.title(patient["patient_name"] + "的档案")
    st.write(patient_summary(patient))
    st.write("主管医生　" + doctor_label(patient.get("primary_doctor")))
    st.write("既往病史", patient.get("medical_history") or "未提供")
    st.subheader("随访安排与报告")
    for task in detail["tasks"]:
        task_card(token, task, doctor=True)
    if not detail["tasks"]:
        st.caption("目前没有可查看的随访任务。")
    for title, rows in (("出院资料", detail["discharges"]), ("正式随访记录", detail["records"])):
        with st.expander(title + f"（{len(rows)}）"):
            if not rows:
                st.caption("暂无记录。")
            for row in rows:
                if title == "正式随访记录":
                    show_record(row)
                else:
                    st.markdown("**" + str(row.get("discharge_date") or "日期未提供") + " 出院**")
                    for key, label in (("discharge_diagnosis", "出院诊断"), ("discharge_instructions", "出院医嘱")):
                        st.write(label, row.get(key) or "未提供")
                    show_discharge_medications(row.get("discharge_medications"))
                    if row.get("baseline_bp_systolic") is not None or row.get("baseline_bp_diastolic") is not None:
                        st.write("出院基线血压", str(row.get("baseline_bp_systolic") or "—") + " / "
                                 + str(row.get("baseline_bp_diastolic") or "—") + " mmHg")
                    if row.get("baseline_heart_rate") is not None:
                        st.write("出院基线心率", str(row["baseline_heart_rate"]) + " 次/分")
                st.divider()


def show_task_page(token, task, doctor):
    """呈现本次随访的当前步骤；采集、人工修改与报告审核分别由明确操作触发。"""
    title = task_name(task)
    if doctor and task.get("patient_name"):
        title = task["patient_name"] + " · " + title
    st.title(title)
    st.write(f"{task['followup_date']}　·　{TASK_STATES.get(task['task_status'], '状态待确认')}")
    if task.get("report_status") != "published" or doctor:
        st.caption("任务负责医生　" + task_doctor(task))
    if task.get("expires_at"):
        st.caption("截止时间　" + task["expires_at"])
    if doctor:
        if task.get("patient_id") and st.button("查看患者档案 · " + str(task.get("patient_name") or ""), key="task_patient"):
            open_patient(task["patient_id"])
        show_report_workflow(token, task["task_id"], doctor=True)
        return
    if st.session_state.pop("draft_submit_notice", None):
        st.success("本次随访已提交，记录已保存。")
    status = task["task_status"]
    if status in ("submitted", "in_review", "needs_info", "completed"):
        show_report_workflow(token, task["task_id"])
    if status in EDITABLE_STATES:
        st.subheader("补充本次情况" if status == "needs_info" else "填写本次随访")
        # 聊天显示在核对单前面；只有草稿读取成功后才启用采集输入。
        conversation = st.container()
        show_patient_draft(token, task["task_id"])
        with conversation:
            if not st.session_state.get("message"):
                st.caption("可以从最近一次血压测量说起。助手会把你说的内容整理到下方核对单。")
            show_chat(token, "collect", task_id=task["task_id"])

# ==================== 页面上方固定内容 ====================
# 先接住浏览器刚提交的输入，再请求身份、任务和草稿，避免网络失败让未保存内容消失。
remember_draft_edits()
# 这几句每次脚本重跑都会执行。set_page_config 设置浏览器标签标题和图标，
st.set_page_config(page_title="高血压随访助手", page_icon="🩺", layout="wide")
# 仅设置布局与主要按钮的颜色，保留 Streamlit 对键盘、窄屏和深色模式的支持。
st.html("""
<style>
    .stMainBlockContainer {max-width: 70rem; padding-top: 3rem;}
    [data-testid="stSidebar"] h1 {font-size: 1.5rem;}
    [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
        background-color: #176c62; border-color: #176c62; color: white;
    }
    [data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {
        background-color: #12584f; border-color: #12584f; color: white;
    }
    @media (max-width: 700px) { .stMainBlockContainer {padding-top: 2rem;} }
</style>
""")


# 未登录时只提供两类已开通账号的登录入口。
if not st.session_state.get("token"):
    clear_login()  # 清理已失效的身份、选择和历史，避免账号之间混用页面内容。
    # 宽屏登录框保持舒适行宽，小屏自动收缩到可用宽度。
    with st.container(horizontal_alignment="center"):
        with st.container(width=480):
            st.title("高血压随访助手")
            st.subheader("登录")
            with st.form("login_form"):
                # 选择身份只表明想进入哪个工作区；后端会再次验证实际账号角色。
                identity = st.radio("登录身份", ["患者", "医生"], horizontal=True)
                name = st.text_input("姓名", max_chars=64)
                password = st.text_input("密码", type="password", max_chars=128)
                submitted = st.form_submit_button("登录", type="primary")

    if submitted:
        if not name.strip() or not password:
            st.warning("请输入姓名和密码。")
        else:
            role = "patient"
            if identity == "医生":
                role = "doctor"
            # 保留原接口的 username 键，里面传用户填写的姓名，例如“李明”。
            login_data = {"username": name.strip(), "password": password, "role": role}
            try:
                result = request_api("POST", "/auth/login", data=login_data).json()
            except requests.RequestException as error:
                show_request_error(error)  # 身份、姓名或密码错误时留在登录页。
            else:
                st.session_state["token"] = result["access_token"]
                st.session_state["user"] = result["user"]  # 工作区只取服务端确认的角色。
                clear_task_selection()
                st.rerun()
    st.stop()  # 未完成身份验证前不显示患者或医生的数据。

# ==================== 已登录区域：先核对身份，再显示聊天 ====================
# 每次页面重跑都向后端核对身份
token: str = st.session_state["token"]
try:
    # token 是 str，api_client 会放入 Authorization: Bearer ... 请求头。
    # GET /me：询问后端“这个凭证现在对应谁，是否还有效”，没有 data 请求体。
    # backend.current_user → user_store.get_current_user → sessions JOIN users。
    # 这里不再发送用户名和密码；每次聊天请求本身也会在后端独立鉴权。
    # 输出 user 是 dict，例如 {"user_id":"U001","username":"patient_sun","role":"patient"}。
    user = request_api("GET", "/me", token=token).json()
except requests.RequestException as error:
    show_request_error(error)
    if st.button("返回登录"):
        clear_login()
        st.rerun()
    # 身份无法核对时，不执行下面的聊天输入代码，避免只凭本地状态继续聊天。
    st.stop()

# 后端可能调整账号角色；先比较身份，变化时清理旧任务和旧对话。
previous_user = st.session_state.get("user", {})
if previous_user.get("user_id") != user["user_id"] or previous_user.get("role") != user["role"]:
    clear_task_selection()
    for key in ("workspace", "selected_patient", "saved_edits", "chat_scope", "scope_risk", "patient_search", "patient_filter",
                "doctor_section", "confirm_logout", "logout_requested"):
        st.session_state.pop(key, None)  # 账户更换后清理之前身份的页面记忆。
# 把本轮已核对的身份保存给下次比较，页面权限以本轮 /me 返回值为准。
st.session_state["user"] = user
# 后端只允许 patient/doctor 通过这里；用户不通过页面自己选择角色。
role_name = "患者"
if user["role"] == "doctor":
    role_name = "医生"
with st.sidebar:
    # sidebar 是 Streamlit 的侧栏容器。其内部元素显示在左侧，而不是正文中间。
    # f 字符串会把花括号中的变量值填进去，例如最终为 "patient_sun · 患者"。
    st.title("随访助手")
    st.write(f"{user['username']} · {role_name}")
    landing = "我的随访"
    if user["role"] == "doctor":
        landing = "医生工作台"
    if st.button(landing, key="return_tasks", width="stretch"):
        go_home()
    if st.button("问助手", key="open_assistant", width="stretch"):
        # 保持当前患者/任务与未保存编辑，查询模式无法向后端提议修改草稿。
        st.session_state["workspace"] = "assistant"
        st.rerun()
    st.divider()
    if st.button("退出登录"):
        if draft_has_changes() or st.session_state.get("saved_edits"):
            st.session_state["confirm_logout"] = True
        else:
            st.session_state["logout_requested"] = True
    if st.session_state.get("confirm_logout"):
        st.warning("有尚未保存的手工修改。可以返回任务继续填写。")
        if st.button("放弃未保存修改并退出"):
            st.session_state["logout_requested"] = True
        if st.button("继续使用"):
            st.session_state.pop("confirm_logout", None)
            st.rerun()
    if st.session_state.pop("logout_requested", False):
        try:
            request_api("POST", "/auth/logout", token=token)
        except requests.RequestException as error:
            show_request_error(error)
            st.stop()
        clear_login()
        st.rerun()

# 先核对当前页面的资料范围，再绘制资料或聊天，避免重跑时沿用失去权限的缓存。
workspace = st.session_state.get("workspace", "home")
doctor = user["role"] == "doctor"
current_task = None
selected_patient = st.session_state.get("selected_patient")
detail = None
if st.session_state.get("selected_task"):
    task_id = st.session_state["selected_task"]["task_id"]
    current_task = get_view(token, f"/tasks/{task_id}")
    if current_task is None:
        if st.button("重新加载当前任务", key="reload_current_task"):
            st.rerun()
        st.stop()
    st.session_state["selected_task"] = current_task
elif selected_patient and doctor:
    detail = get_view(token, f"/patients/{selected_patient}")
    if detail is None:
        st.stop()

if workspace == "assistant":
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
        show_chat(token, "query", patient_id=selected_patient, access_scope=tuple(sorted(access_scope)))
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
elif current_task:
    show_task_page(token, current_task, doctor)
elif detail:
    show_patient_profile(token, detail)
elif doctor:
    show_physician_home(token)
else:
    show_patient_home(token)
