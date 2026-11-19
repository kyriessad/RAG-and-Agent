"""页面会话、身份和导航。"""

import requests
import streamlit as st

from utils import api_client


# 随访草稿字典
DRAFT_TEXT_FIELDS = {
    "systolic_bp": "高压（收缩压，mmHg）",
    "diastolic_bp": "低压（舒张压，mmHg）",
    "measured_at": "什么时候测的？",
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


EDITABLE_STATES = ("pending", "in_progress", "needs_info")


# 清除草稿页面状态，返回 None；只清内存，不删除后端已保存草稿。
def clear_draft_state():
    # list() 先复制键名，避免一边遍历字典一边删除导致错误。
    for key in list(st.session_state):
        if key.startswith("draft_"):
            st.session_state.pop(key, None)  # 退出或切换任务时连输入框一起清理。


# 输入草稿字段字典，返回输入框显示值；例如 systolic_bp=None 变成空字符串
def draft_display_values(data):
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


# 页面输入和加载时的草稿是否变了
def draft_has_changes():
    base = st.session_state.get("draft_base")  # base 是带 revision 的完整接口返回值。
    if base is None:
        return False  # 尚未进入草稿页面，没有可以丢失的修改。
    # 先统一为页面显示的字符串，再比较；否则整数 138 和字符串“138”会被误判。
    return st.session_state["draft_edits"] != draft_display_values(base["data"])


# 备份草稿输入，写进session_state字典
def remember_draft_field(field):
    """只在患者改动控件时更新副本，离开表单不会覆盖未保存的输入。"""
    st.session_state["draft_edits"][field] = st.session_state["draft_field_" + field]


# 把接口草稿放进页面内存session_state
def load_draft_fields(draft):
    st.session_state["draft_base"] = draft  # 保存版本，之后 PUT 用它防止旧页面覆盖新数据。
    # 持久字段转换成可用于 Streamlit 控件的值，然后再恢复输入。
    values = draft_display_values(draft["data"])
    st.session_state["draft_edits"] = dict(values)  # 明确重新加载时同步替换备份，不能恢复被放弃的旧值。


# 清除页面登录信息。
def clear_login():
    clear_draft_state()  # 清除草稿及输入框缓存，防止下一个账号看到旧内容。
    clear_chat_state()
    # user={...} 保存账号字典；token="示例令牌" 保存登录凭证字符串；message=[...] 保存聊天消息列表。
    # agent 是旧版页面留下的键。当前版本的 Agent 对象在后端，本页面不再创建它。
    # 退出或登录过期时，一起清掉身份、令牌和页面里的聊天历史。
    for key in ("user", "token", "message", "agent", "selected_task", "selected_patient",
                "workspace", "chat_scope", "saved_edits", "scope_risk", "patient_filter", "patient_search",
                "doctor_section", "confirm_logout", "logout_requested", "discharge_form"):
        # key 是每轮循环取出的 str，例如第一次为 "user"。
        # pop(key, None) = 删除并返回这个键的旧值；键不存在就返回 None，不报错。
        # 这里没有接收 pop 的返回值，因为我们只想删除，不需要旧内容。
        st.session_state.pop(key, None)


# 返回任务列表前清理当前选择
def clear_chat_state():
    for key in list(st.session_state):
        if key.startswith(("chat_", "card_", "review_", "conversation_")):
            st.session_state.pop(key, None)


def remember_task_edits():
    if draft_has_changes():
        base = st.session_state["draft_base"]
        edits = st.session_state.setdefault("saved_edits", {})
        edits[base["task_id"]] = {"base": base, "edits": dict(st.session_state["draft_edits"])}


def clear_task_selection():
    clear_draft_state()  # 切换任务不沿用上一项任务的草稿和版本。
    # selected_task 是任务字典，例如 {"task_id": "F001", "task_status": "pending", ...}。
    st.session_state.pop("selected_task", None)
    # 切换任务不沿用屏幕上的旧对话，避免把前一项任务的问题显示到新任务下。
    st.session_state["message"] = []
    st.session_state.pop("chat_scope", None)
    st.session_state.pop("scope_risk", None)
    st.session_state.pop("chat_review_prefill", None)
    st.session_state.pop("chat_open_card", None)


# 将 requests 异常转成中文页面提示
def show_request_error(error):
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


def get_view(token, path):
    """用登录凭证读取 JSON，失败展示真实错误并返回 None；空列表表示读取成功。"""
    try:
        return api_client.request_api("GET", path, token=token).json()
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


def go_home(workspace="home"):
    """返回工作区首页，按任务保留尚未保存的表单内容；只改变页面会话。"""
    remember_task_edits()
    clear_task_selection()
    st.session_state.pop("selected_patient", None)
    st.session_state["workspace"] = workspace
    st.rerun()


def open_task(token, task_id, review_card=None):
    """用户点选时通过 GET 重查归属；成功才绑定任务、恢复该任务的未保存修改。"""
    task = get_view(token, f"/tasks/{task_id}")
    if task is None:
        st.stop()
    remember_task_edits()
    clear_task_selection()
    st.session_state.pop("selected_patient", None)
    st.session_state["selected_task"] = task
    if review_card:
        st.session_state["chat_review_prefill"] = review_card
    cached = st.session_state.get("saved_edits", {}).pop(task_id, None)
    if cached and task["task_status"] in EDITABLE_STATES:
        st.session_state["draft_base"] = cached["base"]
        st.session_state["draft_edits"] = cached["edits"]
    st.session_state["workspace"] = "task"
    st.rerun()


def open_patient(patient_id, open_card=None):
    """从医生名册选择一位患者；详情页会再次请求后端核对权限。"""
    remember_task_edits()
    clear_task_selection()
    st.session_state["selected_patient"] = patient_id
    if open_card:
        st.session_state["chat_open_card"] = open_card
    st.session_state["workspace"] = "patient"
    st.rerun()


def verified_user(token):
    try:
        # token 是 str，api_client 会放入 Authorization: Bearer ... 请求头。
        # GET /me：询问后端“这个凭证现在对应谁，是否还有效”，没有 data 请求体。
        # backend.current_user → user_store.get_current_user → sessions JOIN users。
        # 这里不再发送用户名和密码；每次聊天请求本身也会在后端独立鉴权。
        # 输出 user 是 dict，例如 {"user_id":"U001","username":"patient_sun","role":"patient"}。
        user = api_client.request_api("GET", "/me", token=token).json()
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
        clear_chat_state()
        clear_task_selection()
        for key in ("workspace", "selected_patient", "saved_edits", "chat_scope", "scope_risk", "patient_search", "patient_filter",
                    "doctor_section", "confirm_logout", "logout_requested"):
            st.session_state.pop(key, None)  # 账户更换后清理之前身份的页面记忆。
    # 把本轮已核对的身份保存给下次比较，页面权限以本轮 /me 返回值为准。
    st.session_state["user"] = user
    return user


def workspace_context(token, doctor):
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

    return current_task, detail
