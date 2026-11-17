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
#     → utils/user_store.py 的 login_user(username, password)
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
#   患者聊天会附带 task_id；后端重新检查归属，再把任务范围交给 Agent。
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
# 因此，正常运行路径是“后端重启后的第一次聊天触发”，不是登录时或每轮提问时。
# 修改知识资料后要重启后端，等下一次聊天重新检查；前端重跑本身不会触发重建。
# fastapi dev 的自动重载如果重启了后端进程，也会把这个内存标记重新置为 False。
# 每次真正执行 RAG 检索时，仍会将本次查询文本向量化，这与“资料同步”是两件事。
#
# 下面的例子只用来解释类型，示例令牌不是可用的真实登录凭证。

import json
import requests
import streamlit as st
from utils.api_client import request_api


def clear_login():
    """清除页面登录信息。无输入、返回 None；不删除数据库账号或医疗记录。"""
    # user={...} 保存账号字典；token="示例令牌" 保存登录凭证字符串；message=[...] 保存聊天消息列表。
    # agent 是旧版页面留下的键。当前版本的 Agent 对象在后端，本页面不再创建它。
    # 退出或登录过期时，一起清掉身份、令牌和页面里的聊天历史。
    for key in ("user", "token", "message", "agent", "selected_task"):
        # key 是每轮循环取出的 str，例如第一次为 "user"。
        # pop(key, None) = 删除并返回这个键的旧值；键不存在就返回 None，不报错。
        # 这里没有接收 pop 的返回值，因为我们只想删除，不需要旧内容。
        st.session_state.pop(key, None)

    # 这个函数不会删除数据库账号，也不会使后端 token 自动失效。
    # 真正“退出登录”还要调用 POST /auth/logout，见下面侧栏按钮的处理顺序。
    # auth_page 没清理，仍然记录未登录时应该展示 login 还是 register 页面。


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
        detail = "输入格式不正确，请检查用户名、密码或消息长度。"
    st.error(detail)

    # 登录过期后只清理本地状态，下一次脚本重跑便会回到登录页。
    # status_code 是 int：401 表示身份校验失败，409 表示冲突，503 表示服务不可用。
    # 这里不自动调用 st.rerun()，让用户先看到错误；下一次脚本重跑时显示登录区域。
    if response.status_code == 401:
        clear_login()


def clear_task_selection():
    """返回任务列表前清理当前选择。无输入、返回 None；只修改页面会话内存。"""
    # selected_task 是任务字典，例如 {"task_id": "F001", "task_status": "pending", ...}。
    st.session_state.pop("selected_task", None)
    # 切换任务不沿用屏幕上的旧对话，避免把前一项任务的问题显示到新任务下。
    st.session_state["message"] = []


def show_task_details(task):
    """把一个任务字典显示在页面上，返回 None；不请求网络、不写数据库。

    输入示例：{"task_id": "F001", "task_status": "pending", "followup_date": "2026-11-18", ...}。
    输出是页面上的日期、类型、状态和负责医生；缺失字段显示“未提供”。
    """
    # 数据库使用英文状态值，页面将它们显示为患者容易理解的中文。
    status_names = {
        "pending": "待随访",
        "in_progress": "进行中",
        "submitted": "已提交",
        "in_review": "审核中",
        "completed": "已完成",
        "needs_info": "需补充",
    }
    # 未知状态不推断成“已完成”，让页面如实显示状态尚不明确。
    status_name = status_names.get(task.get("task_status"), "状态待确认")
    # task.get() 取得对应字段；没有字段时使用“未提供”，不自行推算随访日期。
    st.write("随访日期：", task.get("followup_date") or "未提供")
    st.write("截止时间：", task.get("expires_at") or "未提供")
    st.write("任务类型：", task.get("task_type") or "未提供")
    st.write("当前状态：", status_name)
    st.write("负责医生：", task.get("doctor_name") or "未提供")


def show_patient_tasks(token):
    """显示患者任务列表或当前任务，返回经后端确认的任务字典。

    输入 token 是登录凭证；返回示例：{"task_id": "F001", "task_status": "pending", ...}。
    尚未选择任务、任务无权访问或请求失败时停止本轮页面，不显示聊天输入。
    本函数会发送只读 HTTP 请求、更新页面选择，不改变医疗库中的任务状态。
    """
    # selected_task 只记录页面正在查看哪项任务，不能替代后端的归属校验。
    selected_task = st.session_state.get("selected_task")
    if selected_task is None:
        # 未选择任务时，先展示服务端根据登录身份筛选的任务列表。
        st.subheader("我的随访任务")
        try:
            # 输出 task_list 是列表，例如 [{"task_id": "F001", ...}, {"task_id": "F002", ...}]。
            task_list = request_api("GET", "/tasks", token=token).json()
        except requests.RequestException as error:
            # 请求失败不等于没有任务；保留错误提示，并提供重新获取的入口。
            show_request_error(error)
            if st.button("重新加载任务", key="reload_tasks"):
                st.rerun()
            st.stop()
        # 没关联患者资料或没有安排时，后端返回空列表，不创建替代任务。
        if not task_list:
            st.info("当前没有可查看的随访任务。若刚注册，请联系负责人员关联患者资料。")
            st.stop()
        # 每项任务单独显示，按钮 key 使用任务编号，避免多个同名按钮相互冲突。
        for task in task_list:
            task_id = task["task_id"]
            st.markdown(f"**随访任务 {task_id}**")
            show_task_details(task)
            if st.button("进入任务", key=f"enter_task_{task_id}"):
                try:
                    # 列表加载后权限可能改变，进入时再请求详情，由后端重新验证归属。
                    task_detail = request_api("GET", f"/tasks/{task_id}", token=token).json()
                except requests.RequestException as error:
                    # 进入失败时不设置 selected_task，也不开放聊天输入。
                    show_request_error(error)
                    st.stop()
                # 仅在后端确认成功后保存选择；新任务从空的页面对话开始。
                clear_task_selection()
                st.session_state["selected_task"] = task_detail
                st.rerun()
            # 分隔不同任务，让日期和“进入任务”按钮的对应关系清楚。
            st.divider()
        # 列表页不执行后面的聊天代码，必须明确进入一项任务。
        st.stop()

    try:
        # 每次重跑（包括发送问题）重新取得详情，不继续相信过期的页面选择。
        task_id = selected_task["task_id"]
        task_detail = request_api("GET", f"/tasks/{task_id}", token=token).json()
    except requests.RequestException as error:
        # 后端把不存在和无权访问统一返回 404；不能继续显示旧任务或旧对话。
        if error.response is not None:
            if error.response.status_code in (403, 404):
                clear_task_selection()
        show_request_error(error)
        # 连接失败时也不显示未经本轮校验的任务信息；用户可以返回后重试。
        if st.button("返回任务列表", key="return_tasks"):
            clear_task_selection()
            st.rerun()
        st.stop()

    # 以本次后端结果更新日期、状态等字段，不用进入时留下的旧字典显示任务。
    st.session_state["selected_task"] = task_detail
    st.subheader(f"当前随访任务：{task_detail['task_id']}")
    show_task_details(task_detail)
    # 当前只完成任务入口和查询对话，不让患者误以为聊天就等于正式提交。
    st.info("当前可围绕这项任务查询和交流，暂不能正式提交随访信息。")
    if st.button("返回任务列表", key="return_tasks"):
        clear_task_selection()
        st.rerun()
    # 返回字典供下面组装聊天请求，task_id 只表示选择，后端仍会独立鉴权。
    return task_detail


# ==================== 页面上方固定内容 ====================
# 这几句每次脚本重跑都会执行。set_page_config 设置浏览器标签标题和图标，
st.set_page_config(page_title="高血压随访助手", page_icon="💬")
st.title("高血压随访助手")
st.caption("查询随访安排、出院信息和近期记录，也可以询问高血压相关知识。")
st.divider()

# ==================== 决定本轮该显示哪个页面 ====================
# 当前会话状态示例（不是一段需要你执行的初始化代码）：
# {
#   "auth_page": "login",            # str，未登录时的页面选择
#   "register_success": True,        # bool，注册成功后的一次性提示，可不存在
#   "user": {"user_id": "U001", "username": "patient_sun", "role": "patient"},
#   "token": "一串随机登录凭证",       # str，后端返回，不能用用户名代替
#   "message": [                     # list[dict]，用于前端重新绘制聊天历史
#       {"role": "user", "content": "你好"},
#       {"role": "assistant", "content": "你好，请问有什么需要帮助？"}
#   ]
# }
# st.session_state 是类似字典的对象，支持 []、get()、pop()、in；
# 在同一连接会话内因按钮等交互重跑时可以继续读取，浏览器刷新/重连则不保证。
if "auth_page" not in st.session_state:
    # 只在键不存在时设默认值，避免每次重跑都把注册页强制改回登录页。
    st.session_state["auth_page"] = "login"

# 旧页面只有 user、没有后端令牌；切换到医疗库后要求重新登录。
if not st.session_state.get("token"):  # get("token") 在键不存在时返回 None
    # 没有token时  清空登录状态
    clear_login()

# 如果还没登录 / 注册
if st.session_state.get("user") is None:
    # 注册页。
    if st.session_state["auth_page"] == "register":
        st.subheader("患者注册")
        st.caption("新账号关联患者资料后可查询随访记录；医生请使用已开通的账号登录。")
        with st.form("register_form"):
            # st.form 返回表单容器，with 表示里面的输入框和按钮放进同一表单。
            # "register_form" 是这个表单的标识，不是接口路径，也不是数据库表名。
            # text_input 返回 str：没填时为 ""，例如填完后 username="patient_sun"。
            username = st.text_input("用户名", max_chars=64)
            # type="password" 只是把屏幕上的字符遮住，变量本身仍是原始密码字符串。
            # 页面不做密码哈希；哈希和保存由后端 user_store.register_user() 执行。
            password = st.text_input("密码", type="password", max_chars=128)
            confirm_password = st.text_input("确认密码", type="password", max_chars=128)
            # submitted 是 bool；本轮因点击“注册”提交表单时为 True，平时为 False。
            # 因此不会在页面刚打开、用户尚未点击注册时就发送注册请求。
            submitted = st.form_submit_button("注册")

        if submitted:
            # strip() 返回去掉两端空白的新字符串，例如 " 小王 " → "小王"。
            # 密码不 strip，用户实际输入的密码内容会原样交给后端。
            if not username.strip() or not password:
                st.warning("请输入用户名和密码。")
            elif password != confirm_password:
                st.warning("两次输入的密码不一致。")
            else:
                try:
                    # method="POST"：提交内容；path="/auth/register"：后端路由。
                    # data 是 dict，例如 {"username": "patient_sun", "password": "示例密码"}。
                    # confirm_password 只用于本地比较，不需要发给后端。
                    # 此时还没登录，省略 token，使用默认值 None。
                    # 注册由后端写入 medical.db，页面不执行 SQL。
                    request_api("POST", "/auth/register", data={"username": username.strip(), "password": password})
                    # 成功返回 Response（HTTP 201），本处不读取其正文。
                    # request_api 内已检查 HTTP 错误；能执行到 try 的结尾即表示请求成功。
                except requests.RequestException as error:
                    # 连接失败、超时、后端返回错误状态等，都用同一个展示函数处理。
                    show_request_error(error)
                else:
                    # try/except/else 的 else 只在 try 没有抛异常时执行。
                    # 先改状态，再重跑；新一轮就会进入下面的登录页分支。
                    st.session_state["auth_page"] = "login"
                    st.session_state["register_success"] = True
                    # st.rerun() 立即结束本轮并安排从头执行，不是仅重新画这一行。
                    st.rerun()

        if st.button("已有账号？去登录"):
            # 普通 button 也返回 bool；点击后的这一轮为 True。
            # 只切换页面状态，不注册、不登录，也不访问数据库。
            st.session_state["auth_page"] = "login"
            st.rerun()
    # 登录页(默认页)
    else:
        st.subheader("登录")
        if st.session_state.pop("register_success", False):
            # pop 同时“取出并删除”这个标记，让成功提示仅在这一轮显示。
            # 键不存在时返回 False，就不会显示注册成功提示。
            st.success("注册成功，请登录。")
        with st.form("login_form"):
            # 两个返回值都是 str；表单提交后，这一轮取得用户提交的文本。
            username = st.text_input("用户名", max_chars=64)
            password = st.text_input("密码", type="password", max_chars=128)
            submitted = st.form_submit_button("登录")

        if submitted:
            if not username.strip() or not password:
                st.warning("请输入用户名和密码。")
            else:
                try:
                    # 这一行可以拆开理解（下面只是注释，不会额外执行两次请求）：
                    # response = request_api("POST", "/auth/login", data={...})
                    # result = response.json()
                    # response 是 HTTP 回复对象；result 才是解析后的 Python 字典。
                    # 后端 login_user 查 medical.db.users、验证密码哈希，成功后
                    # 把新 token 的哈希和过期时间写入 sessions，再返回以下形状：
                    # {
                    #   "access_token": "示例随机令牌",
                    #   "user": {"user_id": "U001", "username": "patient_sun", "role": "patient"}
                    # }
                    result = request_api("POST", "/auth/login", data={"username": username.strip(), "password": password}).json()
                except requests.RequestException as error:
                    show_request_error(error)
                else:
                    # result 示例：{"access_token": "...", "user": {"username": "张某", "role": "patient", ...}}。
                    # 外层字典里的 access_token 是 str，user 是另一个 dict。
                    # token 是登录通行凭证，不是大模型计费用的 token，也不是密码。
                    st.session_state["token"] = result["access_token"]
                    st.session_state["user"] = result["user"]
                    # 新登录清掉之前选择的任务和对话，避免显示上一位用户的信息。
                    clear_task_selection()
                    # 下一轮有 user 和 token，将跳过登录/注册分支，进入聊天页面。
                    st.rerun()

        if st.button("没有账号？去注册"):
            st.session_state["auth_page"] = "register"
            st.rerun()

    # stop 停止本轮脚本后续代码；上面已经画出的登录/注册元素仍然显示。
    # 它不关闭 Streamlit 服务。下次点击按钮仍会触发一轮新执行。
    # 成功登录时 st.rerun() 已经打断本轮，因此不会再走到这里。
    st.stop()  # 没有登录时，不显示后面的聊天页面。

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
# 把本轮已核对的身份保存给下次比较，页面权限以本轮 /me 返回值为准。
st.session_state["user"] = user
# 后端只允许 patient/doctor 通过这里；用户不通过页面自己选择角色。
role_name = "患者"
if user["role"] == "doctor":
    role_name = "医生"
with st.sidebar:
    # sidebar 是 Streamlit 的侧栏容器。其内部元素显示在左侧，而不是正文中间。
    # f 字符串会把花括号中的变量值填进去，例如最终为 "patient_sun · 患者"。
    st.write(f"{user['username']} · {role_name}")
    if st.button("退出登录"):
        try:
            # 先通知后端：删除 medical.db.sessions 中该凭证的记录，并清掉后端 Agent。
            # 这不会删除 medical.db.users 账号或患者的医疗记录。
            request_api("POST", "/auth/logout", token=token)
        except requests.RequestException as error:
            show_request_error(error)
            # 后端退出没有成功时先显示错误，不假装已经完成了凭证撤销。
            st.stop()
        # 后端退出成功后，再删除 Streamlit 会话中的 user/token/message。
        clear_login()
        st.rerun()

    # 这段灰色文字是写在代码里的固定提示，不是从数据库查出来的任务列表。
    st.caption("可以问：我的随访什么时候到期？出院时有哪些注意事项？最近血压记录是什么？")

# 患者先进入任务，医生保持现有的授权查询聊天；这不是新增医生审核功能。
current_task = None
if user["role"] == "patient":
    current_task = show_patient_tasks(token)

# ==================== 用历史数据重新画出聊天消息 ====================
# message = {"role":"assistant","content":"你好"}。
for message in st.session_state.get("message", []):
    # chat_message 返回消息容器，role="user" 和 "assistant" 决定身份样式/图标。
    # with 将这条消息的正文放进对应容器，而不是自动去调用模型。
    with st.chat_message(message["role"]):
        # 重新画历史只读取这个列表，不会再次向后端发送历史里的每条问题。
        st.markdown(message["content"])

# ==================== 接收这一轮新问题 ====================
prompt = st.chat_input("请输入你的问题", max_chars=4000)
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

        try:
            # method="POST"，path="/chat"，token 是登录凭证，data 是问题字典。
            # 医生使用原有通用查询；患者额外发送明确选择的 task_id。
            chat_data = {"message": prompt}
            if current_task is not None:
                # 例如 {"message": "这次随访什么时候到期？", "task_id": "F001"}。
                chat_data["task_id"] = current_task["task_id"]
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

                    # event_type 和 content 在当前协议里都是 str。
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
        else:
            # 发生错误时删除屏幕上的半成品，但保留错误提示和用户已经发送的问题。
            answer_box.empty()
            if not failed:
                # 没有明确报错但连接结束了，仍可能缺少最终 answer，不能当作成功。
                status_box.error("未收到完整回答，请重试。")
