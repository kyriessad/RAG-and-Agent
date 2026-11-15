import time
from agent.tools.agent_tools import rag
from utils.logger_handler import logger
import streamlit as st
from agent.tools.react_agent import ReactAgent
import sqlite3
from utils.path_tool import get_abs_path
from utils.user_store import init_db, hash_password, verify_password


# 注册成功返回 True，用户名已存在返回 False。
def register_user(user_name: str, password: str) -> bool:
    db_path = get_abs_path("data/app.db")  # 数据库文件地址
    conn = sqlite3.connect(db_path)  # 连接数据库
    try:
        password_hash = hash_password(password)
        conn.execute(  # 把用户信息插入数据库
            "INSERT INTO users (user_name, password) VALUES (?, ?)",
            (user_name, password_hash),
        )
        conn.commit()
        return True

    except sqlite3.IntegrityError:
        conn.rollback()
        return False

    finally:
        conn.close()  # 关闭数据库连接


# 登录成功返回用户信息，失败返回 None。
def login_user(user_name: str, password: str):
    db_path = get_abs_path("data/app.db")
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT user_id, user_name, password
            FROM users
            WHERE user_name = ?
            """,
            (user_name, ),
        )

        user = cursor.fetchone()
        if user is None:
            return None

        stored_password = user[2]
        # 检查本次输入的密码是否正确
        if not verify_password(password, stored_password):
            return None
        return {
            "user_id": user[0],
            "user_name": user[1],
        }
    finally:
        conn.close()


# 标题
st.title("智扫通机器人智能客服")
st.divider()

# 初始化数据库
try:
    init_db()
except sqlite3.Error:
    logger.exception("初始化用户数据库失败")
    st.error("用户数据库暂时无法使用。")
    st.stop()

# 登录状态
if "auth_page" not in st.session_state:
    st.session_state["auth_page"] = "login"

if "user" not in st.session_state:
    st.session_state["user"] = None

# 注册、登录
if st.session_state["user"] is None:  # 没有用户
    if st.session_state["auth_page"] == "register":  # 注册状态
        st.subheader("注册")
        with st.form("register_form"):  # 把输入框和按钮组成一个表单，点击注册后统一提交
            user_name = st.text_input("用户名")
            password = st.text_input("密码", type="password")
            confirm_password = st.text_input("确认密码", type="password")

            submitted = st.form_submit_button("注册")

        if submitted:
            user_name = user_name.strip()

            if not user_name or not password:
                st.warning("请输入用户名和密码。")

            elif password != confirm_password:
                st.warning("两次输入的密码不一致。")
            else:
                try:
                    success = register_user(user_name, password)  # 把数据插入数据库
                except sqlite3.Error:
                    logger.exception("注册时数据库操作失败")
                    st.error("注册失败，数据库暂时无法使用，请稍后重试。")
                else:
                    if success:
                        st.session_state["auth_page"] = "login"
                        st.session_state["register_success"] = True
                        st.rerun()  # 刷新页面
                    else:
                        st.warning("用户名已存在，请换一个用户名。")
        if st.button("已有账号？去登录"):
            st.session_state["auth_page"] = "login"
            st.rerun()

    else:  # 登录状态(默认)
        # 登录表单、验证成功后保存user，st.rerun
        st.subheader("登录")
        # 注册完成后，在登录区域显示一次提示
        if st.session_state.pop("register_success", False):  # 取出注册状态，如果不存在则为false
            st.success("注册成功，请登录。")
        # 登录表单
        with st.form("login_form"):
            login_user_name = st.text_input("用户名")
            login_password = st.text_input("密码", type="password")
            login_submitted = st.form_submit_button("登录")

        if login_submitted:
            user_name = login_user_name.strip()
            if not user_name or not login_password:
                st.warning("请输入用户名和密码。")
            else:
                try:
                    user = login_user(user_name, login_password)
                except sqlite3.Error:
                    logger.exception("登录时数据库查询失败")
                    st.error("登录失败，数据库暂时无法使用，请稍后重试。")
                else:
                    if user is None:
                        st.error("用户名或密码错误。")
                    else:
                        st.session_state["user"] = user  # 保存当前登录用户
                        # 本次登录开始新的聊天
                        st.session_state.pop("agent", None)
                        st.session_state.pop("message", None)
                        # 重新执行页面，显示智能客服
                        st.rerun()

        if st.button("没有账号？去注册"):
            st.session_state["auth_page"] = "register"
            st.rerun()

    st.stop()

# 检查、同步知识库
try:
    with st.spinner("正在检查知识库资料..."):
        sync_result: dict[str, int] = rag.vector_store.load_document()  # 自动检查知识库
except Exception:
    logger.exception("知识库同步失败")
    st.error("知识库初始化或更新失败，请检查日志后重试。")
    st.stop()

# 准备当前用户会话的 Agent
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent(user_id=st.session_state["user"]["user_id"])

if "message" not in st.session_state:
    st.session_state["message"] = []

# st.session_state = ｛"agent":ReactAgent(), "message":[  {"role":"user","content":...},{"role":"assistant","content":...}   ]｝

# 渲染历史消息
for message in st.session_state["message"]:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # 第一个位置对应实时回答时的 status_box。
            # 历史回答已经完成，不需要显示状态。
            st.empty()

            # 第二个位置对应实时回答时的 answer_box。
            history_answer_box = st.empty()
            history_answer_box.markdown(message["content"])

        else:
            # 用户消息保持原来的显示方式。
            st.write(message["content"])

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)

    st.session_state["message"].append({
        "role": "user",
        "content": prompt,
    })

    answer = ""

    with st.chat_message("assistant"):
        status_box = st.empty()
        answer_box = st.empty()

        try:
            for event in st.session_state["agent"].execute_stream(prompt):
                event_type = event["type"]
                content = event["content"]
                if event_type == "status":
                    status_box.info(content)
                elif event_type == "reset":
                    answer = ""
                    answer_box.empty()
                elif event_type == "delta":
                    status_box.empty()
                    # 将新增片段追加到已经收到的文字后面。
                    answer += content
                    # 每收到一段，就更新同一个显示位置。
                    answer_box.markdown(answer + " ▌")
                elif event_type == "answer":
                    # 这里收到的是完整答案，不是新增片段。
                    answer = content
                    if answer:
                        answer_box.markdown(answer)
                    else:
                        answer_box.empty()

        except Exception:
            logger.exception("客服回答失败")
            # 失败时清掉未完成的答案，不保存为正常历史消息。
            answer_box.empty()
            status_box.error("本次处理失败，请稍后重试。")
        else:
            status_box.empty()
            if answer:
                answer_box.markdown(answer)
                # 整轮执行成功后，只保存一条完整答案。
                st.session_state["message"].append({
                    "role": "assistant",
                    "content": answer,
                })
            else:
                st.warning("本次没有收到有效回复，请重试。")

