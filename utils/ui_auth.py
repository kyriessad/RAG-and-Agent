"""登录与账号注册。"""

import requests
import streamlit as st

from utils import api_client
from utils.ui_session import clear_login, clear_task_selection, draft_has_changes, show_request_error


def sign_in(name, password, role):
    """用表单姓名、原密码和角色向后端登录。成功后进入对应工作台，失败时显示原因。"""
    try:
        # 两类入口共用同一登录接口，由后端核对实际角色并签发会话凭证。
        result = api_client.request_api("POST", "/auth/login", data={
            "username": name.strip(), "password": password, "role": role,
        }).json()
    except requests.RequestException as error:
        show_request_error(error)
        return
    st.session_state["token"] = result["access_token"]
    st.session_state["user"] = result["user"]
    st.session_state.pop("registering", None)
    clear_task_selection()
    st.rerun()


def show_registration_form():
    """显示所选角色的注册表单，无入参。提交后创建账号和档案，再登录；失败保留输入。"""
    identity = st.radio("注册身份", ["患者", "医生"], horizontal=True, key="signup_identity")
    doctor_labels = {}
    if identity == "患者":
        try:
            # 注册前使用简要名录，让患者按姓名和科室选择，表单只提交对应编号。
            doctors = api_client.request_api("GET", "/auth/doctors").json()["doctors"]
        except requests.RequestException as error:
            show_request_error(error)
            st.button("重新加载医生名录")
            return
        for doctor in doctors:
            doctor_labels[doctor["doctor_id"]] = doctor["name"] + " · " + doctor["department"]
        if not doctor_labels:
            st.info("还没有可选的主管医生。请先注册医生账号，再回来建立患者档案。")
            st.button("刷新医生名录")
            return

    with st.form("signup_" + identity, border=False):
        name = st.text_input("姓名", max_chars=64)
        password = st.text_input("设置密码", type="password", max_chars=128)
        if identity == "患者":
            age_col, gender_col = st.columns(2)
            age = age_col.number_input("年龄", min_value=0, value=None, step=1)
            genders = {"男": "male", "女": "female", "未说明": "unknown"}
            gender = gender_col.selectbox("性别", list(genders), index=None, placeholder="请选择")
            doctor_id = st.selectbox("主管医生", list(doctor_labels), format_func=doctor_labels.get,
                                     index=None, placeholder="选择医生和科室")
        else:
            department = st.text_input("科室", max_chars=64)
            title = st.text_input("职称（选填）", max_chars=64)
        phone = st.text_input("联系电话（选填）", max_chars=32)
        submitted = st.form_submit_button("注册并进入工作台", type="primary", width="stretch")

    if not submitted:
        return
    if not name.strip() or not password:
        st.warning("请填写姓名和密码。")
        return
    payload = {"username": name.strip(), "password": password}
    if identity == "患者":
        if age is None or gender is None or doctor_id is None:
            st.warning("请填写年龄，并选择性别和主管医生。")
            return
        role = "patient"
        path = "/auth/register"
        payload["profile"] = {"age": age, "gender": genders[gender], "doctor_id": doctor_id, "phone": phone.strip()}
    else:
        if not department.strip():
            st.warning("请填写医生所在科室。")
            return
        role = "doctor"
        path = "/auth/register/doctor"
        payload.update({"department": department.strip(), "title": title.strip(), "phone": phone.strip()})
    try:
        # 注册提交一次；如果随后登录请求失败，仍可以用已建立的账号从登录页进入。
        api_client.request_api("POST", path, data=payload)
    except requests.RequestException as error:
        show_request_error(error)
        return
    st.success("注册成功。也可以返回登录，使用刚设置的密码进入工作台。")
    sign_in(name, password, role)


def show_login_page():
    clear_login()  # 清理已失效的身份、选择和历史，避免账号之间混用页面内容。
    # 宽屏登录框保持舒适行宽，小屏自动收缩到可用宽度。
    with st.container(horizontal_alignment="center"):
        with st.container(width=480, border=True, key="login_panel"):
            st.title("高血压随访助手")
            if st.session_state.get("registering"):
                st.subheader("注册账号")
                show_registration_form()
                if st.button("返回登录", width="stretch"):
                    st.session_state.pop("registering", None)
                    st.rerun()
                st.stop()
            st.caption("患者在这里填写随访、阅读报告。医生可以登记出院、审核随访。")
            with st.form("login_form", border=False):
                # 选择身份只表明想进入哪个工作区；后端会再次验证实际账号角色。
                identity = st.radio("登录身份", ["患者", "医生"], horizontal=True)
                name = st.text_input("姓名", max_chars=64)
                password = st.text_input("密码", type="password", max_chars=128)
                submitted = st.form_submit_button("登录", type="primary", width="stretch")
            if st.button("注册新账号", width="stretch"):
                st.session_state["registering"] = True
                st.rerun()

    if submitted:
        if not name.strip() or not password:
            st.warning("请输入姓名和密码。")
        else:
            role = "patient"
            if identity == "医生":
                role = "doctor"
            sign_in(name, password, role)


def show_logout(token):
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
            api_client.request_api("POST", "/auth/logout", token=token)
        except requests.RequestException as error:
            show_request_error(error)
            st.stop()
        clear_login()
        st.rerun()
