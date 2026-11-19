"""高血压出院随访助手的 Streamlit 入口。"""

from pathlib import Path

import streamlit as st

from utils.ui_auth import show_login_page, show_logout
from utils.ui_chat import show_assistant
from utils.ui_home import show_patient_home, show_physician_home
from utils.ui_profile import show_patient_profile, show_patient_settings
from utils.ui_session import go_home, verified_user, workspace_context
from utils.ui_task import show_task_page


st.set_page_config(page_title="高血压随访助手", page_icon="🩺", layout="wide")
st.html(Path(__file__).parent / ".streamlit" / "workspace.css")

token = st.session_state.get("token")
if not token:
    show_login_page()
    st.stop()

user = verified_user(token)
doctor = user["role"] == "doctor"
workspace = st.session_state.get("workspace", "home")
with st.sidebar:
    st.title("随访助手")
    st.write(f"{user['username']} · {'医生' if doctor else '患者'}")
    if st.button("医生工作台" if doctor else "我的随访", key="return_tasks", width="stretch"):
        go_home()
    if st.button("问助手", key="open_assistant", width="stretch"):
        st.session_state["workspace"] = "assistant"
        st.rerun()
    if not doctor and st.button("我的资料", key="open_settings", width="stretch"):
        go_home("settings")
    st.divider()
    show_logout(token)

task, profile = workspace_context(token, doctor)
if workspace == "settings" and not doctor:
    show_patient_settings(token)
elif workspace == "assistant":
    show_assistant(token, doctor, task, profile)
elif task:
    show_task_page(token, task, doctor)
elif profile:
    show_patient_profile(token, profile)
elif doctor:
    show_physician_home(token)
else:
    show_patient_home(token)
