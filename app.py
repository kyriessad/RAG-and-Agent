"""高血压出院随访助手的 Streamlit 入口。"""

from pathlib import Path

import streamlit as st

from utils.ui_auth import show_login_page, show_logout
from utils.ui_chat import show_assistant
from utils.ui_conversations import show_conversation_sidebar
from utils.ui_profile import show_patient_settings
from utils.ui_session import verified_user, workspace_context


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
    show_conversation_sidebar(token, doctor)
    st.divider()
    show_logout(token)

task, profile = workspace_context(token, doctor)
if workspace == "settings" and not doctor:
    show_patient_settings(token)
else:
    show_assistant(token, doctor, task, profile)
