"""聊天首页的会话切换与站内提醒。"""

import streamlit as st

from utils.ui_chat_cards import show_chat_card
from utils.ui_components import doctor_task_action, task_name, task_priority, task_state_label
from utils.ui_session import EDITABLE_STATES, get_view, go_home, open_patient, open_task


def show_conversation_sidebar(token, doctor):
    if st.button("总对话", key="return_tasks", width="stretch", icon=":material/chat:"):
        go_home()
    if doctor:
        patients = get_view(token, "/patients") or []
        if patients:
            labels = {p["patient_id"]: f"{p['patient_name']} · {p['patient_id']}" for p in patients}
            with st.popover("按患者找对话", width="stretch"):
                selected = st.selectbox("搜索或选择患者", list(labels), format_func=labels.get,
                                        index=None, key="conversation_patient_picker")
                if st.button("进入患者对话", disabled=selected is None):
                    open_patient(selected)
    elif st.button("我的资料", key="open_settings", width="stretch", icon=":material/person:"):
        go_home("settings")

    conversations = (get_view(token, "/chat/conversations") or {}).get("conversations", [])
    scoped = [c for c in conversations if c.get("task_id") or c.get("patient_id")]
    if scoped:
        st.caption("专属对话")
    for c in scoped:
        if c.get("task_id"):
            if st.button(c["title"], key="conversation_task_" + c["task_id"], width="stretch"):
                open_task(token, c["task_id"])
        elif st.button(c["title"], key="conversation_patient_" + c["patient_id"], width="stretch"):
            open_patient(c["patient_id"])


def reminder_task(token, task, doctor):
    label = doctor_task_action(task) if doctor else "进入随访"
    text = f"{task_name(task)} · {task['followup_date']} · {task_state_label(task)}"
    if doctor:
        text = f"{task.get('patient_name') or ''}　{text}"
    left, right = st.columns([3, 1], vertical_alignment="center")
    left.write(text)
    if right.button(label, key="enter_task_" + task["task_id"], width="stretch"):
        open_task(token, task["task_id"])


@st.fragment(run_every="60s")
def show_home_reminders(token, doctor):
    if doctor:
        risks = [n for n in get_view(token, "/notifications") or [] if not n.get("handled_at")]
        if risks:
            with st.expander(f"待处理风险提醒（{len(risks)}）", icon="⚠️", expanded=len(risks) == 1):
                for notice in risks:
                    card = dict(notice, kind="risk_notice", notice_id=notice["id"])
                    show_chat_card(token, card, f"home_risk_{notice['id']}")

    tasks = get_view(token, "/tasks") or []
    if doctor:
        reviews = [t for t in tasks if t["task_status"] in ("submitted", "in_review")]
        if reviews:
            with st.expander(f"待审核或发布（{len(reviews)}）", expanded=len(reviews) == 1):
                for task in reviews:
                    reminder_task(token, task, doctor=True)
    notices = {n["task_id"]: n for n in get_view(token, "/notifications/schedule") or []}
    pending = [t for t in tasks if t["task_status"] in EDITABLE_STATES and
               (not t.get("scheduled_at") or t.get("opened_at"))]
    if pending:
        pending.sort(key=task_priority)
        title = "到期和逾期随访" if doctor else "你有随访待填写"
        with st.expander(f"{title}（{len(pending)}）", expanded=not doctor and len(pending) == 1):
            for task in pending:
                notice = notices.get(task["task_id"])
                if notice and notice.get("kind") == "overdue":
                    st.caption(notice["message"])
                reminder_task(token, task, doctor)
