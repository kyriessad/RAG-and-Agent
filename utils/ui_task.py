"""当前随访任务页面。"""

import streamlit as st

from utils.ui_session import EDITABLE_STATES, open_patient
from utils.ui_components import show_contact_phone, task_doctor, task_name, task_state_label
from utils.ui_draft import show_patient_draft
from utils.ui_report import show_report_workflow
from utils.ui_chat import show_chat


def show_task_page(token, task, doctor):
    """呈现本次随访的当前步骤；采集、人工修改与报告审核分别由明确操作触发。"""
    title = task_name(task)
    if doctor and task.get("patient_name"):
        title = task["patient_name"] + " · " + title
    st.title(title)
    st.caption(f"{task['followup_date']}　·　{task_state_label(task)}")
    if doctor:
        st.caption("任务负责医生　" + task_doctor(task))
    else:
        # 联系卡使用当下任务分配；已发布报告中的历史署名由报告视图单独显示。
        with st.container(horizontal=True, vertical_alignment="center", key="workspace_contact_task"):
            st.write("当前随访负责医生　" + task_doctor(task))
            with st.popover("联系医生", icon=":material/call:"):
                show_contact_phone(task.get("doctor_phone"))  # 联系操作可就地打开，让出正文空间。
    planned = task.get("scheduled_at") and not task.get("opened_at")
    if planned:
        st.info("预计 " + task["scheduled_at"] + "（北京时间）开放，开放后患者将收到站内提醒。")
    elif task.get("scheduled_at") and task["task_status"] in EDITABLE_STATES:
        st.caption("填写截止　" + task["expires_at"] + "（北京时间）")
    elif not task.get("scheduled_at") and task.get("expires_at"):
        st.caption("截止时间　" + task["expires_at"])
    if doctor:
        if task.get("patient_id") and st.button("查看患者档案 · " + str(task.get("patient_name") or ""), key="task_patient"):
            open_patient(task["patient_id"])
        if planned:
            return  # 预排期间是医生只读计划，此时尚无可审核的内容。
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
                if status == "needs_info":
                    st.caption("可以直接告诉助手医生要求补充的内容。")
                elif task.get("draft_revision"):
                    st.caption("上次说过的情况已经记下了，可以接着聊。")
                else:
                    st.caption("最近一次是什么时候量的血压？高压和低压各是多少？还没量也可以告诉我。")
            show_chat(token, "collect", task_id=task["task_id"])
