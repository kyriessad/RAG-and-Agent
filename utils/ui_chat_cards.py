"""在对话中核对 Agent 的业务提议，办理时读取当前授权的业务记录。"""

from datetime import date
from uuid import UUID

import requests
import streamlit as st

from utils import api_client
from utils.ui_components import doctor_task_action, task_action, task_name, task_state_label
from utils.ui_draft import show_patient_draft
from utils.ui_report import show_report_sheet, show_report_workflow
from utils.ui_session import EDITABLE_STATES, get_view, open_patient, open_task, show_request_error


def show_task_actions(token, task, doctor):
    """在任务对话中绘制本次任务的核对单或报告；由调用方绘制聊天输入框。"""
    task_id = task["task_id"]
    if task.get("scheduled_at") and not task.get("opened_at"):
        st.info(f"预计 {task['scheduled_at']}（北京时间）开放。")
        return
    if st.session_state.pop("draft_submit_notice", None):
        st.success("本次随访已提交，记录已保存。")
    if doctor or task["task_status"] in ("submitted", "in_review", "needs_info", "completed"):
        intent = st.session_state.get("chat_review_prefill") or {}
        if (intent.get("user_id"), intent.get("task_id")) != (st.session_state["user"]["user_id"], task_id):
            intent = None
        show_report_workflow(token, task_id, doctor, intent)
    if not doctor and task["task_status"] in EDITABLE_STATES:
        if task.get("expires_at"):
            st.caption("本次随访填写截止　" + task["expires_at"])
        show_patient_draft(token, task_id)


def _open_panel(label, key):
    if st.session_state.get("chat_open_card") == key:
        if st.button("收起", key="close_" + key, type="tertiary"):
            st.session_state.pop("chat_open_card", None)
            return False
        return True
    if st.button(label, key="open_" + key):
        st.session_state["chat_open_card"] = key
    return st.session_state.get("chat_open_card") == key


def _discharge_card(token, card, key, message_id, card_index):
    patient_id = card["patient_id"]
    registration = card["kind"] == "register_discharge"
    label = "核对出院登记" if registration else "核对补充资料"
    same_patient_chat = st.session_state.get("selected_patient") == patient_id and not st.session_state.get("selected_task")
    if not same_patient_chat and message_id is not None:
        if st.button(label, key="select_" + key):
            try:
                selected = api_client.request_api("POST", "/chat/cards/select", token=token,
                                                  data={"message_id": message_id, "card_index": card_index}).json()
            except requests.RequestException as error:
                show_request_error(error)
                return
            owner = st.session_state["user"]["user_id"]
            open_patient(selected["patient_id"], open_card=f"chat_card_{owner}_msg_{selected['message_id']}_0")
        return
    if not _open_panel(label, key):
        return
    detail = get_view(token, f"/patients/{patient_id}")
    if not detail:
        return
    if not detail["patient"].get("is_primary"):
        st.info("本项操作需要当前主管医生办理。")
        return
    fields = card.get("fields") or {}
    if registration:
        try:
            request_id = str(UUID(card["request_id"]))
        except (KeyError, ValueError, TypeError):
            st.info("请重新请求一张出院登记卡。")
            return
        discharge_id = "DIS-" + request_id
    else:
        discharge_id = card["discharge_id"]
    row = next((item for item in detail["discharges"] if item["discharge_id"] == discharge_id), None)
    if registration and row:
        st.success("出院已登记。")
        for task in detail["tasks"]:
            if task["discharge_id"] == discharge_id:
                st.write(f"{task_name(task)}　{task['followup_date']}")
        return
    if not registration and not row:
        st.info("当前档案中没有这条可补充的出院记录。")
        return
    labels = {"discharge_instructions": "出院医嘱", "discharge_medications": "出院用药"}
    if row:
        st.write(f"{row['discharge_date']} 出院　{row.get('discharge_diagnosis') or ''}")
        labels = {name: label for name, label in labels.items() if not str(row.get(name) or "").strip()}
        if not labels:
            st.success("出院资料已补全。")
            return
    with st.form("discharge_form_" + key):
        if registration:
            day = st.text_input("出院日期", value=fields.get("discharge_date") or "", placeholder="YYYY-MM-DD")
            diagnosis = st.text_input("出院诊断", value=fields.get("discharge_diagnosis") or "", max_chars=500)
        data = {name: st.text_area(label, value=fields.get(name) or "", max_chars=4000).strip()
                for name, label in labels.items()}
        submitted = st.form_submit_button("确认登记出院" if registration else "确认保存补充资料", type="primary")
    if not submitted:
        return
    if registration:
        try:
            parsed_day = date.fromisoformat(day.strip())
        except ValueError:
            st.warning("请按 YYYY-MM-DD 填写有效的出院日期。")
            return
        if not diagnosis.strip():
            st.warning("请填写出院诊断。")
            return
        data.update(request_id=request_id, discharge_date=parsed_day.isoformat(), discharge_diagnosis=diagnosis.strip())
    data = {field: value for field, value in data.items() if value}
    if not data:
        st.warning("请填写要补充的资料。")
        return
    path = f"/patients/{patient_id}/discharges"
    try:
        if registration:
            api_client.request_api("POST", path, token=token, data=data)
        else:
            api_client.request_api("PATCH", f"{path}/{discharge_id}", token=token, data=data)
    except requests.RequestException as error:
        show_request_error(error)
        return
    st.rerun()


def _risk_card(token, card, key):
    if not _open_panel("查看风险提醒", key):
        if card.get("message"):
            st.write(card["message"])
        return
    notices = get_view(token, "/notifications")
    if notices is None:
        return
    notice = next((item for item in notices if item["id"] == card["notice_id"]), None)
    if not notice:
        st.info("这条通知当前不可查看。")
        return
    st.warning(notice["message"])
    st.caption(notice["created_at"])
    if notice.get("task_id") and st.button("查看相关随访", key="risk_task_" + key):
        open_task(token, notice["task_id"])
    if notice.get("handled_at"):
        st.success("已处理　" + notice["handled_at"])
    elif st.button("标记已处理", key="handle_" + key, type="primary"):
        try:
            api_client.request_api("POST", f"/notifications/{notice['id']}/handle", token=token)
        except requests.RequestException as error:
            show_request_error(error)
        else:
            st.rerun()


def show_chat_card(token, card, key, message_id=None, card_index=None):
    """key 由调用方按消息、卡片序号生成，避免两次查询同一任务时控件冲突。"""
    user = st.session_state["user"]
    doctor = user["role"] == "doctor"
    key = f"chat_card_{user['user_id']}_{key}"
    kind = card.get("kind")
    if kind not in ("task", "patient", "report", "risk_notice", "register_discharge", "supplement_discharge"):
        return
    if kind not in ("task", "report") and not doctor:
        return
    with st.container(border=True):
        if doctor and card.get("patient_id"):
            st.write(f"{card.get('patient_name') or '患者'}　·　{card['patient_id']}")
            if card.get("gender") or card.get("age") is not None:
                age = f"{card['age']} 岁" if card.get("age") is not None else ""
                st.caption("　".join(filter(None, [card.get("gender"), age])))
        if kind == "task":
            st.write(task_name(card))
            available = card.get("available", True)
            status = task_state_label(card) if available else "待开放"
            st.caption(f"{card.get('followup_date') or ''}　·　{status}")
            if available:
                label = doctor_task_action(card) if doctor else task_action(card)
                if label == "开始随访":
                    label = "开始本次随访"
                if st.button(label, key=key):
                    open_task(token, card["task_id"])
            elif st.button("查看当前开放状态", key=key):
                # 历史计划保存的是当时的待开放状态，点开时从本人当前可进入的任务中定位。
                tasks = get_view(token, "/tasks")
                if tasks is not None:
                    if any(row["task_id"] == card["task_id"] for row in tasks):
                        open_task(token, card["task_id"])
                    else:
                        st.info("当前尚无可进入的这次随访。")
        elif kind == "patient":
            if st.button("进入患者对话", key=key):
                open_patient(card["patient_id"])
        elif kind == "report":
            task_id = card["task_id"]
            st.caption(f"{task_name(card)}　{card.get('followup_date') or ''}")
            if doctor:
                if st.button("核对随访报告", key=key):
                    intent = dict(card, user_id=user["user_id"], card_key=key)
                    open_task(token, task_id, review_card=intent)
            elif _open_panel("展开医生发布的原文", key):
                report = get_view(token, f"/tasks/{task_id}/report")
                if report and report.get("task_status") == "completed" and report.get("content"):
                    show_report_sheet(report, "我的最终随访报告", "_" + key)
                elif report:
                    st.info("当前没有可阅读的已发布报告。")
        elif kind == "risk_notice":
            _risk_card(token, card, key)
        else:
            if kind == "supplement_discharge":
                st.caption(f"{card.get('discharge_date') or ''} 出院　·　{card['discharge_id']}")
            _discharge_card(token, card, key, message_id, card_index)
