"""患者档案和出院登记。"""

import json
from uuid import uuid4

import requests
import streamlit as st

from utils import api_client
from utils.ui_session import get_view, show_request_error
from utils.ui_components import (
    doctor_label,
    patient_summary,
    show_contact_phone,
    show_record,
    task_card,
)


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


def show_discharge_registration(token, patient_id):
    """用医生凭证为当前患者登记出院。点击保存才发送 HTTP；失败可用同一请求号重试，无返回值。"""
    scope = (st.session_state["user"]["user_id"], patient_id)  # 切换医生或患者时隔离表单与回执。
    entry = st.session_state.get("discharge_form")
    if entry and entry["scope"] != scope:
        st.session_state.pop("discharge_form")
        entry = None
    if entry and "scheduled_at" in entry:
        st.success("出院已登记。首轮随访计划于 " + entry["scheduled_at"] + "（北京时间）开放。")
        st.session_state.pop("discharge_form")  # 旧表单已关闭，下一次必须明确点击“登记出院”。
        entry = None
    if entry is None:
        if st.button("登记出院", key="start_discharge"):
            st.session_state["discharge_form"] = {"scope": scope, "request_id": str(uuid4())}
            st.rerun()  # 第一次点击只打开空表单，不向后端提交任何内容。
        return

    form_id = entry["request_id"]  # 请求失败或普通重跑沿用原编号；新登记使用新编号。
    with st.form("discharge_" + form_id):
        st.write("登记本次出院")
        st.caption("随访分别在出院后第 7、14、30 天上午 9 点开放，每次填写时间为 48 小时。")
        day = st.date_input("出院日期", format="YYYY-MM-DD")
        diagnosis = st.text_input("出院诊断", max_chars=500)
        instructions = st.text_area("出院医嘱", max_chars=4000)
        medications = st.text_area("出院用药", max_chars=4000, help="可逐行填写药名、剂量和用法。")
        submitted = st.form_submit_button("保存出院登记", type="primary")
        cancelled = st.form_submit_button("取消登记")
    if cancelled:
        st.session_state.pop("discharge_form")
        st.rerun()
    if submitted:
        if not day or not diagnosis.strip():
            st.warning("请填写出院日期和出院诊断。")
            return
        data = {
            "request_id": form_id, "discharge_date": day.isoformat(), "discharge_diagnosis": diagnosis.strip(),
            "discharge_instructions": instructions.strip() or None, "discharge_medications": medications.strip() or None,
        }
        try:
            result = api_client.request_api("POST", f"/patients/{patient_id}/discharges", token=token, data=data).json()
        except requests.RequestException as error:
            show_request_error(error)  # 登录失效由公共处理清理，否则保留输入方便重试。
            return
        entry["scheduled_at"] = result["scheduled_at"]
        st.rerun()  # 回到同一档案，重新读取刚生成的 D7 安排和出院资料。


def show_patient_profile(token, detail):
    """展示已获授权的患者资料、出院、随访与正式记录；主管医生可从这里打开登记表单。"""
    patient = detail["patient"]
    st.title(patient["patient_name"] + "的档案")
    st.write(patient_summary(patient))
    # 医生从已授权档案联系当前患者。这里不使用医生或其他家庭成员的号码填空。
    with st.container(key="workspace_contact_patient"):
        st.write("患者联系电话")
        show_contact_phone(patient.get("phone"))
    st.write("主管医生　" + doctor_label(patient.get("primary_doctor")))
    st.write("既往病史", patient.get("medical_history") or "未提供")
    st.subheader("随访安排与报告")
    if patient.get("is_primary"):
        show_discharge_registration(token, patient["patient_id"])  # 后端会再次校验主管关系。
    # 多次随访优先处理当前待办，再看其他任务；两组内保留接口提供的时间顺序。
    pending, others = [], []
    for task in detail["tasks"]:
        if task["task_status"] in ("submitted", "in_review"):
            pending.append(task)
        else:
            others.append(task)
    for index, task in enumerate(pending + others):
        task_card(token, task, primary=bool(pending) and index == 0, doctor=True)
    if not detail["tasks"]:
        st.caption("目前没有可查看的随访任务。")
    show_profile_records(token, detail, may_supplement=patient.get("is_primary", False))


def show_profile_records(token, detail, may_supplement=False):
    discharges = detail["discharges"]
    with st.expander(f"出院资料（{len(discharges)}）"):
        if not discharges:
            st.caption("暂无出院记录。")
        for row in discharges:
            st.markdown("**" + str(row.get("discharge_date") or "日期未提供") + " 出院**")
            st.write("出院诊断", row.get("discharge_diagnosis") or "未提供")
            st.write("出院医嘱", row.get("discharge_instructions") or "未提供")
            show_discharge_medications(row.get("discharge_medications"))
            high, low = row.get("baseline_bp_systolic"), row.get("baseline_bp_diastolic")
            if high is not None or low is not None:
                st.write(f"出院基线血压　{high or '—'} / {low or '—'} mmHg")
            if row.get("baseline_heart_rate") is not None:
                st.write(f"出院基线心率　{row['baseline_heart_rate']} 次/分")
            if may_supplement:
                show_discharge_supplement(token, detail["patient"]["patient_id"], row)
            st.divider()
    records = detail["records"]
    with st.expander(f"正式随访记录（{len(records)}）"):
        if not records:
            st.caption("暂无随访记录。")
        for record in records:
            show_record(record)


def show_discharge_supplement(token, patient_id, row):
    fields = {"discharge_instructions": "出院医嘱", "discharge_medications": "出院用药"}
    missing = [field for field in fields if not str(row.get(field) or "").strip()]
    if not missing:
        return
    discharge_id = row["discharge_id"]
    with st.popover("补充出院资料", key="supplement_" + discharge_id):
        with st.form("supplement_form_" + discharge_id):
            values = {field: st.text_area(fields[field], max_chars=4000).strip() for field in missing}
            submitted = st.form_submit_button("保存补充资料")
        if submitted:
            changes = {field: value for field, value in values.items() if value}
            if not changes:
                st.warning("请填写要补充的内容。")
                return
            try:
                api_client.request_api("PATCH", f"/patients/{patient_id}/discharges/{discharge_id}",
                                       token=token, data=changes)
            except requests.RequestException as error:
                show_request_error(error)
                return
            st.rerun()


def show_patient_settings(token):
    st.title("我的资料")
    profiles = get_view(token, "/patients")
    if not profiles:
        if profiles == []:
            st.info("该账号尚未关联患者档案。")
        return
    patient_id = profiles[0]["patient_id"]
    detail = get_view(token, f"/patients/{patient_id}")
    if not detail:
        return
    patient = detail["patient"]
    st.subheader(patient["patient_name"])
    st.caption(patient_summary(patient))
    with st.form("profile_" + patient_id):
        phone = st.text_input("联系电话", value=patient.get("phone") or "", max_chars=32)
        history = st.text_area("既往病史", value=patient.get("medical_history") or "", max_chars=2000)
        saved = st.form_submit_button("保存个人资料", type="primary")
    if saved:
        values = {"phone": phone.strip(), "medical_history": history.strip()}
        changes = {key: value for key, value in values.items() if value != (patient.get(key) or "")}
        if changes:
            try:
                api_client.request_api("PATCH", f"/patients/{patient_id}", token=token, data=changes)
            except requests.RequestException as error:
                show_request_error(error)
            else:
                st.success("个人资料已保存。")
        else:
            st.info("个人资料没有改动。")
    show_profile_records(token, detail)
