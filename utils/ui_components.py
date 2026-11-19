"""随访工作区共用的显示组件。"""

import re

import streamlit as st

from utils.ui_session import EDITABLE_STATES, open_task


TASK_STATES = {
    "pending": "待随访", "in_progress": "进行中", "needs_info": "需补充",
    "submitted": "已提交", "in_review": "审核中", "completed": "已完成",
}


# 任务名字
def task_name(task):
    name = task.get("task_type") or "随访"
    if name.startswith("D") and name[1:].isdigit():
        return "出院后第 " + name[1:] + " 天随访"
    return name


# 获取医生信息
def doctor_label(doctor):
    """输入医生公开资料或 None，返回姓名、科室和职称；不请求网络。"""
    if not doctor:
        return "暂未提供"
    parts = []
    for field in ("name", "department", "title"):
        if doctor.get(field):
            parts.append(doctor[field])
    return " · ".join(parts) or "暂未提供"


# 获取当前任务关联的医生
def task_doctor(task):
    return doctor_label({"name": task.get("doctor_name"), "department": task.get("doctor_department"),
                         "title": task.get("doctor_title")})


# 显示联系电话
def show_contact_phone(phone):
    number = (phone or "").strip()  # 保留区号、空格等登记格式，复制时与档案一致。
    if not number:
        st.caption("暂未登记联系电话")
        return
    with st.container(horizontal=True, vertical_alignment="center"):
        # Streamlit 的代码框自带复制按钮；整行号码也能直接选中复制。
        st.code(number, language=None, width=210, wrap_lines=True)
        dial_number = re.sub(r"[\s()（）-]", "", number)
        if re.fullmatch(r"\+?[0-9]+", dial_number):
            # 点击后由设备的电话应用接管；含有分机说明等文字的原始号码仍能复制。
            st.link_button("拨打电话", "tel:" + dial_number, icon=":material/call:")


# 根据报告署名来源显示医生
def report_doctor_label(doctor, source):
    if source == "unverified":
        return "当次负责医生未核实"
    label = doctor_label(doctor)
    if source == "current_task":
        label += "（据当前任务分配）"
    return label


# 输入已授权的正式记录，显示读数、症状和提交版本，显示正式记录
def show_record(record):
    data = record.get("data") or {}
    high, low = record.get("systolic_bp"), record.get("diastolic_bp")
    if high is not None or low is not None:
        st.write(f"血压　{high if high is not None else '—'} / {low if low is not None else '—'} mmHg")
    else:
        measure = record.get("bp_measurement_status")
        st.write("血压　" + {"not_measured": "未测量", "measured": "已测量，未记录读数"}.get(measure, "未提供"))
    st.caption(f"提交时间　{record.get('submitted_at') or '未提供'}　·　提交版本 {record.get('revision') or '未提供'}")
    # 优先使用当前记录明确保存的值；缺失信息如实留空。
    for field, label in (("measured_at", "测量时间"), ("heart_rate", "心率"), ("symptoms", "症状"),
                         ("medication_adherence", "服药情况"), ("missed_dose_reason", "漏服原因"),
                         ("adverse_effects", "用药后不适"), ("risk_reason", "规则提示依据")):
        value = data.get(field)
        if value is None:
            value = record.get(field)
        if field == "medication_adherence" and value is not None:
            # 历史字段的 True/False 仅说明是否服过药，不能据此判断是否规律。
            labels = {"regular": "按时服药", "missed": "有漏服", "not_prescribed": "没有开药",
                      "True": "已服药，是否按时待核对", "False": "未服药，原因待核对"}
            value = labels.get(str(value), value)
        if value is not None and value != "":
            st.write(label, value)


def task_action(task):
    """按服务端状态和已保存草稿决定按钮文字；不修改状态或自动提交。"""
    status = task["task_status"]
    if status == "needs_info":
        return "补充随访信息"
    if status in ("submitted", "in_review"):
        return "查看审核进度"
    if status == "completed":
        if task.get("report_status") == "published":
            return "阅读报告"
        return "查看随访"
    if task.get("draft_needs_clarification") is not None or task.get("draft_revision") or status == "in_progress":
        return "继续随访"
    return "开始随访"


def task_priority(task):
    """给患者首页排序，先补充退回资料，再继续草稿或开始最近任务；只返回排序键。"""
    rank = 3
    if task.get("is_overdue"):
        rank = -1  # 超过本次时限仍可补交，优先展示。
    elif task["task_status"] == "needs_info":
        rank = 0
    elif task["task_status"] in EDITABLE_STATES:
        rank = 2
        if task.get("draft_needs_clarification") is not None or task.get("draft_revision") or task["task_status"] == "in_progress":
            rank = 1
    return rank, task.get("followup_date") or "", task["task_id"]


def task_state_label(task):
    """输入授权任务摘要，返回页面状态文字；计划时间与审核进度分别判断，只读内存。"""
    if task.get("scheduled_at") and not task.get("opened_at"):
        return "待开放"
    if task.get("is_overdue"):
        return "已逾期，仍可补交"
    return TASK_STATES.get(task["task_status"], "状态待确认")


def task_card(token, task, primary=False, doctor=False, notice=None):
    """用授权任务和可选提醒生成操作卡；点击才请求详情，失败留在当前页，不写库。"""
    card_key = "followup_item_" + task["task_id"]
    if primary:
        card_key = "primary_task_" + task["task_id"]  # 当前最需要处理的一项使用独立视觉重点。
    with st.container(border=True, key=card_key):
        description, action = st.columns([3, 1], vertical_alignment="center")
        with description:
            title = task_name(task)
            if doctor and task.get("patient_name"):
                title = task["patient_name"] + " · " + title  # 医生打开随访前可核对患者姓名。
            st.markdown("**" + title + "**")
            status_label = task_state_label(task)
            color = "blue"
            if task.get("is_overdue") or task["task_status"] == "needs_info":
                color = "orange"
            elif task["task_status"] == "completed":
                color = "green"
            st.caption(f":{color}-badge[{status_label}]　{task['followup_date']}")
            if task.get("scheduled_at"):
                if not task.get("opened_at"):
                    st.write("预计开放　" + task["scheduled_at"] + "（北京时间）")
                elif task["task_status"] in EDITABLE_STATES:
                    st.write("填写截止　" + task["expires_at"] + "（北京时间）")
            if notice:
                st.info(notice["message"], icon=":material/notifications:")
            if doctor:
                if task.get("report_status") == "approved":
                    st.caption("审核已通过，等待发布给患者。")
            elif task.get("report_status") == "published":
                st.write("本次负责医生　" + report_doctor_label(task.get("report_doctor"), task.get("report_doctor_source")))
            else:
                st.write("本次负责医生　" + task_doctor(task))
            if task["task_status"] == "needs_info":
                if task.get("review_note"):
                    label = "已请患者补充" if doctor else "医生请你补充"
                    st.write(label, task["review_note"])  # 两端看到同一条意见，按阅读者身份称呼。
                else:
                    st.caption("本次随访需要补充，目前没有可查看的具体意见。")
            elif not doctor and task["task_status"] in ("submitted", "in_review"):
                if task.get("submitted_at"):
                    st.caption("提交时间　" + task["submitted_at"])
                if task["task_status"] == "submitted":
                    st.write("记录已保存，正在等待整理报告。")
                else:
                    st.write("报告已进入审核流程，发布后可以阅读。")
            elif task.get("draft_needs_clarification") is not None and task["task_status"] in EDITABLE_STATES:
                progress = "已保存，可继续填写"
                if task.get("draft_needs_clarification") == 0:
                    progress = "已保存，请核对并确认提交"
                st.caption(progress)
            if task["task_status"] == "completed" and task.get("report_date"):
                st.caption("报告日期　" + task["report_date"])
        with action:
            label = task_action(task)
            if task.get("is_overdue"):
                label = "继续补交"
            if doctor:
                label = doctor_task_action(task)
            style = "secondary"
            if primary:
                style = "primary"
            if st.button(label, key="enter_task_" + task["task_id"], type=style, width="stretch"):
                open_task(token, task["task_id"])


def doctor_task_action(task):
    """输入医生有权查看的任务，返回与当前报告状态相符的按钮文字；只读内存。"""
    if task.get("report_status") == "pending_review":
        return "审核报告"
    if task.get("report_status") == "approved":
        return "前往发布"
    if task["task_status"] == "submitted":
        return "查看生成进度"
    if task.get("report_status") == "published":
        return "阅读报告"
    return "查看本次随访"


def patient_summary(patient):
    """用档案已有字段显示性别、年龄和诊断；不推算未知资料。"""
    parts = []
    if patient.get("gender"):
        gender = str(patient["gender"])
        labels = {"male": "男", "female": "女", "unknown": "性别未记录"}
        parts.append(labels.get(gender, gender))
    if patient.get("age") is not None:
        parts.append(str(patient["age"]) + " 岁")
    if patient.get("diagnosis"):
        parts.append(patient["diagnosis"])
    return " · ".join(parts) or "基本资料暂未提供"


# 展示规则提醒，输入含 risk 和 notification_saved 的字典，返回 None
def show_risk_notice(content):
    """
    例如 risk.level="high" 时显示就医提示；不调用模型、不发请求、不修改病历。
    提醒独立于模型回答，模型失败或清除临时回答时仍须保留。
    """
    risk = content.get("risk", {})  # 缺少规则结果时不推断为安全或低风险。
    message = risk.get("patient_message", "")  # 文案由服务端规则决定，页面不判断血压。
    if risk.get("level") == "high" and message:
        st.error(message)  # 明确红旗提示使用醒目的样式。
    elif risk.get("level") == "attention" and message:
        st.warning(message)  # 需要关注但未命中红旗的规则提示。
    if risk.get("level") in ("high", "attention"):
        # 多条规则可能同时命中，逐条显示，方便患者理解为何收到提醒。
        for reason in risk.get("reasons", []):
            st.write("触发原因：" + reason)  # 保留可解释的规则依据，不展示推理过程。
        # 明确的 False 表示这次写库失败；未传这个标记时不能推断通知投递结果。
        if content.get("notification_reason") == "no_doctor":
            st.warning("尚未关联主管医生，本次无法发送站内提醒。")
        elif content.get("notification_reason") == "no_task":
            st.warning("当前没有关联随访任务，本次未发送医生站内提醒。请直接联系医生。")
        elif content.get("notification_saved") is False:
            st.warning("医生站内提醒尚未保存成功，请直接联系医生；不要等待助手回答。")
        elif content.get("notification_saved") is True:
            st.caption("负责医生的站内提醒已保存。")
