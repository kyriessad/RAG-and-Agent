"""患者首页与医生工作台。"""

import requests
import streamlit as st

from utils import api_client
from utils.ui_session import get_view, open_patient, open_task, show_request_error
from utils.ui_components import (
    doctor_label,
    doctor_task_action,
    patient_summary,
    show_contact_phone,
    task_card,
    task_priority,
)


@st.fragment(run_every="60s")
def show_patient_home(token):
    """用 token 读取本人随访和主管医生联系方式；请求失败会提示，可继续查看其余已授权资料。"""
    st.title("我的随访")
    next_step = st.container()  # 把当前任务放在联系卡之前，手机上进入首页也先看到可执行的操作。
    profiles = get_view(token, "/patients")
    if profiles:
        # 名单不带电话，只有当前账号能打开的档案详情才返回主管医生的联系资料。
        detail = get_view(token, "/patients/" + profiles[0]["patient_id"])
        if detail:
            primary_doctor = detail["patient"].get("primary_doctor")
            with st.container(key="workspace_contact_primary"):
                st.write("主管医生　" + doctor_label(primary_doctor))
                if primary_doctor:
                    show_contact_phone(primary_doctor.get("phone"))
    elif profiles == []:
        st.caption("该账号尚未关联患者档案。")
    tasks = get_view(token, "/tasks")
    if tasks is None:
        if st.button("重新加载任务", key="reload_tasks"):
            st.rerun()
        return
    upcoming, reports, closed = [], [], []
    for task in tasks:
        if task["task_status"] == "completed" and task.get("report_status") == "published":
            reports.append(task)
        elif task["task_status"] == "completed":
            closed.append(task)  # 某些旧任务已结束，但目前没有可以阅读的已发布报告。
        else:
            upcoming.append(task)
    upcoming.sort(key=task_priority)
    # 只在首页轮询，保持已打开页面的提醒更新，患者手工编辑草稿时不会被计时刷新打断。
    notices = get_view(token, "/notifications/schedule")
    notices_by_task = {}
    for notice in notices or []:
        notices_by_task[notice["task_id"]] = notice
    # 先按报告日期定位最新一份，没有待填任务时可直接打开。
    def report_order(task):
        """输入报告摘要，返回排序日期；旧报告缺少发布日期时使用随访日期。"""
        return task.get("report_date") or task.get("followup_date") or ""
    reports.sort(key=report_order, reverse=True)
    if upcoming:
        with next_step:
            st.subheader("下一步")
            task_card(token, upcoming[0], primary=True, notice=notices_by_task.get(upcoming[0]["task_id"]))
            if len(upcoming) > 1:
                with st.expander("其他随访安排", expanded=False):
                    for task in upcoming[1:]:
                        task_card(token, task, notice=notices_by_task.get(task["task_id"]))
    elif reports:
        with next_step:
            st.subheader("最新随访报告")
            task_card(token, reports.pop(0), primary=True)  # 下方历史列表只渲染其余报告，避免重复按钮。
    else:
        with next_step:
            st.subheader("下一步")
            if closed:
                st.info("当前随访已结束，暂无已发布报告。")
            else:
                st.info("目前没有随访安排。")
    if reports:
        st.subheader("历次随访报告")
    for task in reports:
        task_card(token, task)
    if closed:
        with st.expander("其他已结束的随访"):
            for task in closed:
                task_card(token, task)


@st.fragment(run_every="60s")
def show_physician_home(token):
    """用登录凭证读取患者名册或通知；“待我处理”按患者筛选待办，查询失败时提示，不写库。"""
    st.title("医生工作台")
    section = st.radio("工作台栏目", ["我的患者", "随访提醒", "风险通知"], horizontal=True, key="doctor_section")
    if section == "随访提醒":
        notices = get_view(token, "/notifications/schedule")
        if notices == []:
            st.info("当前没有待处理的随访提醒。")
        for notice in notices or []:
            with st.container(border=True):
                st.write(notice["patient_name"])
                if notice["kind"] == "overdue":
                    st.warning(notice["message"])
                else:
                    st.info(notice["message"])
                st.caption("提醒时间　" + notice["created_at"] + "（北京时间）")
                with st.container(horizontal=True):
                    if st.button("联系患者", key="contact_overdue_" + notice["task_id"]):
                        open_patient(notice["patient_id"])
                    if st.button("查看本次随访", key="overdue_task_" + notice["task_id"]):
                        open_task(token, notice["task_id"])
        return
    if section == "风险通知":
        if st.button("刷新风险通知", key="reload_notifications"):
            st.rerun()
        notices = get_view(token, "/notifications")
        if notices == []:
            st.info("当前没有风险通知。")
        for notice in notices or []:
            with st.container(border=True):
                st.markdown("**" + notice["patient_name"] + "**")
                st.warning(notice["message"])
                st.caption(notice["created_at"])
                notice_id = notice["id"]
                with st.container(horizontal=True):
                    if notice.get("task_id"):
                        if st.button("查看相关随访", key=f"notice_{notice_id}"):
                            open_task(token, notice["task_id"])
                    elif st.button("查看患者档案", key=f"notice_{notice_id}"):
                        open_patient(notice["patient_id"])
                    if notice.get("handled_at"):
                        st.caption("已处理　" + notice["handled_at"] + "（UTC）")
                    elif st.button("标记已处理", key=f"handle_notice_{notice_id}"):
                        try:
                            api_client.request_api("POST", f"/notifications/{notice_id}/handle", token=token)
                        except requests.RequestException as error:
                            show_request_error(error)
                        else:
                            st.rerun()
        return
    patients = get_view(token, "/patients")
    if patients is None:
        return
    # 名册先提供“哪位患者需要我处理”的汇总；只有确有待办时，才另读授权任务以生成直达按钮。
    review_count = 0
    for patient in patients:
        if patient.get("has_review"):
            review_count += 1
    st.caption(f"关联患者 {len(patients)} 位　·　待我处理 {review_count} 位")
    review_tasks = {}
    if review_count:
        for task in get_view(token, "/tasks") or []:
            if task["task_status"] in ("submitted", "in_review") and task.get("patient_id"):
                review_tasks.setdefault(task["patient_id"], []).append(task)
    search_col, filter_col = st.columns([1, 2])
    search = search_col.text_input("搜索患者姓名", key="patient_search").strip()
    categories = ["全部", "待我处理", "逾期待联系", "待患者填写", "待开放", "已完成随访", "暂无任务"]
    category = filter_col.selectbox("按随访状态筛选", categories, key="patient_filter")
    flags = {"待我处理": "has_review", "逾期待联系": "has_overdue", "待患者填写": "has_to_fill",
             "待开放": "has_scheduled", "已完成随访": "has_completed"}
    count = 0
    for patient in patients:
        if search and search not in patient["patient_name"]:
            continue
        if category in flags and not patient.get(flags[category]):
            continue
        if category == "暂无任务" and patient["task_count"]:
            continue
        count += 1
        with st.container(border=True, key="roster_row_" + patient["patient_id"]):
            summary, status = st.columns([3, 2], vertical_alignment="center")
            # 姓名按钮用稳定患者编号作 key；同名不会打开对方的档案。
            if summary.button(patient["patient_name"], key="patient_" + patient["patient_id"],
                              help="打开患者档案", type="tertiary", icon=":material/person:"):
                open_patient(patient["patient_id"])
            # 一位患者可由多位医生分次随访；说明当前登录医生为什么能查看此人。
            relation = "我负责其中的随访"
            if patient["is_primary"]:
                relation = "我是主管医生"
            summary.caption(relation + "　·　" + patient_summary(patient))
            states = []
            for title, flag in flags.items():
                if patient.get(flag):
                    states.append(title)
            status.write(" · ".join(states) or "暂无任务")
            if patient.get("latest_followup_date"):
                status.caption("最近随访安排　" + patient["latest_followup_date"])
            pending = review_tasks.get(patient["patient_id"], [])
            if len(pending) == 1:
                work = pending[0]  # 单次待办可直接进入这项任务，再由详情接口重查归属。
                if status.button(doctor_task_action(work), key="roster_work_" + work["task_id"], type="primary"):
                    open_task(token, work["task_id"])
            elif pending:
                if status.button(f"查看 {len(pending)} 项待办", key="roster_pending_" + patient["patient_id"], type="primary"):
                    open_patient(patient["patient_id"])
    if not count:
        if category == "待我处理" and not review_count:
            st.info("当前没有待处理的随访。")
        else:
            st.info("当前没有符合条件的患者。")
