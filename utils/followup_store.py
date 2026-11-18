"""患者名册、档案与随访记录的授权只读查询，供页面和 Agent 共用。"""

from contextlib import closing
import json

from utils import schedule_store
from utils.user_store import get_connection


# 下面的查询共用同一组获授权任务。选择患者、任务只会在登录身份的范围内缩小结果。
# SQL 的 WITH 给这条临时查询结果取名 my_tasks，不会建表或写入任何数据。
MY_TASKS = """
WITH my_tasks AS (
    SELECT t.*, p.patient_id, p.name AS patient_name,
           doctor.name AS doctor_name, doctor.department AS doctor_department,
           doctor.title AS doctor_title,
           -- 首轮实际开放后才开始计时；提交即结束患者本轮待办，退回时重算截止时间。
           CASE WHEN t.scheduled_at IS NOT NULL AND t.opened_at IS NOT NULL
                     AND t.task_status IN ('pending','in_progress','needs_info')
                     AND t.expires_at <= :now THEN 1 ELSE 0 END AS is_overdue
    FROM followup_tasks t
    JOIN discharges h ON h.discharge_id=t.discharge_id
    JOIN patients p ON p.patient_id=h.patient_id
    JOIN doctors doctor ON doctor.doctor_id=t.doctor_id
    JOIN users u ON u.user_id=:user_id AND u.role=:role
    WHERE ((u.role='patient' AND p.user_id=u.user_id
              AND (t.scheduled_at IS NULL OR t.opened_at IS NOT NULL))
        OR (u.role='doctor' AND doctor.user_id=u.user_id))
      AND (:task_id IS NULL OR t.task_id=:task_id)
      AND (:patient_id IS NULL OR p.patient_id=:patient_id)
)
"""

# 只找最新正式记录对应的报告，重交之后不会把上次的退回意见当成本次的处理结果。
TASK_DETAILS = """
SELECT t.task_id,t.discharge_id,t.followup_date,t.task_type,t.task_status,t.expires_at,
       t.scheduled_at,t.opened_at,t.window_revision,t.is_overdue,
       t.patient_id,t.patient_name,t.doctor_id,t.doctor_name,t.doctor_department,t.doctor_title,
       COALESCE(g.revision,0) AS draft_revision,g.needs_clarification AS draft_needs_clarification,
       -- SQLite 自动登记提交/生成时间时使用 UTC，新排期任务在页面上统一与北京时间截止点对照。
       CASE WHEN t.scheduled_at IS NOT NULL THEN datetime(r.submitted_at,'+8 hours')
            ELSE r.submitted_at END AS submitted_at,
       CASE WHEN b.status='returned' AND t.task_status='needs_info' THEN b.review_note END AS review_note,
       CASE WHEN :role='doctor' OR (b.status='published' AND t.task_status='completed')
            THEN CASE WHEN t.scheduled_at IS NOT NULL THEN datetime(b.created_at,'+8 hours')
                      ELSE b.created_at END END AS report_date,
       CASE WHEN :role='doctor' THEN b.status
            WHEN b.status='published' AND t.task_status='completed' THEN 'published' END AS report_status,
       CASE WHEN b.status='published' AND t.task_status='completed'
            THEN json_extract(b.content_json,'$._responsible_doctor') END AS doctor_snapshot,
       CASE WHEN b.status='published' AND t.task_status='completed' THEN b.reviewer_id END AS signed_by
FROM my_tasks t
LEFT JOIN task_drafts g ON :role='patient' AND g.task_id=t.task_id
LEFT JOIN followup_records r ON r.record_id=(
    SELECT record_id FROM followup_records WHERE task_id=t.task_id
    ORDER BY revision DESC,submitted_at DESC,record_id DESC LIMIT 1)
LEFT JOIN reports b ON b.record_id=r.record_id AND b.task_id=t.task_id
ORDER BY t.followup_date,t.task_id
"""

# 患者本人可以查看自己的出院资料；医生只看与获授权任务相关的出院资料。
DISCHARGES = """
SELECT d.discharge_id,d.discharge_date,d.discharge_diagnosis,d.discharge_instructions,
       d.discharge_medications,d.baseline_bp_systolic,d.baseline_bp_diastolic,d.baseline_heart_rate,
       p.patient_id,p.name AS patient_name
FROM discharges d
JOIN patients p ON p.patient_id=d.patient_id
JOIN users u ON u.user_id=:user_id AND u.role=:role
WHERE (:patient_id IS NULL OR p.patient_id=:patient_id)
  AND ((:task_id IS NULL AND u.role='patient' AND p.user_id=u.user_id)
       OR EXISTS (SELECT 1 FROM my_tasks t WHERE t.discharge_id=d.discharge_id))
ORDER BY d.discharge_date DESC,d.discharge_id
"""

# 同一次任务多次提交分别列出。工具只读最近的记录，选中患者的完整档案读取该范围的全部历史。
RECORDS = """
SELECT r.record_id,r.task_id,r.bp_measurement_status,r.systolic_bp,r.diastolic_bp,r.heart_rate,
       r.symptoms,r.medication_adherence,
       CASE WHEN t.scheduled_at IS NOT NULL THEN datetime(r.submitted_at,'+8 hours')
            ELSE r.submitted_at END AS submitted_at,r.revision,t.patient_id,t.patient_name
FROM my_tasks t
JOIN followup_records r ON r.task_id=t.task_id
ORDER BY r.submitted_at DESC,r.revision DESC,r.record_id DESC
"""

# 主责关系允许查看患者档案，任务关系允许查看负责过的患者；分类只计算本人可见的任务。
# 带 task_id 时连主管医生都必须先有这项任务的权限，防止用别人的任务号查询患者。
PATIENTS = """
SELECT p.patient_id,p.name AS patient_name,p.age,p.gender,p.diagnosis,p.medical_history,
       primary_doc.doctor_id,primary_doc.name AS doctor_name,
       primary_doc.department AS doctor_department,primary_doc.title AS doctor_title,
       CASE WHEN primary_doc.user_id=u.user_id AND u.role='doctor' THEN 1 ELSE 0 END AS is_primary,
       COUNT(t.task_id) AS task_count,MAX(t.followup_date) AS latest_followup_date,
       MAX(CASE WHEN t.task_status IN ('submitted','in_review') THEN 1 ELSE 0 END) AS has_review,
       MAX(CASE WHEN t.task_status IN ('pending','in_progress','needs_info')
                 AND (t.scheduled_at IS NULL OR t.opened_at IS NOT NULL) THEN 1 ELSE 0 END) AS has_to_fill,
       MAX(CASE WHEN t.scheduled_at IS NOT NULL AND t.opened_at IS NULL THEN 1 ELSE 0 END) AS has_scheduled,
       COALESCE(MAX(t.is_overdue),0) AS has_overdue,
       MAX(CASE WHEN t.task_status='completed' THEN 1 ELSE 0 END) AS has_completed
FROM patients p
JOIN users u ON u.user_id=:user_id AND u.role=:role
LEFT JOIN doctors primary_doc ON primary_doc.doctor_id=p.doctor_id
LEFT JOIN my_tasks t ON t.patient_id=p.patient_id
WHERE (:patient_id IS NULL OR p.patient_id=:patient_id)
  AND (:task_id IS NULL OR t.task_id IS NOT NULL)
  AND ((u.role='patient' AND p.user_id=u.user_id)
    OR (u.role='doctor' AND (primary_doc.user_id=u.user_id OR t.task_id IS NOT NULL)))
GROUP BY p.patient_id
ORDER BY p.name,p.patient_id
"""


def query_rows(conn, sql, user_id, role, task_id, patient_id):
    """在给定连接中读取授权字典列表，无匹配返回 []；不支持的角色抛 PermissionError，不写库。

    sql 是本文件中的固定查询，两个编号都是可选范围；查询值通过 SQLite 参数绑定传入。
    """
    if role not in ("patient", "doctor"):
        raise PermissionError("当前账号不能查询医疗资料。")
    # 身份来自后端，选择编号来自页面或服务端工具上下文，不能覆盖身份。
    params = {"user_id": user_id, "role": role, "task_id": task_id, "patient_id": patient_id}
    # 截止时刻和排期服务使用同一北京时间，测试可把时钟固定在开放或逾期的边界。
    params["now"] = schedule_store.beijing_time_text()
    rows = conn.execute(MY_TASKS + sql, params).fetchall()
    items = []
    for row in rows:
        items.append(dict(row))  # 变成普通字典，可用于接口 JSON 或 Agent 工具结果。
    return items


def doctor_card(row, prefix):
    """从已授权查询行取负责或审核医生名片，缺少编号返回 None；仅整理内存数据。"""
    # 例如 doctor_name 表示当前负责人姓名；reviewer_name 表示审核人姓名。
    if row[prefix + "_id"] is None:
        return None
    return {"doctor_id": row[prefix + "_id"], "name": row[prefix + "_name"],
            "department": row[prefix + "_department"], "title": row[prefix + "_title"]}


def report_assignment(row, snapshot, reviewer_id):
    """输入当前任务、当次名片与审核人编号，返回（负责医生名片或 None，依据）；不读写数据库。"""
    if snapshot is not None:
        return snapshot, "snapshot"  # 保存的当次资料优先，即使后来医生改名或任务改派。
    assigned = doctor_card(row, "doctor")
    if reviewer_id is None:
        return assigned, "current_task"  # 只能说明任务当前分给谁，页面须说明来源。
    if reviewer_id == row["doctor_id"]:
        return assigned, "task_and_reviewer"  # 旧报告的实际签署人与任务记录一致。
    return None, "unverified"  # 当前任务与历史审核记录不同，不能猜测当时是谁负责。


def task_summaries(conn, user_id, role, task_id, patient_id):
    """输入连接、身份和可选范围；返回任务列表及已发布报告的负责医生，不返回报告正文。"""
    tasks = query_rows(conn, TASK_DETAILS, user_id, role, task_id, patient_id)
    for task in tasks:
        task["is_overdue"] = bool(task["is_overdue"])  # 接口返回布尔值，任务原有状态仍独立保留。
        saved = task.pop("doctor_snapshot")  # 临时 SQL 字段不直接作为接口内容。
        reviewer_id = task.pop("signed_by")
        task["report_doctor"] = None
        task["report_doctor_source"] = None
        if task["report_status"] == "published" and task["task_status"] == "completed":
            snapshot = None
            if saved is not None:
                snapshot = json.loads(saved)  # 固定查询仅选择了医生名片，未取建议或正文。
            doctor, source = report_assignment(task, snapshot, reviewer_id)
            task["report_doctor"] = doctor
            task["report_doctor_source"] = source
    return tasks


def list_followup_tasks(user_id: str, role: str, task_id=None, patient_id=None) -> list[dict]:
    """输入登录身份和可选任务、患者编号，返回授权任务列表；无匹配返回 []，只读数据库。

    每项含患者、负责医生、状态和相关日期，患者还能看到自己的草稿进度及退回说明。
    """
    with closing(get_connection()) as conn:
        return task_summaries(conn, user_id, role, task_id, patient_id)


def contact_phone(conn, person_role, person_id):
    """在当前只读事务里按已获准的人员编号取电话；空号返回 None，角色错误抛 ValueError。"""
    # 联系人编号取自同一事务刚查出的授权档案或任务，不能直接使用页面传来的编号。
    if person_role == "doctor":
        sql = "SELECT phone FROM doctors WHERE doctor_id=?"
    elif person_role == "patient":
        sql = "SELECT phone FROM patients WHERE patient_id=?"
    else:
        raise ValueError("未知的联系人角色。")
    row = conn.execute(sql, (person_id,)).fetchone()  # 只取需要展示的一位联系人。
    if row is None or not row["phone"]:
        return None
    return row["phone"].strip() or None  # 只录入了空格的号码也视为尚未登记。


def get_followup_task(user_id: str, role: str, task_id: str, *, include_contact=False) -> dict | None:
    """输入后端身份和任务编号，返回授权任务字典；无权或不存在返回 None，只读数据库。

    患者打开任务详情时 include_contact=True，结果另含本次负责医生的 doctor_phone。
    """
    with closing(get_connection()) as conn:
        conn.execute("BEGIN")  # 同一快照内确认任务归属和负责医生，避免改派期间混用两次查询。
        tasks = task_summaries(conn, user_id, role, task_id, None)
        if not tasks:
            return None  # 没有可访问的任务就没有联系人。
        task = tasks[0]  # 主键限定了至多一项。
        if include_contact and role == "patient":
            task["doctor_phone"] = contact_phone(conn, "doctor", task["doctor_id"])
        return task


def list_discharges(user_id: str, role: str, task_id=None, patient_id=None) -> list[dict]:
    """输入登录身份和可选范围，返回按日期倒序的授权出院信息；无匹配返回 []，只读数据库。"""
    with closing(get_connection()) as conn:
        return query_rows(conn, DISCHARGES, user_id, role, task_id, patient_id)


def list_recent_records(user_id: str, role: str, task_id=None, patient_id=None) -> list[dict]:
    """输入登录身份及可选范围，返回最近 20 条获准查看的正式记录；无匹配返回 []，只读数据库。"""
    with closing(get_connection()) as conn:
        return query_rows(conn, RECORDS + " LIMIT 20", user_id, role, task_id, patient_id)


def patient_profiles(conn, user_id, role, patient_id, task_id):
    """输入连接和授权范围，读取患者资料与独立的主管医生名片；无匹配返回 []，不写库。"""
    patients = query_rows(conn, PATIENTS, user_id, role, task_id, patient_id)
    for patient in patients:
        # 组装显示资料的同时移走临时列，未分配主管医生时返回 None。
        doctor = {"doctor_id": patient.pop("doctor_id"), "name": patient.pop("doctor_name"),
                  "department": patient.pop("doctor_department"), "title": patient.pop("doctor_title")}
        patient["primary_doctor"] = None
        if doctor["doctor_id"] is not None:
            patient["primary_doctor"] = doctor
        # SQLite 聚合标记为 0/1，接口以布尔值区分能进入哪些筛选。
        for field in ("is_primary", "has_review", "has_to_fill", "has_scheduled", "has_overdue", "has_completed"):
            patient[field] = bool(patient[field])
    return patients


def list_patients(user_id: str, role: str, patient_id=None, task_id=None) -> list[dict]:
    """输入当前身份及可选范围，返回患者名册；患者只见自己，无资料返回 []，只读数据库。"""
    with closing(get_connection()) as conn:
        return patient_profiles(conn, user_id, role, patient_id, task_id)


def get_patient_detail(user_id: str, role: str, patient_id: str) -> dict | None:
    """输入后端身份、患者编号，返回 patient/discharges/tasks/records；无权或不存在返回 None。

    患者看到主管医生的 primary_doctor.phone，医生看到患者的 patient.phone，未登记返回 None。
    在同一个只读事务中组装档案，避免医生改派时读到不同权限时刻的数据。
    """
    with closing(get_connection()) as conn:
        conn.execute("BEGIN")  # 直到档案组装完成，资料和联系人都来自同一份数据库快照。
        profiles = patient_profiles(conn, user_id, role, patient_id, None)
        if not profiles:
            return None
        patient = profiles[0]  # 查询条件是患者编号，至多有一份授权档案。
        if role == "doctor":
            patient["phone"] = contact_phone(conn, "patient", patient["patient_id"])
        elif patient["primary_doctor"] is not None:
            doctor = patient["primary_doctor"]  # 尚无任务的患者也能联系自己的主管医生。
            doctor["phone"] = contact_phone(conn, "doctor", doctor["doctor_id"])
        # 出院、随访和正式记录各自只读获授权范围，主管身份本身不会开放其他医生的任务。
        return {"patient": patient,
                "discharges": query_rows(conn, DISCHARGES, user_id, role, None, patient_id),
                "tasks": task_summaries(conn, user_id, role, None, patient_id),
                "records": query_rows(conn, RECORDS, user_id, role, None, patient_id)}
