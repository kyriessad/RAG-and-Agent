"""读取当前账号有权查看的医疗记录；这里不提交或修改随访数据。"""

from utils.user_store import get_connection


def list_followup_tasks(user_id: str, role: str, task_id: str | None = None) -> list[dict]:
    """输入登录身份和可选任务编号，返回获授权的任务字典列表；只读数据库。

    例如 ("U001", "patient", "F001") 只查询 F001；不传编号时查所有获授权任务。
    找不到或没有任务权限时返回 []；不支持的角色抛出 PermissionError。
    """

    # user_id、role 来自后端登录态，例如："U001"、"patient"。
    # 不接受聊天里自报的身份，也不让管理员默认取得全部医疗数据。
    if role not in ("patient", "doctor"):
        raise PermissionError("当前账号不能查询随访任务。")

    conn = get_connection()
    try:
        # t 是任务，d 是出院记录，p 是患者，doctor 是负责该任务的医生。
        # 同一条 SQL 中校验归属与任务编号，括号保证编号过滤同时约束两种角色。
        rows = conn.execute(
            """
            SELECT t.task_id, t.discharge_id, t.followup_date,
                   t.task_type, t.task_status, t.expires_at,
                   p.patient_id, p.name AS patient_name,
                   doctor.name AS doctor_name
            FROM followup_tasks AS t
            JOIN discharges AS d ON d.discharge_id = t.discharge_id
            JOIN patients AS p ON p.patient_id = d.patient_id
            JOIN doctors AS doctor ON doctor.doctor_id = t.doctor_id
            WHERE ((? = 'patient' AND p.user_id = ?)
               OR (? = 'doctor' AND doctor.user_id = ?))
              AND (? IS NULL OR t.task_id = ?)
            ORDER BY t.followup_date, t.task_id
            """,
            (role, user_id, role, user_id, task_id, task_id),
        ).fetchall()

        # rows 是 sqlite3.Row 列表；转成字典后，接口和 Agent 都能直接使用。
        # 没绑定患者资料或没有任务时，返回 []，不编造随访计划。
        tasks = []  # 例如 [{"task_id": "F001", "task_status": "pending", ...}]。
        for row in rows:
            tasks.append(dict(row))  # 每条数据库记录转换为一个普通字典。
        return tasks
    finally:
        conn.close()


def get_followup_task(user_id: str, role: str, task_id: str) -> dict | None:
    """输入登录身份和任务编号，返回一个任务字典或 None；只读数据库。

    例如 ("U001", "patient", "F001") 返回 F001 的详情；不存在和越权都返回 None。
    不支持的角色仍由查询函数抛出 PermissionError。
    """
    # 复用同一套 SQL 授权条件，列表与详情不会采用不同的归属标准。
    tasks = list_followup_tasks(user_id, role, task_id)
    if not tasks:  # 空列表表示没有找到可访问的任务。
        return None
    return tasks[0]  # task_id 是主键，指定编号时最多返回一条记录。


def list_discharges(user_id: str, role: str, task_id: str | None = None) -> list[dict]:
    """输入登录身份和可选任务编号，返回获授权的出院信息列表；只读数据库。

    例如指定 F001 时只返回该任务对应的出院信息；无权访问或不存在返回 []。
    不传编号时保留原有查询范围；不支持的角色抛出 PermissionError。
    """

    if role not in ("patient", "doctor"):
        raise PermissionError("当前账号不能查询出院信息。")

    conn = get_connection()
    try:
        # 医生权限依据任务分配，而不是仅凭患者姓名或患者编号。
        # EXISTS 表示“存在符合条件的任务”；同一次出院的多项任务不会导致重复返回。
        # 医生的 EXISTS 也检查 task_id，避免拿另一项任务的授权查看本次任务。
        # 最后的 EXISTS 把患者查询也限制到选中任务关联的那一次出院。
        rows = conn.execute(
            """
            SELECT d.discharge_id, d.discharge_date, d.discharge_diagnosis,
                   d.discharge_instructions, d.discharge_medications,
                   d.baseline_bp_systolic, d.baseline_bp_diastolic,
                   d.baseline_heart_rate, p.patient_id, p.name AS patient_name
            FROM discharges AS d
            JOIN patients AS p ON p.patient_id = d.patient_id
            WHERE ((? = 'patient' AND p.user_id = ?)
               OR (? = 'doctor' AND EXISTS (
                   SELECT 1
                   FROM followup_tasks AS t
                   JOIN doctors AS doctor ON doctor.doctor_id = t.doctor_id
                   WHERE t.discharge_id = d.discharge_id
                     AND doctor.user_id = ?
                     AND (? IS NULL OR t.task_id = ?)
               )))
              AND (? IS NULL OR EXISTS (
                  SELECT 1
                  FROM followup_tasks AS selected_task
                  WHERE selected_task.discharge_id = d.discharge_id
                    AND selected_task.task_id = ?
              ))
            ORDER BY d.discharge_date DESC, d.discharge_id
            """,
            (role, user_id, role, user_id, task_id, task_id, task_id, task_id),
        ).fetchall()

        # 只选择业务所需字段，不把账号、令牌或其他患者资料交给模型。
        discharges = []  # 例如 [{"discharge_id": "DIS001", "discharge_date": "2026-11-01", ...}]。
        for row in rows:
            discharges.append(dict(row))  # 保留 SQL 明确选择的出院字段。
        return discharges
    finally:
        conn.close()


def list_recent_records(user_id: str, role: str, task_id: str | None = None) -> list[dict]:
    """输入登录身份和可选任务编号，返回最近 20 条获授权记录；只读数据库。

    例如指定 F001 时只返回 F001 的记录，保留 record_id 和 revision 区分各次提交。
    不存在或无权访问时返回 []；不支持的角色抛出 PermissionError。
    """

    if role not in ("patient", "doctor"):
        raise PermissionError("当前账号不能查询随访记录。")

    conn = get_connection()
    try:
        # 同一任务可以有多次提交，所以保留 record_id 和 revision 区分记录。
        # 明确列出返回字段；待审核报告、审核意见不会混在血压记录中返回。
        rows = conn.execute(
            """
            SELECT r.record_id, r.task_id, r.bp_measurement_status,
                   r.systolic_bp, r.diastolic_bp, r.heart_rate,
                   r.symptoms, r.medication_adherence, r.submitted_at,
                   r.revision, p.patient_id, p.name AS patient_name
            FROM followup_records AS r
            JOIN followup_tasks AS t ON t.task_id = r.task_id
            JOIN discharges AS d ON d.discharge_id = t.discharge_id
            JOIN patients AS p ON p.patient_id = d.patient_id
            JOIN doctors AS doctor ON doctor.doctor_id = t.doctor_id
            WHERE ((? = 'patient' AND p.user_id = ?)
               OR (? = 'doctor' AND doctor.user_id = ?))
              AND (? IS NULL OR t.task_id = ?)
            ORDER BY r.submitted_at DESC, r.revision DESC, r.record_id DESC
            LIMIT 20
            """,
            (role, user_id, role, user_id, task_id, task_id),
        ).fetchall()

        # submitted_at 是入库时间，并不代表实际测量时间。
        records = []  # 例如 [{"record_id": "R001", "task_id": "F001", "revision": 1, ...}]。
        for row in rows:
            records.append(dict(row))  # 每条记录仍保留明确的编号、版本和排序。
        return records
    finally:
        conn.close()
