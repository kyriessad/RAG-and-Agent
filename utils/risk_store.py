"""保存规则升级通知；不调用模型，不生成报告，不改变随访任务状态。"""

import hashlib
import json
import sqlite3
from contextlib import closing
from uuid import uuid4

from utils.user_store import get_connection
from utils.chat_store import save_receipt


def migrate_risk_notifications():
    """无任务的首页告警直接关联患者档案，既有任务告警仍按原任务查找。"""
    with closing(get_connection()) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        if not conn.execute("SELECT 1 FROM pragma_table_info('notifications') WHERE name='patient_id'").fetchone():
            conn.execute("ALTER TABLE notifications ADD COLUMN patient_id TEXT REFERENCES patients(patient_id)")
        if not conn.execute("SELECT 1 FROM pragma_table_info('notifications') WHERE name='handled_at'").fetchone():
            conn.execute("ALTER TABLE notifications ADD COLUMN handled_at TEXT")


def patient_doctor(conn, user_id, task_id):
    if task_id is None:
        query = """SELECT du.user_id,p.patient_id,d.name,d.phone FROM patients p
                   JOIN doctors d ON d.doctor_id=p.doctor_id
                   JOIN users du ON du.user_id=d.user_id AND du.role='doctor'
                   JOIN users u ON u.user_id=p.user_id AND u.role='patient'
                   WHERE u.user_id=?"""
        params = (user_id,)
    else:
        query = """SELECT du.user_id,p.patient_id,d.name,d.phone FROM followup_tasks t
                   JOIN doctors d ON d.doctor_id=t.doctor_id
                   JOIN users du ON du.user_id=d.user_id AND du.role='doctor'
                   JOIN discharges h ON h.discharge_id=t.discharge_id
                   JOIN patients p ON p.patient_id=h.patient_id
                   JOIN users u ON u.user_id=p.user_id AND u.role='patient'
                   WHERE u.user_id=? AND t.task_id=?
                     AND (t.scheduled_at IS NULL OR t.opened_at IS NOT NULL)"""
        params = (user_id, task_id)
    return conn.execute(query, params).fetchone()


def contact_text(row, task_id):
    if row is None:
        return "未查到对应医生的联系方式。"
    label = "本次随访医生" if task_id is not None else "主管医生"
    phone = (row["phone"] or "").strip()
    return f"{label} {row['name']} 的联系电话为 {phone}。" if phone else f"{label} {row['name']} 尚未登记联系电话。"


def patient_doctor_contact(user_id, task_id=None):
    """查询本患者、当前任务内的负责医生；无任务时读取主管医生。"""
    with closing(get_connection()) as conn:
        return contact_text(patient_doctor(conn, user_id, task_id), task_id)


def save_risk_notification(conn, user_id, task_id, risk, source):
    """在调用方已授权的事务中保存高风险通知，返回 bool；此处不 commit。

    输入例：risk={level: high, rule_codes: [chest_pain], reasons: [...], ...}。
    True 表示通知已存在或本次成功写入；非高风险返回 False。
    同任务、同负责医生、同规则组合只通知一次；换医生后可以重新通知。
    数据库错误交给调用方回滚，不能把失败说成已经通知医生。
    """
    if risk["level"] != "high":  # 不把缺信息或未命中规则当成紧急告警。
        return False
    doctor = patient_doctor(conn, user_id, task_id)
    if doctor is None:
        if task_id is None:
            return False
        raise sqlite3.DatabaseError("风险通知无法找到负责医生账号。")
    patient_id = doctor["patient_id"] if task_id is None else None
    identity = json.dumps([task_id or patient_id, doctor["user_id"], sorted(risk["rule_codes"])], ensure_ascii=False)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    event_key = "risk:" + digest
    if task_id is None:
        # 首页没有任务生命周期。五分钟内同一患者、医生和规则重发只入库一次，之后可再次升级。
        prefix = "risk:home:" + digest + ":"
        repeated = conn.execute("""SELECT 1 FROM notifications WHERE event_key LIKE ?
                                   AND created_at >= datetime('now','-5 minutes')""", (prefix + "%",)).fetchone()
        if repeated:
            return True
        event_key = prefix + uuid4().hex
    message = "随访风险提醒：" + "；".join(risk["reasons"])  # 保存命中依据，方便医生核对。
    message += "。信息来源：" + source + "。请及时核对患者情况；规则提醒不代表诊断。"
    cursor = conn.execute(
        """INSERT INTO notifications(user_id,task_id,patient_id,message,event_key,created_at)
           VALUES(?,?,?,?,?,CURRENT_TIMESTAMP) ON CONFLICT(event_key) DO NOTHING""",
        (doctor["user_id"], task_id, patient_id, message, event_key),
    )  # 重复检查返回已有通知，不重复插入，也不把其他约束错误吞掉。
    if cursor.rowcount == 1:
        details = json.dumps({"source": source, "risk": risk, "recipient_id": doctor["user_id"]}, ensure_ascii=False)
        conn.execute(
            """INSERT INTO audit_logs(actor_id,task_id,action,details_json,created_at)
               VALUES(?,?,'risk_escalated',?,CURRENT_TIMESTAMP)""",
            (user_id, task_id, details),
        )  # 通知与审计在同一事务中，要么一起保存，要么一起回滚。
    return True


def list_task_risk_alerts(conn, task_id):
    """在已经核对患者任务权限的连接上读取历史规则提醒，返回字典列表；不写库。

    仅取本功能产生的 risk: 通知，不把其他通知或医生审核意见暴露给患者。
    保留已发生的提醒，即使患者后来清空草稿，也不能把曾发生的升级抹掉。
    """
    rows = conn.execute(
        """SELECT id,task_id,message,created_at FROM notifications
           WHERE task_id=? AND event_key LIKE 'risk:%' ORDER BY id DESC""", (task_id,),
    ).fetchall()
    alerts = []  # 无历史提醒时返回 []。
    for row in rows:
        alerts.append(dict(row))
    return alerts


def save_message_risk(user_id, role, task_id, risk):
    """患者原话先经纯规则检查，独立提交通知并返回同一快照内的医生联系方式。

    权限失败抛 PermissionError，库错误向上抛出；不读草稿、不依赖模型成功。
    例如患者说“现在胸痛”，即使草稿缺项或随后模型出错，通知仍已持久保存。
    """
    from utils.draft_store import require_patient_task  # 在函数内引用，避免草稿模块与本模块循环导入。

    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")  # 独立事务先保存告警，再进入可能失败的模型调用。
        if role != "patient":
            raise PermissionError("只有患者可以报告本人的不适。")
        if task_id is not None:
            require_patient_task(conn, user_id, role, task_id)
        saved = save_risk_notification(conn, user_id, task_id, risk, "患者对话，尚未经确认提交")
        contact = contact_text(patient_doctor(conn, user_id, task_id), task_id)
        conn.commit()
        return saved, contact
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


DOCTOR_ALERTS = """
    SELECT n.id,n.task_id,p.patient_id,p.name AS patient_name,n.message,n.created_at,n.read_at,n.handled_at
    FROM notifications AS n
    LEFT JOIN followup_tasks AS t ON t.task_id=n.task_id
    LEFT JOIN discharges AS d ON d.discharge_id=t.discharge_id
    JOIN patients AS p ON p.patient_id=CASE WHEN n.task_id IS NULL THEN n.patient_id ELSE d.patient_id END
    JOIN doctors AS doctor ON doctor.doctor_id=CASE WHEN n.task_id IS NULL THEN p.doctor_id ELSE t.doctor_id END
    JOIN users AS u ON u.user_id=doctor.user_id AND u.role='doctor'
    WHERE n.user_id=:doctor AND u.user_id=:doctor AND n.event_key LIKE 'risk:%'
"""


def handle_doctor_notification(user_id, role, notice_id):
    """在复核归属的事务内登记首次处理时间。多次确认返回原回执。"""
    if role != "doctor":
        raise PermissionError("只有负责医生可以处理风险通知。")
    with closing(get_connection()) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        scoped_notice = DOCTOR_ALERTS + " AND n.id=:id"
        params = {"doctor": user_id, "id": notice_id}
        row = conn.execute(scoped_notice, params).fetchone()
        if row is None:
            return None
        if row["handled_at"] is None:
            conn.execute("UPDATE notifications SET handled_at=CURRENT_TIMESTAMP WHERE id=?", (notice_id,))
            row = conn.execute(scoped_notice, params).fetchone()
        save_receipt(conn, user_id, role, "handle_risk", notice_id, "风险提醒已标记为已处理。",
                     patient_id=row["patient_id"], task_id=row["task_id"], notice_id=notice_id, handled_at=row["handled_at"])
        return dict(row)


def list_doctor_notifications(user_id, role):
    """返回本人作为任务负责人或主管医生收到的风险通知。

    患者或已改成其他角色的账号抛 PermissionError；无通知返回 []。
    同时核对通知接收人和任务当前负责医生，任务改派后旧医生不能继续读取。
    """
    if role != "doctor":
        raise PermissionError("只有负责医生可以查看风险通知。")
    conn = get_connection()
    try:
        conn.execute("BEGIN")  # 角色核对与通知查询使用同一快照。
        account = conn.execute("SELECT role FROM users WHERE user_id=?", (user_id,)).fetchone()
        if account is None or account["role"] != "doctor":
            raise PermissionError("只有负责医生可以查看风险通知。")
        rows = conn.execute(DOCTOR_ALERTS + " ORDER BY (n.handled_at IS NOT NULL), n.id DESC",
                            {"doctor": user_id}).fetchall()
        notifications = []
        for row in rows:
            notifications.append(dict(row))
        return notifications
    finally:
        conn.close()
