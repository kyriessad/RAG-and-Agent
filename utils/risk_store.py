"""保存规则升级通知；不调用模型，不生成报告，不改变随访任务状态。"""

import hashlib
import json
import sqlite3

from utils.user_store import get_connection


def save_risk_notification(conn, user_id, task_id, risk, source):
    """在调用方已授权的事务中保存高风险通知，返回 bool；此处不 commit。

    输入例：risk={level: high, rule_codes: [chest_pain], reasons: [...], ...}。
    True 表示通知已存在或本次成功写入；非高风险返回 False。
    同任务、同负责医生、同规则组合只通知一次；换医生后可以重新通知。
    数据库错误交给调用方回滚，不能把失败说成已经通知医生。
    """
    if risk["level"] != "high":  # 不把缺信息或未命中规则当成紧急告警。
        return False
    doctor = conn.execute(
        """SELECT u.user_id FROM followup_tasks AS t
           JOIN doctors AS d ON d.doctor_id=t.doctor_id
           JOIN users AS u ON u.user_id=d.user_id AND u.role='doctor'
           WHERE t.task_id=?""", (task_id,),
    ).fetchone()  # 用任务上的当前负责医生，不能使用模型或患者传入的接收人。
    if doctor is None:
        raise sqlite3.DatabaseError("风险通知无法找到负责医生账号。")
    identity = json.dumps([task_id, doctor["user_id"], sorted(risk["rule_codes"])], ensure_ascii=False)
    event_key = "risk:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()  # 数据库唯一键负责去重。
    message = "随访风险提醒：" + "；".join(risk["reasons"])  # 保存命中依据，方便医生核对。
    message += "。信息来源：" + source + "。请及时核对患者情况；规则提醒不代表诊断。"
    cursor = conn.execute(
        """INSERT INTO notifications(user_id,task_id,message,event_key,created_at)
           VALUES(?,?,?,?,CURRENT_TIMESTAMP) ON CONFLICT(event_key) DO NOTHING""",
        (doctor["user_id"], task_id, message, event_key),
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
    """患者原话先经纯规则检查，再调用此函数独立提交通知；返回是否已保存。

    权限失败抛 PermissionError，库错误向上抛出；不读草稿、不依赖模型成功。
    例如患者说“现在胸痛”，即使草稿缺项或随后模型出错，通知仍已持久保存。
    """
    from utils.draft_store import require_patient_task  # 在函数内引用，避免草稿模块与本模块循环导入。

    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")  # 独立事务先保存告警，再进入可能失败的模型调用。
        require_patient_task(conn, user_id, role, task_id)  # 写入前再次核对真实角色和任务归属。
        saved = save_risk_notification(conn, user_id, task_id, risk, "患者对话，尚未经确认提交")
        conn.commit()
        return saved
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def list_doctor_notifications(user_id, role):
    """输入服务端医生身份，返回当前名下任务的风险通知列表；只读数据库。

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
        rows = conn.execute(
            """SELECT n.id,n.task_id,p.name AS patient_name,n.message,n.created_at,n.read_at
               FROM notifications AS n
               JOIN followup_tasks AS t ON t.task_id=n.task_id
               JOIN doctors AS doctor ON doctor.doctor_id=t.doctor_id
               JOIN discharges AS d ON d.discharge_id=t.discharge_id
               JOIN patients AS p ON p.patient_id=d.patient_id
               WHERE n.user_id=? AND doctor.user_id=? AND n.event_key LIKE 'risk:%'
               ORDER BY n.id DESC""", (user_id, user_id),
        ).fetchall()  # SQL 内过滤权限，不先读取所有患者再让页面过滤。
        notifications = []
        for row in rows:
            notifications.append(dict(row))
        return notifications
    finally:
        conn.close()
