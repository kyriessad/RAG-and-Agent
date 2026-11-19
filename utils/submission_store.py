"""患者确认提交：保存一份不可覆盖的随访记录，并原子更新任务状态。"""

import json

from pydantic import BaseModel, ConfigDict, Field

from utils.draft_store import confirmed_record_id, missing_draft_fields, read_draft, require_patient_task, validate_draft
from utils.risk_rules import evaluate_risk
from utils.risk_store import save_risk_notification
from utils.user_store import get_connection
from utils.chat_store import save_receipt


class SubmitDraftData(BaseModel):
    """确认请求只接收旧版本与明确确认，例如 {expected_revision: 2, confirmed: true}。"""

    model_config = ConfigDict(extra="forbid", strict=True)  # 不能从请求指定身份、风险或正式记录。
    expected_revision: int = Field(ge=0)  # 必须是页面刚核对过的已保存草稿版本。
    confirmed: bool  # 不默认设为 True；必须由患者明确确认。


def submission_result(task_id, record_id, revision, draft_revision, status, risk, repeated):
    """输入已保存的记录标识和规则结果，输出页面需要的字典；不读写数据库。

    revision 是正式记录版本，draft_revision 是患者确认的草稿版本，两者不能混用。
    repeated=True 表示本次只是返回之前的成功结果，没有创建第二条记录。
    """
    return {"task_id": task_id, "record_id": record_id, "revision": revision,
            "draft_revision": draft_revision, "task_status": status,
            "already_submitted": repeated, "risk": risk}


def submit_task_draft(user_id, role, task_id, expected_revision, confirmed):
    """确认并提交已保存的完整草稿；成功返回记录编号、版本、状态与规则结果。

    输入例：('U001','patient','F001',2,True)。不接受模型提议或页面未保存内容。
    越权抛 PermissionError；缺项/未确认抛 ValueError；过期版本或状态错误抛 RuntimeError。
    写 followup_records、followup_tasks、audit_logs，必要时写 notifications；不调用模型。
    所有写入处于同一事务，任何一步出错都回滚；同任务同草稿版本重试返回同一记录。
    """
    if confirmed is not True:  # API 严格验证类型，内部调用仍不能用真假相似的值绕过确认。
        raise ValueError("请先核对已保存的信息，并勾选确认后提交。")
    if type(expected_revision) is not int or expected_revision < 0:
        raise ValueError("草稿版本必须是非负整数。")
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")  # SQLite 写锁串行处理同时提交，防止双方插入成功。
        status = require_patient_task(conn, user_id, role, task_id)  # 事务内重新核对身份与任务归属。
        draft = read_draft(conn, task_id, status)
        if draft["revision"] != expected_revision:
            raise RuntimeError("草稿已更新，请重新加载并核对后再确认提交。")
        # 同一草稿版本得到固定记录编号；网络重试无需新建表或接收前端生成的编号。
        record_id = confirmed_record_id(task_id, expected_revision)
        existing = conn.execute(
            "SELECT revision,data_json FROM followup_records WHERE record_id=? AND task_id=?",
            (record_id, task_id),
        ).fetchone()
        if existing is not None:
            if status not in ("submitted", "in_review", "completed"):
                raise RuntimeError("这份草稿已经提交过，请先补充并保存新版本，再重新确认提交。")
            saved = json.loads(existing["data_json"])  # 使用当时保存的规则快照，不重新改写历史记录。
            return submission_result(task_id, record_id, existing["revision"], expected_revision,
                                     status, saved["_risk"], True)
        if status not in ("pending", "in_progress", "needs_info"):
            raise RuntimeError("当前任务已提交或结束，不能再次提交新的随访记录。")
        data = validate_draft(draft["data"])  # 提交时再次校验，不直接相信页面或 needs_clarification 标记。
        missing = missing_draft_fields(data)
        if missing:
            raise ValueError("请先补充并保存：" + "、".join(missing) + "。")
        risk = evaluate_risk(data=data)  # 独立规则，不等待报告或模型决定风险。
        row = conn.execute(
            "SELECT COALESCE(MAX(revision),0) AS revision FROM followup_records WHERE task_id=?", (task_id,),
        ).fetchone()
        revision = row["revision"] + 1  # 任务可以有多次正式提交，始终保留旧记录并增加记录版本。
        snapshot = dict(data)  # 当前已确认字段快照；旧心率等未经此次确认的字段不擅自带入。
        snapshot["_submission"] = {"draft_revision": expected_revision, "confirmed_by": user_id}
        snapshot["_risk"] = risk  # 保存规则版本及命中依据，以后规则升级不会改变旧结论。
        payload = json.dumps(snapshot, ensure_ascii=False)
        conn.execute(
            """INSERT INTO followup_records(
               record_id,task_id,systolic_bp,diastolic_bp,bp_measurement_status,symptoms,
               medication_adherence,risk_level,risk_reason,data_json,revision)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (record_id, task_id, data["systolic_bp"], data["diastolic_bp"], data["bp_measurement_status"],
             data["symptoms"], data["medication_adherence"], risk["level"],
             "；".join(risk["reasons"]), payload, revision),
        )  # 测量时间、漏服原因、不良反应保留在 data_json 中，不改已有表结构。
        conn.execute("UPDATE followup_tasks SET task_status='submitted' WHERE task_id=?", (task_id,))
        details = json.dumps({"record_id": record_id, "revision": revision, "draft_revision": expected_revision,
                              "from_status": status, "to_status": "submitted"}, ensure_ascii=False)
        conn.execute(
            """INSERT INTO audit_logs(actor_id,task_id,action,details_json,created_at)
               VALUES(?,?,'followup_submitted',?,CURRENT_TIMESTAMP)""", (user_id, task_id, details),
        )  # 审计、状态、正式记录必须一起成功，不能只留下其中一部分。
        save_risk_notification(conn, user_id, task_id, risk, "患者确认提交")  # 高风险时同事务保存医生通知。
        save_receipt(conn, user_id, role, "submit", record_id, "随访信息已提交，等待医生审核。",
                     task_id=task_id, record_id=record_id, draft_revision=expected_revision)
        conn.commit()
        return submission_result(task_id, record_id, revision, expected_revision, "submitted", risk, False)
    except Exception:
        conn.rollback()  # 例如通知或审计写入失败，也不能留下已提交状态。
        raise
    finally:
        conn.close()  # 重试分支只有读取，关闭连接时释放事务和锁。
