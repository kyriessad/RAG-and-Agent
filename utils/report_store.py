"""报告业务入口。报告绑定正式记录，医生审核和发布都在 SQLite 事务内完成。"""

import json
# closing 管连接的释放；后面的 with conn 单独管理提交与回滚。
from contextlib import closing
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.tools.report_generation import build_report
from utils import schedule_store
from utils.followup_store import doctor_card, report_assignment
from utils.logger_handler import logger
from utils.user_store import get_connection
from utils.chat_store import save_receipt


class ReportError(Exception):
    """status 是接口错误码，message 是可以直接展示给当前账号的提示。"""

    def __init__(self, status, message):
        super().__init__(message)  # 例如 str(error) 为“任务已更新，请刷新。”
        self.status = status


class GenerateRequest(BaseModel):
    """请求只能选择记录；不能提交风险、来源、草稿或医生身份。"""

    # 生成接口只接收 record_id，患者不能将草稿正文塞进 POST 请求。
    model_config = ConfigDict(extra="forbid", strict=True)
    record_id: str = Field(min_length=1, max_length=128)


class ReviewRequest(BaseModel):
    """医生点击时提交所见报告版本；可以改摘要、关注项、保留的引用建议，并填写意见。"""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    # 模型不能调用此接口，四个可选动作由医生按钮触发。
    action: Literal["approve", "edit", "return", "publish"]
    # 报告编号识别是哪一次提交的报告，不能只凭任务编号审核。
    report_id: str = Field(min_length=1, max_length=128)
    # 例如页面打开的是版本 1，其他页面已经改成 2 时，应由事务拒绝旧操作。
    expected_version: int = Field(ge=1)
    # 退回时这段文字会交给患者；通过时可以留空。
    note: str = Field(default="", max_length=2000)
    summary: str = Field(default="", max_length=8000)
    # None 表示沿用原有关注项，空列表表示医生明确删除了全部关注项。
    abnormal_points: list[str] | None = Field(default=None, max_length=24)
    keep_sources: list[int] | None = Field(default=None, max_length=3)  # 例如 [2] 表示只保留原稿第二条引用建议。


def current_case(conn, user, task_id):
    """查询授权任务、最新正式记录和对应报告。越权与不存在均为 404；只读取数据库。

    一条 SQL 同时取得任务状态、负责人及记录，避免查看时拼接不同时间点的数据。
    即使同一个任务曾被退回，也只用最新 record_id 对应的报告。
    """
    # LEFT JOIN 让尚未生成报告的任务也能返回；ORDER BY 优先选择正式记录版本最大的那次。
    row = conn.execute("""
        SELECT t.task_status, t.doctor_id, d.name AS doctor_name, d.department AS doctor_department,
               d.title AS doctor_title, r.record_id, r.revision,
               -- 新排期任务的显示时间与 48 小时填写期限使用同一时区，保留旧记录的登记格式。
               CASE WHEN t.scheduled_at IS NOT NULL THEN datetime(r.submitted_at,'+8 hours')
                    ELSE r.submitted_at END AS submitted_at,
               r.systolic_bp, r.diastolic_bp, r.bp_measurement_status, r.heart_rate, r.symptoms,
               r.medication_adherence, r.risk_level, r.risk_reason, r.data_json,
               b.report_id, b.status, b.version, b.content_json, b.review_note,
               CASE WHEN t.scheduled_at IS NOT NULL THEN datetime(b.reviewed_at,'+8 hours')
                    ELSE b.reviewed_at END AS reviewed_at,
               reviewer.doctor_id AS reviewer_id, reviewer.name AS reviewer_name,
               reviewer.department AS reviewer_department, reviewer.title AS reviewer_title
        FROM followup_tasks t
        JOIN doctors d ON d.doctor_id=t.doctor_id
        JOIN discharges h ON h.discharge_id=t.discharge_id
        JOIN patients p ON p.patient_id=h.patient_id
        JOIN users u ON u.user_id=? AND u.role=?
        LEFT JOIN followup_records r ON r.record_id=(
            SELECT record_id FROM followup_records WHERE task_id=t.task_id
            ORDER BY revision DESC, submitted_at DESC, record_id DESC LIMIT 1)
        LEFT JOIN reports b ON b.record_id=r.record_id AND b.task_id=t.task_id
        LEFT JOIN doctors reviewer ON reviewer.doctor_id=b.reviewer_id
        WHERE t.task_id=? AND (? IS NULL OR p.patient_id=?)
          AND ((u.role='patient' AND p.user_id=u.user_id
                 AND (t.scheduled_at IS NULL OR t.opened_at IS NOT NULL))
                           OR (u.role='doctor' AND d.user_id=u.user_id))
        """, (user["user_id"], user["role"], task_id, user.get("patient_id"), user.get("patient_id"))).fetchone()
    # 即使请求指定正确的 task_id，角色或任务归属不匹配也读不到任何行。
    if row is None:
        raise ReportError(404, "任务不存在或无权访问。")
    return row


def original_record(row):
    """从授权查询结果提取医生核对所需的记录和原始 JSON；无记录返回 None。"""
    # 未确认提交的任务没有正式记录，不从可编辑草稿中拼出一份替代品。
    if row["record_id"] is None:
        return None
    record = {}
    # 这里显式挑选医生需要的列，避免把无关的登录和授权数据带入报告。
    for name in ("record_id", "revision", "submitted_at", "systolic_bp", "diastolic_bp",
                 "bp_measurement_status", "heart_rate", "symptoms", "medication_adherence",
                 "risk_level", "risk_reason"):
        record[name] = row[name]  # 保留 None，不能把缺失值当成“没有不舒服”。
    record["data"] = json.loads(row["data_json"])  # 包括测量时间等患者确认的原话。
    return record


def report_view(user, task_id):
    """输入身份和任务编号，只读报告进度、当次医生与可见正文；越权抛 404。

    新报告的医生名片从快照读取，旧报告的审核人由既有 reviewer_id 核实。
    患者在发布前只能看到任务进度、负责医生及需要补充的意见。
    """
    with closing(get_connection()) as conn:
        row = current_case(conn, user, task_id)
    # 先构造可以给患者的最小响应，按角色逐个添加经过授权的字段。
    assigned = doctor_card(row, "doctor")  # 尚无正式报告时，展示实时任务分配。
    view = {"task_status": row["task_status"], "record_id": row["record_id"],
            "responsible_doctor": assigned, "responsible_doctor_source": "current_task",
            "reviewer": None, "reviewed_at": None}
    if user["role"] == "doctor":
        view["record"] = original_record(row)  # 只有负责医生能在审核页面对照原始数据。
        view["report_id"] = row["report_id"]
        # 医生审核按钮带回这个版本，重复请求和过时请求由数据库区分。
        view["version"] = row["version"]
        view["status"] = row["status"]
    # 报告和任务的终态都必须满足才向患者开放正文。
    published = row["status"] == "published" and row["task_status"] == "completed"
    if row["report_id"] and (user["role"] == "doctor" or published):
        content = json.loads(row["content_json"])  # 保留发布时的引用和原文，名片单列给界面。
        snapshot = content.pop("_responsible_doctor", None)
        reviewer = content.pop("_reviewer", None)
        if reviewer is None:
            reviewer = doctor_card(row, "reviewer")  # 老报告有审核编号时可核实医生身份。
        # 与历史报告列表共用同一判定：无快照时说明来源，记录有冲突时保留独立审核人。
        responsible, source = report_assignment(row, snapshot, row["reviewer_id"])
        view["responsible_doctor"] = responsible
        view["responsible_doctor_source"] = source
        view["reviewer"] = reviewer
        view["reviewed_at"] = row["reviewed_at"]
        view["content"] = content
        view["note"] = row["review_note"]
    elif row["status"] == "returned" and row["task_status"] == "needs_info":
        view["note"] = row["review_note"]  # 患者需要这段文字来补充资料，不返回报告正文。
    return view


def audit(conn, user, task_id, report_id, action, details):
    """向调用方的事务追加审计，不自行提交。写入异常会使整次状态变更回滚。"""
    conn.execute("""INSERT INTO audit_logs(actor_id,task_id,report_id,action,details_json,created_at)
                    VALUES(?,?,?,?,?,CURRENT_TIMESTAMP)""",
                 (user["user_id"], task_id, report_id, action, json.dumps(details, ensure_ascii=False)))


def generation_needed(row, record_id):
    """核对点击时的正式记录与实时状态。已生成返回 False；过时请求抛 409。"""
    # 客户端可能还开着上一次被退回的记录，此时不能生成过时报告。
    if row["record_id"] != record_id:
        raise ReportError(409, "正式记录已变化，请重新打开任务。")
    if row["report_id"]:
        return False  # 同一记录重试只能拿到回执，不能重开已经退回或发布的报告。
    if row["task_status"] != "submitted":
        raise ReportError(409, "请先确认提交本次随访记录。")
    return True


def generate_report(user, task_id, record_id):
    """为当前记录生成报告，仅返回编号回执。模型异常返回 503，记录保留且可以重试。

    输入例如 F001、FR_…；模型执行期间释放数据库连接。落库前在写事务中重新核对。
    同一记录并发请求可能各调用一次模型，但最多产生一份报告和一条生成审计。
    """
    receipt = {"record_id": record_id}  # 患者也可以触发重试，响应绝不能附带待审核正文。
    with closing(get_connection()) as conn:
        row = current_case(conn, user, task_id)
        # 已经生成过时直接返回回执，刷新或重试不会改写原稿。
        if not generation_needed(row, record_id):
            return receipt
        record = original_record(row)
    try:
        content = build_report(record)  # 只把已授权的正式记录送去整理，不读取可编辑草稿。
    except Exception as error:
        logger.warning("报告生成未完成，错误类型：%s", type(error).__name__)  # 不记录原话、令牌或模型回复。
        raise ReportError(503, "报告生成失败，可重新生成。已确认的随访记录已保存。") from error
    with closing(get_connection()) as conn, conn:  # 正常退出提交，发生任何异常自动回滚，再关闭连接。
        # 开写锁后重查一次；调用模型或检索资料的等待时间不占用这个锁。
        conn.execute("BEGIN IMMEDIATE")
        row = current_case(conn, user, task_id)
        # 等待模型期间另一个请求可能已经生成成功，第一份持久化结果为准。
        if not generation_needed(row, record_id):
            return receipt
        report_id = "RP_" + record_id  # reports.record_id 另有数据库唯一约束。
        # 生成时使用事务里重新核实的任务医生名片；以后的改派或职称变更不会覆盖当次信息。
        content = dict(content)
        content["_responsible_doctor"] = doctor_card(row, "doctor")
        conn.execute("""INSERT INTO reports(report_id,record_id,task_id,content_json,status,generation_mode,created_at)
                        VALUES(?,?,?,?,'pending_review','rag',CURRENT_TIMESTAMP)""",
                     (report_id, record_id, task_id, json.dumps(content, ensure_ascii=False)))
        # 报告和任务状态必须一起提交，医生重新打开任务即可看见待审核报告。
        conn.execute("UPDATE followup_tasks SET task_status='in_review' WHERE task_id=?", (task_id,))
        # 记录是哪位账号触发生成、关联的正式记录是什么，便于事后追溯。
        audit(conn, user, task_id, report_id, "report_generated", {"record_id": record_id})
        save_receipt(conn, user["user_id"], user["role"], "generate", record_id, "随访报告已生成，等待医生审核。",
                     task_id=task_id, record_id=record_id)
    return receipt


def review_report(user, task_id, request):
    """负责医生按报告编号和版本审核、退回或发布。精确重试返回原回执，其他过时操作为 409。"""
    if user["role"] != "doctor":
        raise ReportError(403, "仅负责医生可以审核或发布报告。")
    action = request.action
    # 没有补充要求的退回对患者没有操作依据，拒绝这次状态变化。
    if action == "return" and not request.note:
        raise ReportError(422, "请说明患者需要补充什么。")
    # 医生修改后的摘要存入同一报告的新版本，不能用空字符串覆盖。
    if action == "edit" and not request.summary:
        raise ReportError(422, "请填写修改后的报告摘要。")
    if action != "edit" and (request.summary or request.abnormal_points is not None or request.keep_sources is not None):
        raise ReportError(422, "只有修改后通过可以更改报告内容。")
    if action == "publish" and request.note:
        raise ReportError(422, "发布时不能变更已审核意见。")
    details = request.model_dump()  # 保存原请求，既能还原修改内容，也能区分重试和冲突。
    # 回执给出本次动作完成时的版本；精确重试会拿到完全相同的回执。
    receipt = {"report_id": request.report_id, "version": request.expected_version + 1}
    with closing(get_connection()) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")  # 两位同时打开页面时，只接受首先成功的版本。
        row = current_case(conn, user, task_id)
        # 记录重交后报告编号改变，即使新旧都叫“版本 1”也不能混审。
        if row["report_id"] != request.report_id:
            raise ReportError(409, "报告已变化，请刷新后查看最新提交。")
        if row["version"] != request.expected_version:
            # 查当前医生的历史操作，排除其他医生碰巧提交了相同内容的情况。
            previous = conn.execute("""SELECT details_json FROM audit_logs
                WHERE actor_id=? AND task_id=? AND report_id=? AND action=? ORDER BY id DESC LIMIT 1""",
                (user["user_id"], task_id, request.report_id, "report_" + action)).fetchone()
            if previous and json.loads(previous["details_json"]).get("request") == details:
                return receipt  # 网络丢失成功响应，再点同一个动作也不新增审计。
            raise ReportError(409, "报告版本已更新，请刷新后重新核对。")
        # 已完成或需补充的任务没有待审核动作，未命中精确重试则拒绝。
        if row["task_status"] != "in_review":
            raise ReportError(409, "当前任务不在审核中。")
        if action == "publish":
            if row["status"] != "approved":
                raise ReportError(409, "请先完成医生审核，再发布给患者。")
            conn.execute("UPDATE reports SET status='published', version=version+1 WHERE report_id=?",
                         (request.report_id,))  # 保留审核人、审核时间和已经核定的正文。
            # 发布与任务完成在同一事务中落库，患者不会只读到一半状态。
            task_status = "completed"
        else:
            if row["status"] != "pending_review":
                raise ReportError(409, "只有待审核报告可以审核或退回。")
            # 修改的是数据库刚取出的内容，风险和来源不会由浏览器任意指定。
            content = json.loads(row["content_json"])
            status = "approved"
            task_status = "in_review"  # 通过后待发布，此时患者仍看不到报告。
            if action == "return":
                status = "returned"
                task_status = "needs_info"  # 原版本留存，下一次提交产生新的 record_id。
            if action == "edit":
                original = dict(content)  # 各字段整体替换，不原地修改旧列表，保留医生看到的原稿。
                content["summary"] = request.summary
                # 未传此字段说明保留原稿；传空列表则表示医生核对后删除全部关注项。
                if request.abnormal_points is not None:
                    for point in request.abnormal_points:
                        if not point.strip() or len(point) > 2000:
                            raise ReportError(422, "每一项需关注情况请填写 1 至 2000 个字。")
                    content["abnormal_points"] = request.abnormal_points
                if request.keep_sources is not None:
                    sources = []
                    suggestions = []
                    # 例如 [2] 表示仅保留原稿第二条建议，最终显示时会重新编号为〔1〕。
                    for number in request.keep_sources:
                        if number < 1 or number > len(original["sources"]):
                            raise ReportError(422, "所选引用超出了原稿范围。")
                        source = original["sources"][number - 1]
                        # 防止请求重复选择一条原文，让多条建议实际都指向同一出处。
                        if source in sources:
                            raise ReportError(422, "同一条引用只能选择一次。")
                        sources.append(source)  # 只从已保存原文取值，医生页面也不能上传一个来源冒充引用。
                        suggestions.append(f"{source['quote']}〔{len(sources)}〕")
                    content["sources"] = sources
                    content["recommendations"] = suggestions
                # 如果没改任何业务字段，提示直接通过，审计中的操作名才真实。
                if content == original:
                    raise ReportError(422, "报告尚未修改，可选择直接通过。")
                # 名片单独保存，修改审计的 original_content 只记录医生核对的业务正文。
                original.pop("_responsible_doctor", None)
                original.pop("_reviewer", None)
                details["original_content"] = original  # 即使此次删除建议，也能还原带引用的原稿。
            # 审核动作只能由事务核实过的当前任务医生执行，审核人可能不同于生成时负责人。
            content["_reviewer"] = doctor_card(row, "doctor")
            conn.execute("""UPDATE reports SET content_json=?, status=?, reviewer_id=?,
                            reviewed_at=CURRENT_TIMESTAMP, review_note=?, version=version+1 WHERE report_id=?""",
                         (json.dumps(content, ensure_ascii=False), status, row["doctor_id"],
                          request.note, request.report_id))
        # 状态与审核意见共享事务，写审计失败时也会一起撤销。
        conn.execute("UPDATE followup_tasks SET task_status=? WHERE task_id=?", (task_status, task_id))
        if action == "return":
            # 与退回意见一起提交补充期限；重复审核请求在版本检查处返回，不延长同一窗口。
            schedule_store.reset_return_window(conn, task_id)
        audit(conn, user, task_id, request.report_id, "report_" + action,
              {"request": request.model_dump(), "original_content": details.get("original_content")})
        actions = {"approve":"审核已通过，等待发布。", "edit":"报告修改已保存并审核通过，等待发布。",
                   "return":"报告已退回，等待患者补充。", "publish":"报告已发布给患者。"}
        save_receipt(conn, user["user_id"], user["role"], action, f"{request.report_id}:{request.expected_version}",
                     actions[action], task_id=task_id, **receipt)
    return receipt
