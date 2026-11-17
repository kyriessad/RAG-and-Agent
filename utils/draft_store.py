"""随访采集草稿：校验字段、检查任务归属，并读写已有的 task_drafts 表。"""

import json
import sqlite3
from typing import Literal  # 固定选项同时生成工具和接口的字段说明。

from pydantic import BaseModel, ConfigDict, Field

from utils.user_store import get_connection


class DraftFields(BaseModel):
    """草稿字段，None 表示尚未说明；允许只填写一部分，不代表正式随访记录。"""

    # 拒绝额外字段和自动类型转换，例如不把 True 当成血压 1。
    model_config = ConfigDict(extra="forbid", strict=True)
    # Literal 是固定选项；Field 描述也会进入模型看到的工具字段说明。
    bp_measurement_status: Literal["measured", "not_measured"] | None = Field(default=None, description="本次明确已测量或尚未测量；没说则省略")
    systolic_bp: int | None = Field(default=None, ge=1, le=999, description="大压，整数，单位 mmHg")
    diastolic_bp: int | None = Field(default=None, ge=1, le=999, description="小压，整数，单位 mmHg")
    measured_at: str | None = Field(default=None, max_length=100, description="测量时间完整原话子句，不补日期")
    symptoms: str | None = Field(default=None, max_length=1000, description="是否不舒服的完整原话子句，保留否定")
    medication_adherence: Literal["regular", "missed", "not_prescribed"] | None = Field(default=None, description="明确按时/漏服/未开药；不确定则省略")
    missed_dose_reason: str | None = Field(default=None, max_length=500, description="漏服原因完整原话子句")
    adverse_effects: str | None = Field(default=None, max_length=1000, description="服药后是否不舒服的完整原话子句")


class SaveDraftData(BaseModel):
    """页面保存请求，例如 {expected_revision: 0, data: {systolic_bp: 138}}。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    expected_revision: int = Field(ge=0)  # 页面读取时的版本，不能自行增加后再保存。
    data: DraftFields  # 完整表单；没有填写的字段会成为 None。


def validate_draft(data):
    """输入字段字典，输出规范化字典；错误抛 ValueError，不请求网络、不写库。

    例如 {"symptoms": "  "} 变成 symptoms=None；不把空症状变为“没有不舒服”。
    数字范围仅为输入格式检查，不是风险分层或医疗判断。
    """
    result = DraftFields.model_validate(data).model_dump()  # Pydantic 检查类型和长度。
    for name, value in result.items():  # 文本只清理首尾空格，不推断业务事实。
        if isinstance(value, str):
            result[name] = value.strip() or None
    if result["bp_measurement_status"] not in (None, "measured", "not_measured"):
        raise ValueError("测量情况只能选择未说明、已测量或尚未测量。")
    if result["medication_adherence"] not in (None, "regular", "missed", "not_prescribed"):
        raise ValueError("请从页面提供的选项中选择服药情况。")
    if result["medication_adherence"] in ("regular", "not_prescribed") and result["missed_dose_reason"] is not None:
        raise ValueError("当前未漏服时，请核对并清空不再适用的漏服原因。")
    if result["medication_adherence"] == "not_prescribed" and result["adverse_effects"] is not None:
        raise ValueError("当前没有开药时，请核对并清空不再适用的服药后不适描述。")
    if result["bp_measurement_status"] == "not_measured":
        for name in ("systolic_bp", "diastolic_bp", "measured_at"):
            if result[name] is not None:
                raise ValueError("尚未测量时，请先清空本次血压和测量时间。")
    high = result["systolic_bp"]  # 收缩压，页面用“大压”解释。
    low = result["diastolic_bp"]  # 舒张压，页面用“小压”解释。
    if high is not None and low is not None and high <= low:
        raise ValueError("大压应高于小压，请核对是否填反；不会自动交换数值。")
    return result


def missing_draft_fields(data):
    """输入校验后的字段字典，输出待追问的中文字段名列表；无副作用。

    明确尚未测量时不追问数值；有漏服才要求原因；未说明不等于没有症状。
    """
    required = {"bp_measurement_status": "是否测过血压", "symptoms": "有没有不舒服",
                "medication_adherence": "服药情况"}
    if data["bp_measurement_status"] != "not_measured":
        required.update({"systolic_bp": "大压", "diastolic_bp": "小压", "measured_at": "测量时间"})
    if data["medication_adherence"] == "missed":
        required["missed_dose_reason"] = "漏服原因"
    if data["medication_adherence"] != "not_prescribed":
        required["adverse_effects"] = "服药后有没有不舒服"
    missing = []  # 例如 ["测量时间", "有没有不舒服"]。
    for name, label in required.items():
        if data[name] is None:
            missing.append(label)
    return missing


def require_patient_task(conn, user_id, role, task_id):
    """用当前连接核对患者、任务及实时角色，返回状态；越权或不存在均抛 PermissionError。

    调用方已开启事务，因此归属检查与草稿读写处在同一数据库快照中。
    """
    if role != "patient":  # 医生不能通过此患者接口查看或修改未确认的采集草稿。
        raise PermissionError("任务不存在或无权访问。")
    row = conn.execute(
        """SELECT t.task_status FROM followup_tasks AS t
           JOIN discharges AS d ON d.discharge_id = t.discharge_id
           JOIN patients AS p ON p.patient_id = d.patient_id
           JOIN users AS u ON u.user_id = p.user_id
           WHERE t.task_id = ? AND p.user_id = ? AND u.role = 'patient'""",
        (task_id, user_id),  # 参数绑定，不把患者输入拼接到 SQL。
    ).fetchone()
    if row is None:
        raise PermissionError("任务不存在或无权访问。")
    return row["task_status"]


def read_draft(conn, task_id, task_status):
    """在已授权的连接上读取草稿，返回接口字典；不存在返回版本 0，不创建记录。

    内容无法解析时抛数据库错误，避免把损坏或不兼容的旧草稿当成空白覆盖。
    """
    row = conn.execute("SELECT draft_json, revision FROM task_drafts WHERE task_id = ?", (task_id,)).fetchone()
    revision = 0  # 尚未保存的页面草稿以版本 0 表示。
    data = DraftFields().model_dump()  # 所有字段都是 None，没有默认血压或默认症状。
    if row is not None:
        try:
            data = validate_draft(json.loads(row["draft_json"]))
            revision = row["revision"]
        except ValueError as error:
            raise sqlite3.DatabaseError("已有草稿格式无法读取，请检查后端记录。") from error
    return {"task_id": task_id, "revision": revision, "data": data,
            "missing_fields": missing_draft_fields(data),
            "editable": task_status in ("pending", "in_progress", "needs_info")}


def get_task_draft(user_id, role, task_id):
    """输入服务端身份和任务编号，返回持久草稿；只读，不调用模型。

    例如患者读取 F001 得到 {task_id: F001, revision: 0, data: {...}, ...}。
    越权抛 PermissionError，数据库异常交给接口统一处理。
    """
    conn = get_connection()  # 使用 user_store 配置的库；自动化测试替换为临时库。
    try:
        conn.execute("BEGIN")  # 将授权与读取放在同一只读事务中。
        status = require_patient_task(conn, user_id, role, task_id)
        return read_draft(conn, task_id, status)
    finally:
        conn.close()  # 只读事务随连接关闭，不产生业务写入。


def save_task_draft(user_id, role, task_id, expected_revision, data):
    """验证并保存完整采集草稿，返回新版本；只写 task_drafts，不提交随访。

    输入旧版本 0，首次成功返回版本 1。越权抛 PermissionError；
    旧版本/任务不可编辑抛 RuntimeError；字段错误抛 ValueError；失败自动回滚。
    """
    data = validate_draft(data)  # 手工表单与模型提议经过同一套类型、组合检查。
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")  # 取得写锁，避免两个会话同时读旧版本并覆盖。
        status = require_patient_task(conn, user_id, role, task_id)  # 写入时再次核对实时归属。
        current = read_draft(conn, task_id, status)
        if not current["editable"]:
            raise RuntimeError("当前任务已提交或结束，不能修改采集草稿。")
        if current["revision"] != expected_revision:
            raise RuntimeError("草稿已被更新，请重新加载最新草稿后再修改；本次内容未保存。")
        revision = current["revision"] + 1  # 版本由后端增加，不由模型或页面决定。
        missing = missing_draft_fields(data)
        needs_clarification = int(bool(missing))  # 有缺项为 1，已回答所需字段为 0。
        payload = json.dumps(data, ensure_ascii=False)  # 只保存规范字段，不存模型原始回答。
        row = conn.execute("SELECT task_id FROM task_drafts WHERE task_id = ?", (task_id,)).fetchone()
        if row is None:
            conn.execute("INSERT INTO task_drafts(task_id, draft_json, revision, needs_clarification) VALUES(?,?,?,?)",
                         (task_id, payload, revision, needs_clarification))
        else:
            conn.execute("UPDATE task_drafts SET draft_json=?, revision=?, needs_clarification=? WHERE task_id=?",
                         (payload, revision, needs_clarification, task_id))
        conn.commit()  # 全部检查通过才持久保存；不改变 followup_tasks/followup_records。
        return {"task_id": task_id, "revision": revision, "data": data,
                "missing_fields": missing, "editable": True}
    except Exception:
        conn.rollback()  # 错误时保持此前的草稿与版本。
        raise
    finally:
        conn.close()
