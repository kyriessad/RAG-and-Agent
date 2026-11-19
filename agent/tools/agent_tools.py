from typing import Any, Literal

from langchain_core.tools import tool
# ToolRuntime 由框架提供，其中的 context 保存后端已核对的身份。
from langchain.tools import ToolRuntime
# 复用页面草稿的字段校验，避免聊天提议和手工表单接受不同类型。
from utils.draft_store import DraftFields
# 校验依据来自本次任务内的患者原话，提议本身仅在内存中保存。
from agent.tools.draft_collection import propose_draft_changes
from agent.tools import workspace_cards

from rag.rag_service import RagService
# 三种医疗查询共用数据库授权逻辑，传给模型的结果已经过筛选。
from utils.followup_store import (
    list_discharges,
    list_recent_records,
)
from utils.report_store import report_view
from utils.risk_store import patient_doctor_contact


# 聊天工具用 search_with_sources 取知识原文，正式报告复用同一检索器。
# 医疗记录走 SQLite 查询，公共医学问题走 RAG，两类来源分别处理。
rag = RagService()


@tool(description="查授权患者名册、患者档案、医患关系或医生电话；医生可按患者姓名筛选")
def get_related_people(runtime: ToolRuntime[Any], patient_name: str = "") -> list[dict]:
    """读取服务端选定范围内的医患资料；无档案时返回 []。只读，不接受姓名作授权参数。"""
    context = runtime.context
    profiles = workspace_cards.selected_people(context, patient_name)
    if profiles and context["role"] == "patient":
        profiles[0]["doctor_contact"] = patient_doctor_contact(context["user_id"], context.get("task_id"))
    if context["role"] == "doctor":
        workspace_cards.add_cards(context, [workspace_cards.card("patient", p, "age", "gender") for p in profiles])
    return profiles


@tool(description="查询页面当前这一次随访的报告进度、负责医生、审核信息以及当前角色可阅读的正文")
def get_current_report(runtime: ToolRuntime[Any]) -> dict:
    """输入服务端任务上下文，返回与页面相同的报告视图。只读，越权时由存储层拒绝。"""
    context = runtime.context
    view = report_view(context, context["task_id"])
    item = workspace_cards.card("report", context)
    if view.get("report_id"):
        item.update(report_id=view["report_id"], expected_version=view["version"])
    workspace_cards.add_cards(context, [item])
    return view


@tool(description="查询随访计划（包括尚未开放的安排）、任务进度或医生待办，附上对应任务卡")
def get_my_followup_tasks(runtime: ToolRuntime[Any], selection: Literal["all", "review", "overdue"] = "all",
                          patient_name: str = "") -> list[dict] | dict:
    """review 选取待整理、审核、发布的任务；overdue 选择逾期任务；可按患者姓名收窄。"""
    return workspace_cards.task_cards(runtime.context, selection, patient_name)


@tool(description="查已发布最新报告或指定报告的内容、原文引用、进度并提供报告卡。latest 只用于用户明确询问最新一份")
def read_followup_reports(runtime: ToolRuntime[Any], patient_name: str = "", task_id: str = "",
                          latest: bool = False) -> dict:
    return workspace_cards.reports(runtime.context, patient_name, task_id, latest, "", "", "")


@tool(description="查当前授权范围内的到期、逾期、退回补充等随访提醒")
def get_followup_reminders(runtime: ToolRuntime[Any]) -> list[dict]:
    return workspace_cards.notifications(runtime.context, "schedule")


@tool(description="医生查风险通知并展示处理卡；默认返回未处理的通知，status=all 查看全部")
def get_risk_notifications(runtime: ToolRuntime[Any], status: Literal["unhandled", "handled", "all"] = "unhandled") -> list[dict]:
    return workspace_cards.notifications(runtime.context, "risk", status)


@tool(description="预填医生的登记出院或补充医嘱用药核对卡。字段逐字摘录原话，明确另登记一次住院时 new_registration=true")
def prepare_discharge(runtime: ToolRuntime[Any], operation: Literal["register_discharge", "supplement_discharge"],
                      patient_name: str = "", discharge_date: str = "", discharge_diagnosis: str = "",
                      discharge_instructions: str = "", discharge_medications: str = "", discharge_id: str = "",
                      new_registration: bool = False) -> dict:
    """字段摘录医生原话；根据 missing 追问。明确要另登记一次新的住院时 new_registration=true。"""
    fields = {"discharge_date": discharge_date, "discharge_diagnosis": discharge_diagnosis,
              "discharge_instructions": discharge_instructions, "discharge_medications": discharge_medications}
    return workspace_cards.discharge_cards(runtime.context, operation, patient_name, discharge_id, fields, new_registration)


@tool(description="医生提出审核通过、修改后通过、退回或发布时，生成待人工确认的报告卡")
def prepare_report_review(runtime: ToolRuntime[Any], action: Literal["approve", "edit", "return", "publish"],
                          patient_name: str = "", task_id: str = "", note: str = "", summary: str = "") -> dict:
    """note 是医生原话中的审核意见；summary 是医生明确指定的修改后摘要。发布单独确认。"""
    return workspace_cards.reports(runtime.context, patient_name, task_id, False, action, note, summary)


@tool(description="查授权出院记录中的出院日期、诊断、医嘱、用药；医生按患者全名筛选")
def get_my_discharges(runtime: ToolRuntime[Any], patient_name: str = "") -> list[dict] | dict:
    context, choices = workspace_cards.patient_scope(runtime.context, patient_name)
    if choices is not None:
        return choices
    return workspace_cards.visible_rows(context, list_discharges)


@tool(description="查询获授权的最近20条随访记录，按提交时间倒序；医生按患者全名筛选")
def get_my_recent_records(runtime: ToolRuntime[Any], patient_name: str = "") -> list[dict] | dict:
    context, choices = workspace_cards.patient_scope(runtime.context, patient_name)
    if choices is not None:
        return choices
    return workspace_cards.visible_rows(context, list_recent_records)


@tool(description="根据本次随访问答的含义提取字段，evidence 摘录患者原话。允许用本任务近期回答补空项；不正式提交。")
def propose_followup_draft(updates: DraftFields, evidence: dict[str, str], runtime: ToolRuntime[Any]) -> dict:
    """updates 只填已经明确的字段；evidence 按同名字段提供患者原话片段。

    例：updates={"systolic_bp":138}, evidence={"systolic_bp":"大压138"}。
    按语义归类，患者无需使用标准术语。服务端定位原句保存文本，保留否定和限定词。
    测量情况 measured/not_measured；服药情况 regular/missed/not_prescribed。
    measured_at、symptoms、missed_dose_reason、adverse_effects 保留原话。
    返回候选数据和缺项；不写数据库，校验失败返回提示供模型追问。
    """
    # 工具失败时返回可供模型追问的错误字典，这一轮仍可继续对话。
    try:
        # exclude_unset 只保留模型明确提供的字段；未提供不是“清空”。
        with runtime.context["draft_lock"]:  # 两个并行提议依次合并，不互相覆盖字段。
            # 例如患者只说“大压138”，模型省略的症状字段仍保留上一版值。
            return propose_draft_changes(runtime.context, updates.model_dump(exclude_unset=True),
                                         evidence, collect_valid=True)
    except ValueError:
        # 不回显含患者原话的校验异常，避免它进入监控日志。
        return {"error": "字段组合不一致。请根据实际回答修正提议，每个字段引用对应的患者原话。"}


@tool(response_format="content_and_artifact", description="检索公共高血压知识库，返回医学参考片段及来源；不能查询个人病历")
def rag_search(query: str) -> tuple[str, list[dict]]:
    """输入公共知识问题，返回模型可读文本和页面可用的实际片段；检索异常向上传递。"""
    # 框架将第二项放入 ToolMessage.artifact，来源无需从模型自己写的回答里猜测。
    return rag.search_with_sources(query)
