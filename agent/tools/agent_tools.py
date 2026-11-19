from typing import Any

# @tool 把普通 Python 函数注册成模型可选择的工具，描述会进入工具清单。
from langchain_core.tools import tool
# ToolRuntime 由框架提供，其中的 context 保存后端已核对的身份。
from langchain.tools import ToolRuntime
# 复用页面草稿的字段校验，避免聊天提议和手工表单接受不同类型。
from utils.draft_store import DraftFields
# 校验依据来自本次任务内的患者原话，提议本身仅在内存中保存。
from agent.tools.draft_collection import propose_draft_changes

from rag.rag_service import RagService
# 三种医疗查询共用数据库授权逻辑，传给模型的结果已经过筛选。
from utils.followup_store import (
    list_patients,
    list_followup_tasks,
    list_discharges,
    list_recent_records,
)
from utils.report_store import report_view
from utils.risk_store import patient_doctor_contact


# 聊天工具用 search_with_sources 取知识原文，正式报告复用同一检索器。
# 医疗记录走 SQLite 查询，公共医学问题走 RAG，两类来源分别处理。
rag = RagService()


@tool(description="查询当前授权范围内的患者基本资料与医患关系；患者查询本轮随访医生或主管医生电话时使用")
def get_related_people(runtime: ToolRuntime[Any]) -> list[dict]:
    """读取服务端选定范围内的医患资料；无档案时返回 []。只读，不接受姓名作授权参数。"""
    context = runtime.context
    profiles = list_patients(context["user_id"], context["role"],
                             patient_id=context.get("patient_id"), task_id=context.get("task_id"))
    if profiles and context["role"] == "patient":
        profiles[0]["doctor_contact"] = patient_doctor_contact(context["user_id"], context.get("task_id"))
    return profiles


@tool(description="查询页面当前这一次随访的报告进度、负责医生、审核信息以及当前角色可阅读的正文")
def get_current_report(runtime: ToolRuntime[Any]) -> dict:
    """输入服务端任务上下文，返回与页面相同的报告视图。只读，越权时由存储层拒绝。"""
    context = runtime.context
    return report_view(context, context["task_id"])


@tool(description="查询当前患者自己的随访任务；医生查询分配给自己的任务；已进入任务时仅返回当前任务")
def get_my_followup_tasks(runtime: ToolRuntime[Any]) -> list[dict]:
    """输入后端运行上下文，输出授权任务字典列表；无记录返回 []。

    例如 context 中 task_id="F001"，只读取 F001，不读取同账号的其他任务。
    本工具只读数据库；查询失败交给外层工具监控处理。
    """
    # runtime 由 Agent 框架注入，不是模型需要填写的工具参数。
    # context 是后端提供的字典，例如 {"user_id": "U001", "role": "patient", "task_id": "F001"}。
    return list_followup_tasks(
        # 例如 U001 来自登录会话，工具参数中没有可供模型填写的用户名。
        user_id=runtime.context["user_id"],
        # patient 与 doctor 会在同一查询中匹配不同的归属条件。
        role=runtime.context["role"],
        task_id=runtime.context.get("task_id"),  # 未进入任务时为 None，保留原查询方式。
        patient_id=runtime.context.get("patient_id"),  # 医生打开某人的档案后仅查询该患者。
    )


@tool(description="查询授权的出院信息；已进入任务时仅查询当前任务对应的出院记录")
def get_my_discharges(runtime: ToolRuntime[Any]) -> list[dict]:
    """输入后端身份与任务范围，输出出院信息列表；无记录返回 []。

    例如 task_id="F001"，只返回该任务对应的出院信息，不混入其他住院记录。
    本工具只读数据库，不根据用户在聊天里提供的编号改变范围。
    """
    # 身份取自登录上下文；聊天中出现的姓名、编号不能扩大查询范围。
    return list_discharges(
        # 数据库将登录账号关联到患者或负责医生，姓名不是查询授权凭证。
        user_id=runtime.context["user_id"],
        # 结合下一行的任务编号，只查看本次任务所属的出院记录。
        role=runtime.context["role"],
        task_id=runtime.context.get("task_id"),  # 由服务器限定关联的出院记录。
        patient_id=runtime.context.get("patient_id"),
    )


@tool(description="查询获授权的最近20条随访记录，按提交时间倒序；已进入任务时仅查询当前任务的记录")
def get_my_recent_records(runtime: ToolRuntime[Any]) -> list[dict]:
    """输入后端身份与任务范围，输出最多 20 条随访记录；无记录返回 []。

    例如 task_id="F001"，返回的每条记录都属于 F001，并保留 record_id/revision。
    本工具只读数据库，不提交数据；查询异常由外层工具监控处理。
    """
    # 返回已提交的原始记录，不返回待审核报告或医生审核意见。
    return list_recent_records(
        # 即便同名患者也用不同 user_id 区分；不会因聊天自报姓名扩大权限。
        user_id=runtime.context["user_id"],
        # 医生读取的是分配给自己的任务记录，患者读取自己的记录。
        role=runtime.context["role"],
        task_id=runtime.context.get("task_id"),  # 限定当前任务，不能把其他任务的记录混进来。
        patient_id=runtime.context.get("patient_id"),
    )


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
