from langchain_core.tools import tool
from langchain.tools import ToolRuntime

from rag.rag_service import RagService
from utils.followup_store import (
    list_followup_tasks,
    list_discharges,
    list_recent_records,
)


# rag 是知识库服务对象：vector_store 管理向量库，rag_search() 检索参考资料。
# 医疗记录走 SQLite 查询，公共医学问题走 RAG，两类来源分别处理。
rag = RagService()


@tool(description="查询当前患者自己的随访任务；医生查询分配给自己的任务；已进入任务时仅返回当前任务")
def get_my_followup_tasks(runtime: ToolRuntime) -> list[dict]:
    """输入后端运行上下文，输出授权任务字典列表；无记录返回 []。

    例如 context 中 task_id="F001"，只读取 F001，不读取同账号的其他任务。
    本工具只读数据库；查询失败交给外层工具监控处理。
    """
    # runtime 由 Agent 框架注入，不是模型需要填写的工具参数。
    # context 是后端提供的字典，例如 {"user_id": "U001", "role": "patient", "task_id": "F001"}。
    return list_followup_tasks(
        user_id=runtime.context["user_id"],
        role=runtime.context["role"],
        task_id=runtime.context.get("task_id"),  # 未进入任务时为 None，保留原查询方式。
    )


@tool(description="查询授权的出院信息；已进入任务时仅查询当前任务对应的出院记录")
def get_my_discharges(runtime: ToolRuntime) -> list[dict]:
    """输入后端身份与任务范围，输出出院信息列表；无记录返回 []。

    例如 task_id="F001"，只返回该任务对应的出院信息，不混入其他住院记录。
    本工具只读数据库，不根据用户在聊天里提供的编号改变范围。
    """
    # 身份取自登录上下文；聊天中出现的姓名、编号不能扩大查询范围。
    return list_discharges(
        user_id=runtime.context["user_id"],
        role=runtime.context["role"],
        task_id=runtime.context.get("task_id"),  # 由服务器限定关联的出院记录。
    )


@tool(description="查询获授权的最近20条随访记录，按提交时间倒序；已进入任务时仅查询当前任务的记录")
def get_my_recent_records(runtime: ToolRuntime) -> list[dict]:
    """输入后端身份与任务范围，输出最多 20 条随访记录；无记录返回 []。

    例如 task_id="F001"，返回的每条记录都属于 F001，并保留 record_id/revision。
    本工具只读数据库，不提交数据；查询异常由外层工具监控处理。
    """
    # 返回已提交的原始记录，不返回待审核报告或医生审核意见。
    return list_recent_records(
        user_id=runtime.context["user_id"],
        role=runtime.context["role"],
        task_id=runtime.context.get("task_id"),  # 限定当前任务，不能把其他任务的记录混进来。
    )


@tool(description="检索公共高血压知识库，返回医学参考片段及来源；不能查询个人病历")
def rag_search(query: str) -> str:
    # query 是模型整理出的检索问题，例如“家庭血压测量前需要注意什么”。
    return rag.rag_search(query)


@tool(description="切换到随访报告整理提示词；只准备文字草稿，不保存、不审核、不发布")
def fill_context_for_report() -> str:
    # 中间件会根据这个工具名切换提示词，工具本身不修改数据库。
    return "已切换报告整理提示词；生成内容仅为草稿，尚未保存或经过医生审核。"
