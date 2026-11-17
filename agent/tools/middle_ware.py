from typing import Callable

from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompts, load_system_prompts
from agent.tools.draft_collection import collection_prompt


# 工具执行的监控，参数：request请求的数据封装,  handler执行的函数本身
@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command]
) -> ToolMessage | Command:

    logger.info(f"[tool monitor]执行工具: {request.tool_call['name']}")
    # 记录工具名称即可，参数可能含患者描述，不把完整健康信息写入日志。

    try:
        # 真正开始执行工具
        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True
        result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")
        return result
    except Exception as e:
        # Pydantic 错误可能带 input_value；只记录类型，避免患者原话进入日志。
        logger.error("工具%s调用失败，错误类型：%s", request.tool_call['name'], type(e).__name__)
        raise e


# 在模型执行前输出日志
@before_model
def log_before_model(
        state: AgentState,  # 整个Agent智能体中的状态记录
        runtime: Runtime,  # 记录了整个执行过程中的上下文信息
):
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")

    # 只记录消息数量，不记录聊天正文、密码或登录凭证。

    return None


# 每次在生成提示词之前调用此函数   prompt_template之前
@dynamic_prompt                 # 每一次在生成提示词之前，调用此函数
def report_prompt_switch(request: ModelRequest):  # 动态切换提示词
    """输入框架模型请求，返回系统提示词及当前采集草稿；不请求网络或写库。"""
    is_report = request.runtime.context.get("report", False)
    if is_report:               # 是报告生成场景，返回报告生成提示词内容
        return load_report_prompts()

    # 每次推理携带后端本轮读取的草稿；患者手工修改优先于旧聊天记忆。
    return load_system_prompts() + collection_prompt(request.runtime.context)
