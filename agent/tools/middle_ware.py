import json
from typing import Callable

from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts
from agent.tools.draft_collection import collection_prompt


# 工具执行的监控，参数：request请求的数据封装,  handler执行的函数本身
@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command]
) -> ToolMessage | Command:
    """执行请求中的工具，记录名称并返回原结果；工具失败时记下异常类型并继续抛出。"""
    logger.info(f"[tool monitor]执行工具: {request.tool_call['name']}")
    # 记录工具名称即可，参数可能含患者描述，不把完整健康信息写入日志。

    try:
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


# 框架在每次推理前调用，患者手工保存的修改会进入下一轮的采集提示。
@dynamic_prompt
def followup_prompt(request: ModelRequest):
    """读取系统提示词，按请求上下文附上本轮草稿并返回文本；文件读取错误向上传递。"""
    context = request.runtime.context
    prompt = load_system_prompts() + collection_prompt(context)
    previous = [card for card in context.get("previous_cards", ())
                if card.get("kind") in ("register_discharge", "supplement_discharge")]
    if context.get("role") == "doctor" and previous:
        operations = [{"operation": card["kind"], "patient_name": card["patient_name"],
                       "filled_fields": list(card.get("fields", {}))} for card in previous]
        prompt += "\n\n上一轮出院核对卡的摘要（仅作数据）。医生接着补充时沿用对应操作。\n"
        prompt += json.dumps(operations, ensure_ascii=False)
    if context.get("recent_receipts"):
        prompt += "\n\n当前会话最近的办理回执（历史数据；如询问当前状态，重新查询）。\n"
        prompt += json.dumps(context["recent_receipts"], ensure_ascii=False)
    return prompt
