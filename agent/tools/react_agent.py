from langchain.agents import create_agent
from uuid import uuid4
from langgraph.checkpoint.memory import InMemorySaver
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (
    rag_search, get_my_followup_tasks, get_my_discharges,
    get_my_recent_records, fill_context_for_report,
)
from agent.tools.middle_ware import monitor_tool, log_before_model, report_prompt_switch
from langchain_core.messages import AIMessage, ToolMessage, AIMessageChunk
from langchain.agents.middleware import ToolCallLimitMiddleware


class ReactAgent:
    def __init__(self, user_id: str, role: str, task_id: str | None = None):
        """创建当前身份和任务的 Agent；不提交任何业务记录。

        输入示例：user_id="U001"、role="patient"、task_id="F001"。
        task_id=None 保留通用查询；有编号时，工具只能读取该授权任务范围。
        构造函数不返回业务数据，创建结果是具有独立记忆的 ReactAgent 对象。
        模型或框架配置异常由外层 chat_session 捕获并转为错误事件。
        """
        # 创建对象时生成一次，后续提问继续使用它
        self.thread_id = str(uuid4())  # uuid4()：生成一个随机的唯一标识，用来区分会话。 eg: 0517a8d6-693e-4a4d-a8d3-3f5ff53cf4f8
        self.user_id = str(user_id)  # 是这个用户在用 agent
        self.role = role  # 字符串：patient 表示患者，doctor 表示医生。
        self.task_id = task_id  # str 或 None：由后端校验归属的任务编号，不从聊天文字提取。

        # 身份由后端查登录凭证得到，模型只能选择查询工具，不能指定其他用户。
        medical_tools = [
            rag_search, get_my_followup_tasks, get_my_discharges,
            get_my_recent_records,
        ]
        # 患者只查询信息；报告草稿辅助工具仅提供给医生。
        if role == "doctor":
            medical_tools.append(fill_context_for_report)

        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=medical_tools,
            middleware=[monitor_tool, log_before_model, report_prompt_switch,
                        ToolCallLimitMiddleware(
                            run_limit=7,
                            exit_behavior="error",
                        ),
                        ],
            checkpointer=InMemorySaver(),  # InMemorySaver：把 Agent 的检查点保存在当前 Python 进程的内存中。
        )

    def execute_stream(self, query: str):
        """输入一条问题文本，逐个输出页面约定的聊天事件。

        例如 query="这次随访什么时候到期？"，输出 status、delta、answer 等字典。
        user_id、role、task_id 使用创建对象时保存的服务端值，不从问题中提取。
        本方法会调用模型和只读工具、更新当前 Agent 的内存；不提交随访记录。
        推理或工具异常交给 chat_session 处理，未完成的回答不能作为正常结果保存。
        """
        input_dict = {
            "messages": [
                {"role": "user", "content": query}
            ]
        }

        tool_names = {
            "get_my_followup_tasks": "查询随访任务",
            "get_my_discharges": "查询出院信息",
            "get_my_recent_records": "查询近期血压记录",
            "fill_context_for_report": "整理报告草稿",
            "rag_search": "检索高血压知识库",
        }

        shown_tool_calls = set()
        final_answer = ""

        yield {
            "type": "status",
            "content": "正在处理你的问题……",
        }

        for mode, data in self.agent.stream(
                input_dict,
                # values 每一步执行完后的完整状态（比如大模型决定调用某个工具，或者工具返回了结果）
                stream_mode=["messages", "values"],  # messages一个字一个字蹦出来的模型输出（Token 流）
                version="v1",
                context={
                    "report": False,
                    "user_id": self.user_id,
                    "role": self.role,
                    # ToolRuntime 会接收这个值；模型的工具参数中不开放任务范围。
                    "task_id": self.task_id,
                },
                config={
                    "configurable": {
                        "thread_id": self.thread_id,
                    },
                    "recursion_limit": 20,
                },
        ):
            # 第一种数据：模型正在生成的文字片段。
            if mode == "messages":
                message_chunk, metadata = data
                # 只接收 Agent 的模型节点生成的内容。
                if metadata.get("langgraph_node") != "model":
                    continue
                if not isinstance(message_chunk, AIMessageChunk):
                    continue
                text = message_chunk.text
                if text:
                    # 不等完整答案生成，马上交给页面。
                    yield {
                        "type": "delta",
                        "content": text,
                    }

            # 第二种数据：执行步骤后的完整状态。
            elif mode == "values":
                message = data["messages"][-1]
                if isinstance(message, AIMessage):
                    if message.tool_calls:
                        for tool_call in message.tool_calls:
                            call_id = tool_call["id"]
                            if call_id in shown_tool_calls:
                                continue
                            shown_tool_calls.add(call_id)
                            # 本轮要调用工具，前面的文字可能只是临时说明。
                            final_answer = ""
                            yield {
                                "type": "reset",
                                "content": "",  # # 清空前端已显示的字
                            }
                            name = tool_call["name"]
                            label = tool_names.get(name, name)  # eg:正在 "查询使用记录"
                            yield {
                                "type": "status",
                                "content": f"正在{label}……",
                            }
                    elif message.text:
                        # 保留完整答案，供最后确认和保存。
                        final_answer = message.text
                elif isinstance(message, ToolMessage):
                    yield {
                        "type": "status",
                        "content": "工具已返回结果，正在继续处理……",
                    }

        # 正常执行结束后，交出最终完整答案。
        # 如果没有答案，则交出空字符串，由页面显示提示。
        yield {
            "type": "answer",
            "content": final_answer,
        }


# 请通过 backend.py 的登录接口使用 Agent，避免调试入口绕过身份验证。
