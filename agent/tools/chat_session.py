"""管理登录会话里的 Agent；FastAPI 只负责把消息交到这里。"""
from threading import Lock
from time import monotonic

from utils.logger_handler import logger
from utils.draft_store import get_task_draft, save_task_draft


# 字典的键是登录令牌的哈希，不保存原始令牌。
# 每个值保存一个 Agent、当前任务编号、一把锁、最近使用时间；不同登录会话不会共用记忆。
chat_sessions = {}
sessions_lock = Lock()
knowledge_lock = Lock()
knowledge_ready = False


def create_chat_agent(user_id, role, task_id=None):
    """创建与登录身份、当前任务绑定的 Agent。

    输入示例：user_id="U001"、role="patient"、task_id="F001"。
    task_id=None 表示未指定任务，保留原来的通用查询方式。
    输出：一个 ReactAgent 对象；不保存随访记录或修改任务状态。
    首次调用会加载模型和同步知识库；配置或同步失败时异常交给 stream_chat 处理。
    """
    # 到第一次聊天才加载模型。没有配置模型密钥，也可以先正常登录。
    from agent.tools.react_agent import ReactAgent
    from agent.tools.agent_tools import rag

    global knowledge_ready
    # 一个后端进程启动后同步一次知识库；修改资料后重启后端即可重新检查。
    with knowledge_lock:
        if not knowledge_ready:
            rag.vector_store.load_document()
            knowledge_ready = True

    # 编号由接口校验归属后传入，不能由模型在工具参数中指定。
    return ReactAgent(user_id=user_id, role=role, task_id=task_id)


def forget_session(session_key):
    # 退出登录时清掉这个会话的 Agent；下一次登录会获得新会话。
    with sessions_lock:
        chat_sessions.pop(session_key, None)


def stream_chat(session_key, user_id, role, message, task_id=None):
    """在已授权的任务范围内处理一条消息，逐个返回流式事件。

    输入：会话哈希、服务端身份、消息文本，以及可选任务编号。
    例如 task_id="F001"、message="这个任务什么时候到期？"。
    输出：依次产生 {"type": "status", "content": "..."} 等事件。
    同一任务保留多轮记忆；身份或任务改变时重建 Agent，避免混入旧任务信息。
    患者任务每轮读取持久草稿；仅在模型完整结束后保存经校验的提议。
    同会话并发、模型失败或版本冲突返回 error，不提交正式记录或改变任务状态。
    """
    # monotonic() 是只向前走的计时值，用于清理一天没有使用的内存会话。
    now = monotonic()
    with sessions_lock:
        for key, item in list(chat_sessions.items()):
            if now - item["last_used"] > 86400 and not item["lock"].locked():
                del chat_sessions[key]

        if session_key not in chat_sessions:
            chat_sessions[session_key] = {
                "agent": None, "task_id": None, "lock": Lock(), "last_used": now,
            }
        session = chat_sessions[session_key]

    # 同一个会话上一轮没结束时，不让下一轮同时改写 Agent 记忆。
    if not session["lock"].acquire(blocking=False):
        yield {"type": "error", "content": "上一条消息还在处理中，请稍后再试。"}
        return

    try:
        # 草稿按任务保存，不依赖聊天记忆；换登录或重启后仍读取同一份业务数据。
        draft = None
        if role == "patient" and task_id is not None:
            draft = get_task_draft(user_id, role, task_id)
        # 如果身份或任务范围发生变化，丢弃旧 Agent，重新建立独立的任务记忆。
        agent = session["agent"]
        if agent is not None:
            # 普通 if 分开检查，便于看清三种需要重建 Agent 的原因。
            if agent.user_id != user_id or agent.role != role:
                session["agent"] = None  # 登录身份变了，旧权限不能继续使用。
            if session["task_id"] != task_id:
                session["agent"] = None  # 切换任务或回到通用聊天，不沿用另一范围的记忆。
        session["task_id"] = task_id  # 锁内记录范围；同会话不能同时处理两项任务。
        if session["agent"] is None:
            yield {"type": "status", "content": "正在准备 Agent 和医学知识库……"}
            # 同时传入身份与任务范围，后续工具查询由这些服务端值限定。
            session["agent"] = create_chat_agent(user_id, role, task_id=task_id)

        # 普通查询沿用原签名；患者任务把最新草稿传入每一轮推理。
        if draft is None:
            events = session["agent"].execute_stream(message)
        else:
            events = session["agent"].execute_stream(message, draft=draft)
        proposal = None  # 暂存工具提议，模型失败时不能把半成品写入业务库。
        answer = None  # 最终回答等草稿保存成功后才发出。
        failed = False
        for event in events:
            if event["type"] == "draft_proposal":
                proposal = event["content"]  # 内部事件不直接发给页面。
            elif event["type"] == "answer":
                answer = event
            else:
                if event["type"] == "error":
                    failed = True
                yield event  # status/reset/delta 继续实时显示。
        if failed or answer is None or not answer["content"]:
            session["agent"] = None  # 不沿用未完成推理，也不保存它的提议。
            if not failed:
                yield {"type": "error", "content": "未收到完整回答，草稿未更新，请重试。"}
            return
        if draft is not None and proposal is not None and proposal != draft["data"]:
            try:
                # 写事务重新核对身份、归属、状态和旧版本；不会覆盖另一个页面的新修改。
                saved = save_task_draft(user_id, role, task_id, draft["revision"], proposal)
            except (PermissionError, RuntimeError, ValueError):
                session["agent"] = None  # 提议没存成，不能让后续记忆把它当成成功。
                yield {"type": "error", "content": "本轮草稿未保存：任务权限、状态或草稿内容已变化，请重新加载草稿后重试。"}
                return
            yield {"type": "draft", "content": saved}  # 页面只接收后端确认保存的新版本。
        yield answer
    except GeneratorExit:
        # 页面中途断开时也丢弃未完成的推理记忆，下次从干净状态继续。
        session["agent"] = None
        raise
    except Exception as error:
        # 工具参数错误可能含患者原话；这里只记类型，不输出原始参数或异常堆栈。
        logger.error("Agent 对话处理失败，错误类型：%s", type(error).__name__)
        # 失败的一轮可能留下未完成的工具调用，下一次使用新的 Agent。
        session["agent"] = None
        yield {"type": "error", "content": "本次回答失败，请重试；若持续失败，请检查后端模型配置和日志。"}
    finally:
        session["last_used"] = monotonic()
        session["lock"].release()
