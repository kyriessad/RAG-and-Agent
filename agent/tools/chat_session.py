"""管理登录会话里的 Agent；FastAPI 只负责把消息交到这里。"""
# 这里有三种锁：全体会话目录、知识库首次同步、单次登录的推理。
from threading import Lock
from time import monotonic

# 日志只记录工具名或错误类型，避免把患者整段聊天写到磁盘。
from utils.logger_handler import logger
# 每次推理前读取已保存草稿，推理后按旧版本尝试合并。
from utils.draft_store import get_task_draft, save_task_draft
from utils import followup_store
from utils.risk_rules import evaluate_risk
# save_message_risk 在模型创建前可以独立提交通知。
from utils.risk_store import save_message_risk


# 字典的键是登录令牌的哈希，不保存原始令牌。
# 每个值保存一个 Agent、当前任务编号、一把锁、最近使用时间；不同登录会话不会共用记忆。
chat_sessions = {}
# 只在取用和清理字典时持锁，模型运行期间其他账号可以同时聊天。
sessions_lock = Lock()
# 锁住首次同步，两个账号同时打开聊天也不会同时重建向量索引。
knowledge_lock = Lock()
# 同步完整成功才把它改成 True；失败后保持可重试。
knowledge_ready = False


def create_chat_agent(user_id, role, task_id=None, patient_id=None, mode="query"):
    """创建与登录身份、当前任务绑定的 Agent。

    输入示例：user_id="U001"、role="patient"、task_id="F001"。
    task_id=None 表示未指定任务，保留原来的通用查询方式。
    输出：一个 ReactAgent 对象；不保存随访记录或修改任务状态。
    首次调用会加载模型和同步知识库；配置或同步失败时异常交给 stream_chat 处理。
    """
    # 到第一次聊天才加载模型。没有配置模型密钥，也可以先正常登录。
    from agent.tools.react_agent import ReactAgent
    ready_knowledge()  # 聊天和独立报告生成共用相同的向量库，加载失败可在下次重试。
    return ReactAgent(user_id=user_id, role=role, task_id=task_id, patient_id=patient_id, mode=mode)


def ready_knowledge():
    """返回已同步的 RagService；首次调用可能请求嵌入服务并更新本地向量库。"""
    from agent.tools.agent_tools import rag  # 登录、普通查询无需提前创建 Chroma 客户端。
    # global 表示这里更新进程共用的标记，而非同名的函数局部变量。
    global knowledge_ready
    # 一个后端进程启动后同步一次知识库；修改资料后重启后端即可重新检查。
    with knowledge_lock:
        if not knowledge_ready:
            # 根据磁盘资料和指纹处理新增、修改、删除的知识片段。
            rag.vector_store.load_document()
            # 若上一行抛出异常，执行不到这行，下次请求仍会尝试同步。
            knowledge_ready = True

    return rag  # 同一进程内多个会话复用，后续检索仍各自调用嵌入服务。


def forget_session(session_key):
    """按令牌摘要清除一次登录的内存对象，返回 None。会话已不存在时也可调用。"""
    # 退出登录时清掉这个会话的 Agent；下一次登录会获得新会话。
    with sessions_lock:
        # pop 的默认值为 None，已清理或从未聊天的会话退出也能成功。
        chat_sessions.pop(session_key, None)


def stream_chat(session_key, user_id, role, message, task_id=None, patient_id=None, mode="query"):
    """在已授权的任务范围内处理一条消息，逐个返回流式事件。

    输入：会话哈希、服务端身份、消息、患者和任务范围。mode 为 query 或 collect。
    例如 task_id="F001"、message="这个任务什么时候到期？"。
    输出：依次产生 {"type": "status", "content": "..."} 等事件。
    同一任务保留多轮记忆；身份或任务改变时重建 Agent，避免混入旧任务信息。
    患者原话先经过独立规则，命中后先持久保存通知并产生 risk 事件，再创建模型。
    患者任务每轮读取持久草稿；仅在模型完整结束后保存经校验的提议。
    同会话并发、模型失败或版本冲突返回 error，不提交正式记录或改变任务状态。
    """
    # 风险检查不占用模型会话锁；上一条消息还在回答时，新危险信息仍应先提醒。
    if role == "patient":
        risk = evaluate_risk(message=message)  # 不等模型提取、不等草稿完整，也不读旧草稿才能检查。
        # 只有命中升级规则才提前写通知，普通消息继续向下采集。
        if risk["level"] == "high":
            if task_id is None:
                # 首页不自动挑选一次历史随访发通知，立即给出可见的规则提示。
                yield {"type": "risk", "content": {"risk": risk, "notification_saved": False,
                                                    "notification_reason": "no_task"}}
            else:
                try:
                    saved = save_message_risk(user_id, role, task_id, risk)
                except Exception as error:
                    logger.error("风险通知保存失败，错误类型：%s", type(error).__name__)
                    yield {"type": "risk", "content": {"risk": risk, "notification_saved": False}}
                    yield {"type": "error", "content": "医生通知未能保存。请直接联系医生或及时就医，不要等待助手回复。"}
                    return
                yield {"type": "risk", "content": {"risk": risk, "notification_saved": saved}}

    # monotonic() 是只向前走的计时值，用于清理一天没有使用的内存会话。
    now = monotonic()
    with sessions_lock:
        # 复制会话条目后再遍历，循环内删除过期项不会改变正在遍历的对象。
        for key, item in list(chat_sessions.items()):
            # 一天未使用且没有执行中的请求，才可释放这份对话记忆。
            if now - item["last_used"] > 86400 and not item["lock"].locked():
                del chat_sessions[key]

        if session_key not in chat_sessions:
            # 新 token 首次聊天只建一个空槽位，等拿到本会话锁才初始化模型。
            chat_sessions[session_key] = {
                "agent": None, "scope": None, "lock": Lock(), "last_used": now,
            }
        # 从此处起 session 是本次登录专属的字典，与其他账号的 Agent 独立。
        session = chat_sessions[session_key]

    # 同一个会话上一轮没结束时，不让下一轮同时改写 Agent 记忆。
    # blocking=False 表示锁忙就提示重试，不让第二条消息等着覆盖刚才的草稿。
    if not session["lock"].acquire(blocking=False):
        yield {"type": "error", "content": "上一条消息还在处理中，请稍后再试。"}
        return

    try:
        # 草稿按任务保存，不依赖聊天记忆；换登录或重启后仍读取同一份业务数据。
        draft = None
        if mode == "collect":
            if role != "patient" or task_id is None:
                yield {"type": "error", "content": "请先进入自己的随访任务。"}
                return
            # 例如 {"revision":2,"editable":true,"data":{…}}，里面的版本用于末尾保存。
            draft = get_task_draft(user_id, role, task_id)
            if not draft["editable"]:
                yield {"type": "error", "content": "这次随访已提交，请使用「问助手」继续查询。"}
                return
        # 同一医生在同一档案中也可能失去某次任务权限。授权范围变化时清掉旧模型记忆。
        # 保存稳定编号而非病历正文，普通状态变化不打断正在追问的对话。
        access = []
        if task_id is None:
            for person in followup_store.list_patients(user_id, role, patient_id=patient_id):
                access.append(("patient", person["patient_id"], person["is_primary"]))
            for task in followup_store.list_followup_tasks(user_id, role, patient_id=patient_id):
                access.append(("task", task["task_id"], task["patient_id"], task["discharge_id"]))
        scope = (user_id, role, patient_id, task_id, mode, sorted(access))
        if session.get("scope") != scope:
            session["agent"] = None
            session["scope"] = scope
        if session["agent"] is None:
            # 提前显示准备状态，让用户看到首次加载知识库的等待原因。
            yield {"type": "status", "content": "正在准备 Agent 和医学知识库……"}
            # 同时传入身份与任务范围，后续工具查询由这些服务端值限定。
            session["agent"] = create_chat_agent(user_id, role, task_id=task_id, patient_id=patient_id, mode=mode)

        # 普通查询沿用原签名；患者任务把最新草稿传入每一轮推理。
        if draft is None:
            # 通用查询没有可写的任务草稿，沿用原有单参数调用。
            events = session["agent"].execute_stream(message)
        else:
            # 患者进入任务后的每轮推理都读取最新草稿，包括上轮手工编辑的值。
            events = session["agent"].execute_stream(message, draft=draft)
        proposal = None  # 暂存工具提议，模型失败时不能把半成品写入业务库。
        answer = None  # 最终回答等草稿保存成功后才发出。
        # 如果流里已经报告失败，后面即使有片段也不能算作提交成功。
        failed = False
        # 迭代时才真正运行生成器；每个事件只处理一次。
        for event in events:
            if event["type"] == "draft_proposal":
                proposal = event["content"]  # 内部事件不直接发给页面。
            elif event["type"] == "answer":
                # 暂存完整 answer，等候选草稿通过版本检查后再发给页面。
                answer = event
            else:
                if event["type"] == "error":
                    # 设置失败标记，确保后面不会把本轮半成品写入 task_drafts。
                    failed = True
                yield event  # status/reset/delta 继续实时显示。
        if failed or answer is None or not answer["content"]:
            session["agent"] = None  # 不沿用未完成推理，也不保存它的提议。
            if not failed:
                yield {"type": "error", "content": "未收到完整回答，草稿未更新，请重试。"}
            return
        # 字段没有实际变化时不写新草稿版本，避免患者白白遇到版本冲突。
        if draft is not None and proposal is not None and proposal != draft["data"]:
            try:
                # 写事务重新核对身份、归属、状态和旧版本；不会覆盖另一个页面的新修改。
                saved = save_task_draft(user_id, role, task_id, draft["revision"], proposal)
            except (PermissionError, RuntimeError, ValueError):
                session["agent"] = None  # 提议没存成，不能让后续记忆把它当成成功。
                yield {"type": "error", "content": "本轮草稿未保存：任务权限、状态或草稿内容已变化，请重新加载草稿后重试。"}
                return
            yield {"type": "draft", "content": saved}  # 页面只接收后端确认保存的新版本。
        # 页面收到 answer 后才把这一轮助手回答加入完成历史。
        yield answer
    except GeneratorExit:
        # 页面中途断开时也丢弃未完成的推理记忆，下次从干净状态继续。
        session["agent"] = None
        # 重新抛出关闭信号，使流正常结束并执行 finally。
        raise
    except Exception as error:
        # 工具参数错误可能含患者原话；这里只记类型，不输出原始参数或异常堆栈。
        logger.error("Agent 对话处理失败，错误类型：%s", type(error).__name__)
        # 失败的一轮可能留下未完成的工具调用，下一次使用新的 Agent。
        session["agent"] = None
        yield {"type": "error", "content": "本次回答失败，请重试；若持续失败，请检查后端模型配置和日志。"}
    finally:
        # 不论正常结束、失败还是中断，都更新最后活动时间。
        session["last_used"] = monotonic()
        # 必须释放会话锁，否则同一 token 的下一条消息会一直提示“正在处理”。
        session["lock"].release()
