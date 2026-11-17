"""FastAPI 入口：验证身份、接收参数，再调用 utils 和 Agent。"""
import hashlib
import json
import sqlite3

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from agent.tools.chat_session import forget_session, stream_chat
from utils import user_store
from utils.followup_store import get_followup_task, list_followup_tasks
from utils.draft_store import SaveDraftData, get_task_draft, save_task_draft
from utils.submission_store import SubmitDraftData, submit_task_draft
from utils.risk_store import list_doctor_notifications
from utils.risk_rules import evaluate_risk
from utils.logger_handler import logger

app = FastAPI(title="高血压随访助手")
bearer = HTTPBearer(auto_error=False)


class LoginData(BaseModel):
    # 这里只接收用户名和密码，不能通过请求指定 doctor 等角色。
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class ChatData(BaseModel):
    # 例如 {"message": "查一下本次安排", "task_id": "F001"}；user_id 不能由前端传入。
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    message: str = Field(min_length=1, max_length=4000)
    # 不传 task_id 或传 null 表示普通聊天；传任务编号时，后端还要检查任务归属。
    task_id: str | None = Field(default=None, min_length=1, max_length=64)


def current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    # Depends 表示每个需要登录的接口都会先执行这个身份检查。
    if credentials is None:
        raise HTTPException(status_code=401, detail="请先登录。")
    user = user_store.get_current_user(credentials.credentials)
    if user is None:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录。")

    # session_key 只在后端区分聊天会话，不作为用户可指定的参数。
    user["session_key"] = hashlib.sha256(credentials.credentials.encode()).hexdigest()
    return user


@app.exception_handler(sqlite3.Error)
async def database_error(request, error):
    # 不向页面暴露数据库路径和 SQL；具体原因只写后端日志。
    logger.error("医疗数据库操作失败：%s", error)
    return JSONResponse(status_code=503, content={"detail": "医疗数据库暂时无法使用，请检查后端日志。"})


@app.post("/auth/register", status_code=201)
def register(data: LoginData):
    # 患者自主注册；医生使用医疗库中已经分配好角色的账号。
    if not data.username.strip():
        raise HTTPException(status_code=422, detail="用户名不能为空。")
    if not user_store.register_user(data.username, data.password):
        raise HTTPException(status_code=409, detail="用户名已存在。")
    return {"message": "注册成功，请登录。"}


@app.post("/auth/login")
def login(data: LoginData):
    result = user_store.login_user(data.username, data.password)
    if result is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误，或账号未开通医患访问权限。")
    return result


@app.get("/me")
def me(user: dict = Depends(current_user)) -> dict:
    # 只返回页面展示需要的账号信息，不返回密码哈希或内部会话键。
    return {"user_id": user["user_id"], "username": user["username"], "role": user["role"]}


@app.post("/auth/logout")
def logout(user: dict = Depends(current_user), credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    user_store.logout_user(credentials.credentials)
    forget_session(user["session_key"])
    return {"message": "已退出登录。"}


@app.get("/tasks")
def tasks(user: dict = Depends(current_user)):
    # 查询和归属过滤写在 utils，路由只转交服务端已确认的身份。
    return list_followup_tasks(user["user_id"], user["role"])


@app.get("/tasks/{task_id}")
def task_detail(task_id: str, user: dict = Depends(current_user)):
    """输入任务编号，返回获授权的任务字典；找不到或无权查看均返回 404。

    例如 GET /tasks/F001 返回含 task_id、task_status 的字典。
    本接口只读数据库，不把任务改为进行中，也不创建 Agent。
    """
    # 查询时同时检查编号和登录身份，不能先取出任务再让前端判断权限。
    task = get_followup_task(user["user_id"], user["role"], task_id)
    # 两种情况使用相同响应，不泄露其他账号的任务是否存在。
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在或无权访问。")
    # 只返回查询层明确选择的字段，不包含未审核报告或原始 JSON。
    return task


@app.get("/tasks/{task_id}/draft")
def task_draft(task_id: str, user: dict = Depends(current_user)):
    """读取当前患者的采集草稿；无草稿返回版本 0，越权返回 404，不调用模型。"""
    try:
        return get_task_draft(user["user_id"], user["role"], task_id)
    except PermissionError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.put("/tasks/{task_id}/draft")
def update_task_draft(task_id: str, data: SaveDraftData, user: dict = Depends(current_user)):
    """输入完整表单和读取时版本，保存草稿并返回新版本；不正式提交随访。

    例如 {expected_revision: 0, data: {symptoms: "没有不舒服"}}。
    字段错误返回 422，过期版本或任务不可编辑返回 409，越权返回 404。
    """
    risk = evaluate_risk(data=data.data.model_dump())  # 先纯计算，以便保存失败时仍可展示明确的风险。
    try:
        return save_task_draft(user["user_id"], user["role"], task_id,
                               data.expected_revision, data.data.model_dump())
    except PermissionError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        return draft_error_response(409, str(error), risk)
    except ValueError as error:
        return draft_error_response(422, str(error), risk)
    except sqlite3.Error as error:
        logger.error("医疗草稿保存失败，错误类型：%s", type(error).__name__)  # 不记录患者原话或表单。
        return draft_error_response(503, "医疗数据库暂时无法使用，请检查后端日志。", risk)


def draft_error_response(status_code, detail, risk):
    """输入失败状态、提示与纯规则结果，输出 JSON 错误；无数据库或网络操作。

    高风险时额外返回 {risk: {...}, notification_saved: false}，让页面独立保留就医提醒。
    这不表示草稿保存成功，也不声称失败事务里的通知已经发出；无权访问不走此函数。
    """
    content = {"detail": detail}  # 保持既有页面和接口的错误字段。
    if risk["level"] == "high":
        content["risk"] = risk
        content["notification_saved"] = False  # 本次失败不能保证已保存通知，不能误导患者等待。
    return JSONResponse(status_code=status_code, content=content)


@app.post("/tasks/{task_id}/submit")
def submit_followup(task_id: str, data: SubmitDraftData, user: dict = Depends(current_user)):
    """患者明确确认后提交已保存草稿；返回记录编号与风险结果，不调用 Agent。

    输入例：{expected_revision: 2, confirmed: true}。缺项为 422，版本/状态冲突为 409。
    身份来自登录凭证；不存在或越权为 404。数据库写入和事务放在 utils 中。
    """
    try:
        return submit_task_draft(user["user_id"], user["role"], task_id, data.expected_revision, data.confirmed)
    except PermissionError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/notifications")
def doctor_notifications(user: dict = Depends(current_user)):
    """返回当前负责医生收到的风险通知列表；只读，不实现通知处理或报告审核。"""
    try:
        return list_doctor_notifications(user["user_id"], user["role"])
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error


@app.post("/chat")
def chat(data: ChatData, user: dict = Depends(current_user)):
    """输入消息和可选任务编号，返回聊天事件流；任务无权访问时先返回 404。

    例如 {"message": "查看本次安排", "task_id": "F001"} 只进入该任务的聊天。
    每条消息重新核对归属；会话层在调用模型前执行规则并按需保存风险通知。
    聊天不能修改任务状态或提交正式随访记录。
    """
    # 已在页面进入任务也不能跳过检查，任务可能已被重新分配给其他医生。
    if data.task_id is not None:
        task = get_followup_task(user["user_id"], user["role"], data.task_id)
        # 在开始流式回答、创建 Agent 之前拒绝越权请求。
        if task is None:
            raise HTTPException(status_code=404, detail="任务不存在或无权访问。")

    def events():
        """把事件转为逐行 JSON；草稿校验保存由会话层调用 utils 完成。"""
        # 一行一个 JSON 事件，前端收到一行便可更新工具状态或回答。
        for event in stream_chat(
            user["session_key"], user["user_id"], user["role"], data.message,
            task_id=data.task_id,
        ):
            # 例如 {"type": "status", "content": "正在查询"} 后面加一个换行。
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")
