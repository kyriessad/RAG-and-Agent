"""FastAPI 入口：验证身份、接收参数，再调用 utils 和 Agent。"""
# 给内存会话生成摘要键，退出时能精确找到该登录对应的 Agent。
import hashlib
# 聊天事件需要逐条编码成 JSON，不能等所有事件生成后才一次发送。
import json
# 捕获 SQLite 的表、连接或事务异常，统一转成前端可处理的 HTTP 状态。
import sqlite3
from contextlib import asynccontextmanager
from datetime import date, timedelta, timezone
from typing import Literal
from uuid import UUID

# 随访的定时扫描运行在后端，浏览器关闭后仍能按时开放任务。
from apscheduler.schedulers.background import BackgroundScheduler
# Depends 会在路由执行前调用鉴权函数，HTTPException 会提前结束本次请求。
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
# 普通失败用 JSONResponse；聊天用 StreamingResponse 连续发送多行。
from fastapi.responses import JSONResponse, StreamingResponse
# HTTPBearer 读取 Authorization 请求头，例如 Bearer 后跟登录得到的随机串。
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
# BaseModel 描述请求体，Field 在进入业务函数前检查长度等条件。
from pydantic import BaseModel, ConfigDict, Field

# stream_chat 返回事件生成器；forget_session 清理当前 token 对应的内存记忆。
from agent.tools.chat_session import forget_session, stream_chat
# 账号查询和令牌撤销集中在 user_store，页面无需了解密码存储格式。
from utils import followup_store, schedule_store, user_store
from utils import report_store  # 独立处理报告的授权查询、版本检查和事务。
# 工具查询和 HTTP 查询复用相同的任务权限条件。
from utils.followup_store import get_followup_task, list_followup_tasks
# SaveDraftData 接收表单和旧版本，保存时仍允许暂时缺项。
from utils.draft_store import SaveDraftData, get_task_draft, save_task_draft
# 正式提交单独检查患者确认，成功后返回不可覆盖的记录编号。
from utils.submission_store import SubmitDraftData, submit_task_draft
# 医生工作台从 notifications 读取已经写入的风险通知。
from utils.risk_store import handle_doctor_notification, list_doctor_notifications, migrate_risk_notifications
# 规则判断可以独立执行，即使后面的写库或模型调用失败也能提醒。
from utils.risk_rules import evaluate_risk
from utils.logger_handler import logger

@asynccontextmanager
async def lifespan(application):
    """启动时升级排期字段、补扫到期事项，每分钟继续扫描；数据库错误向启动方报告。"""
    user_store.init_db()
    migrate_risk_notifications()
    schedule_store.migrate_schedule()  # 首次启用时标记已有出院记录，后续启动保留原标记。
    schedule_store.scan_schedule()  # 停机期间到期的任务从本次实际开放起算 48 小时。
    scheduler = BackgroundScheduler(timezone=timezone(timedelta(hours=8)))
    # 每个整分钟扫描。同进程最多运行一份，数据库事务和唯一键负责跨进程去重。
    scheduler.add_job(schedule_store.scan_schedule, "cron", second=0, coalesce=True, max_instances=1)
    scheduler.start()
    try:
        yield
    finally:
        scheduler.shutdown(wait=True)  # 让正在提交的事务完成，再结束后台线程。


# FastAPI 启动时收集下方路由并运行一次表结构检查。
app = FastAPI(title="高血压随访助手", lifespan=lifespan)
# 缺少凭证时交给 current_user 返回统一的 401 提示。
bearer = HTTPBearer(auto_error=False)


# 装饰器注册错误处理函数；下方所有报告接口共享一套错误回复格式。
@app.exception_handler(report_store.ReportError)
async def report_error(request, error):
    """把业务异常转换为统一的 HTTP 回复，如 409 表示需要刷新；不记录病历正文。"""
    return JSONResponse(status_code=error.status, content={"detail": str(error)})



class AccountData(BaseModel):
    """姓名及密码请求。注册接口禁止指定角色，已有客户端继续使用 username 传姓名。"""
    model_config = ConfigDict(extra="forbid")
    # 长度检查通过后，注册入口还会排除全空格用户名。
    username: str = Field(min_length=1, max_length=64)
    # 密码只作为本次校验输入，成功响应不返回这个字段。
    password: str = Field(min_length=1, max_length=128)


class LoginData(AccountData):
    """登录时可附加页面入口，实际角色取自账号。旧客户端可省略入口。"""
    role: Literal["patient", "doctor"] | None = None


class RegistrationPatientProfile(BaseModel):
    """注册表单中的患者档案；例如 age=56、gender='female'、doctor_id='D002'。"""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    # 与现库年龄字段一致。SQLite 的 INTEGER 能保存的最大值是有符号 64 位整数上限。
    age: int = Field(strict=True, ge=0, le=9223372036854775807)
    gender: Literal["female", "male", "unknown"]
    doctor_id: str = Field(min_length=1, max_length=64)  # 主管医生由患者在简要名录中选择。
    phone: str | None = Field(default=None, max_length=32)


class PatientRegistrationData(AccountData):
    """网页传完整档案，已有客户端仍可只传账号密码。"""
    profile: RegistrationPatientProfile | None = None


class DoctorRegistrationData(AccountData):
    """本机医生直接注册时所需科室和选填的联系资料；角色由医生注册路由固定。"""
    department: str = Field(min_length=1, max_length=64)
    title: str | None = Field(default=None, max_length=64)
    phone: str | None = Field(default=None, max_length=32)


class ChatData(BaseModel):
    # 例如 {"message": "查一下本次安排", "task_id": "F001"}；user_id 不能由前端传入。
    # 字符串先清理首尾空白，全空格的消息会在长度检查时被拒绝。
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    message: str = Field(min_length=1, max_length=4000)
    # 不传 task_id 或传 null 表示普通聊天；传任务编号时，后端还要检查任务归属。
    task_id: str | None = Field(default=None, min_length=1, max_length=64)
    # 档案页选中的患者还须在服务端核实；查询为默认，只有主动采集才提供草稿工具。
    patient_id: str | None = Field(default=None, min_length=1, max_length=64)
    mode: Literal["query", "collect"] = "query"


def current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    """读取请求头中的令牌，返回实时身份和会话键；凭证缺失或过期返回 401，只读库。"""
    # Depends 表示每个需要登录的接口都会先执行这个身份检查。
    # 还没有 Authorization 请求头，不能执行后面的医疗查询。
    if credentials is None:
        raise HTTPException(status_code=401, detail="请先登录。")
    # credentials.credentials 是令牌正文；查询 sessions 时由存储层计算哈希。
    user = user_store.get_current_user(credentials.credentials)
    # 找不到令牌、登录过期或角色失效都需要重新登录。
    if user is None:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录。")

    # session_key 只在后端区分聊天会话，不作为用户可指定的参数。
    user["session_key"] = hashlib.sha256(credentials.credentials.encode()).hexdigest()
    # 后续路由拿到 user 字典，通过 user_id 和 role 在数据库重新限制业务范围。
    return user


# 注册底层数据库故障处理，避免前端把读取失败显示成“没有任务”。
@app.exception_handler(sqlite3.Error)
async def database_error(request, error):
    """记录 SQLite 异常并返回 503 JSON，供页面提示稍后重试；不改业务状态。"""
    # 不向页面暴露数据库路径和 SQL；具体原因只写后端日志。
    logger.error("医疗数据库操作失败：%s", error)
    return JSONResponse(status_code=503, content={"detail": "医疗数据库暂时无法使用，请检查后端日志。"})


def register_account(data: AccountData, role: str, profile: dict | None):
    """转交角色专用入口的数据并写库，返回注册回执。重复或医生失效返回 409，无效字段返回 422。"""
    try:
        created = user_store.register_user(data.username, data.password, role=role, profile=profile)
    except LookupError as error:  # 名录与提交间发生变化，提示重新选择。
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:  # 姓名和科室经过清理后也必须是非空文本。
        raise HTTPException(status_code=422, detail=str(error)) from error
    if not created:
        raise HTTPException(status_code=409, detail="该姓名已注册，请登录或核对姓名。")
    return {"message": "注册成功，请登录。"}


# 注册前可查看主管医生选项，只公开名片摘要，不需要已有登录凭证。
@app.get("/auth/doctors")
def registration_doctors():
    """无入参，只读可选医生；即便名录为空，也返回 {"doctors": []}。数据库错误返回 503。"""
    return {"doctors": user_store.list_registration_doctors()}


@app.post("/auth/register", status_code=201)
def register(data: PatientRegistrationData):
    """接收患者姓名、密码和可选档案，写入账号并返回成功回执；冲突和校验错误交给共同处理。"""
    profile = None
    if data.profile is not None:
        profile = data.profile.model_dump()  # 档案的字段、类型与必填项已经校验。
    return register_account(data, "patient", profile)


@app.post("/auth/register/doctor", status_code=201)
def register_doctor(data: DoctorRegistrationData):
    """接收医生姓名、密码、科室和选填资料，同事务创建账号与医生档案，重复姓名返回 409。"""
    profile = data.model_dump(exclude={"username", "password"})  # 档案中只传科室、职称和电话。
    return register_account(data, "doctor", profile)


# POST 表示凭证放在请求体，避免密码出现在 URL 中。
@app.post("/auth/login")
def login(data: LoginData):
    """核对账号密码，记录新会话并返回令牌及身份；登录失败返回 401。"""
    # 存储层校验密码并插入一条有过期时间的登录会话。
    result = user_store.login_user(data.username, data.password, data.role)
    # 统一登录失败提示，避免通过响应区分哪个用户名存在。
    if result is None:
        raise HTTPException(status_code=401, detail="请核对姓名、密码和选择的登录身份。")
    # 成功形状示例 {"access_token":"…","user":{"user_id":"U1","role":"patient",…}}。
    return result


# 每次 Streamlit 重跑都可 GET /me 重新核对身份。
@app.get("/me")
def me(user: dict = Depends(current_user)) -> dict:
    """返回当前用户的三个公开身份字段，供页面核对登录状态；本接口不写库。"""
    # 只返回页面展示需要的账号信息，不返回密码哈希或内部会话键。
    return {"user_id": user["user_id"], "username": user["username"], "role": user["role"]}


# logout 同时需要身份和原始令牌，才能清理数据库与聊天内存。
@app.post("/auth/logout")
def logout(user: dict = Depends(current_user), credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    """撤销当前登录令牌并删除对应的内存会话；成功返回退出提示。"""
    # 删除对应 sessions 行后，此 token 下次访问就会收到 401。
    user_store.logout_user(credentials.credentials)
    # 使用令牌摘要定位这次登录的 Agent，不影响同账号的其他设备。
    forget_session(user["session_key"])
    return {"message": "已退出登录。"}


@app.get("/patients")
def patients(user: dict = Depends(current_user)):
    """按登录身份返回名册；患者只见自己，无档案时为空列表。仅查询数据库。"""
    return followup_store.list_patients(user["user_id"], user["role"])


@app.get("/patients/{patient_id}")
def patient_detail(patient_id: str, user: dict = Depends(current_user)):
    """按患者编号只读授权档案、对方的联系方式、出院和随访；无关联与不存在均返回 404。"""
    detail = followup_store.get_patient_detail(user["user_id"], user["role"], patient_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="患者不存在或无权访问。")
    return detail


class PatientProfileUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    phone: str | None = Field(default=None, max_length=32)
    medical_history: str | None = Field(default=None, max_length=2000)


@app.patch("/patients/{patient_id}")
def update_patient_profile(patient_id: str, data: PatientProfileUpdate, user: dict = Depends(current_user)):
    """患者可以修改本人的联系电话与既往病史，未提交的字段保留原值。"""
    if user["role"] != "patient":
        raise HTTPException(status_code=403, detail="只有患者本人可以修改个人资料。")
    changes = data.model_dump(exclude_unset=True)
    if not changes:
        raise HTTPException(status_code=422, detail="请选择要修改的资料。")
    if not followup_store.update_own_profile(user["user_id"], patient_id, changes):
        raise HTTPException(status_code=404, detail="患者不存在或无权访问。")
    return {"message": "个人资料已保存。"}


class DischargeDetails(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    discharge_instructions: str | None = Field(default=None, max_length=4000)
    discharge_medications: str | None = Field(default=None, max_length=4000)


class DischargeRegistration(DischargeDetails):
    """保存主管医生填写的出院原文，旧表单仍可以只提供日期和诊断。"""
    request_id: UUID  # 同一次表单提交使用固定编号，断线可原样重试。
    discharge_date: date  # 例如 "2026-11-18"，不要求浏览器自行计算 D7。
    discharge_diagnosis: str = Field(min_length=1, max_length=500)


# FastAPI 把 JSON 校验为 DischargeRegistration 后再调用存储层，在一个事务内完成登记。
@app.post("/patients/{patient_id}/discharges", status_code=201)
def register_discharge(patient_id: str, data: DischargeRegistration, user: dict = Depends(current_user)):
    """主管医生登记出院并取得三轮随访安排。scheduled_at 保留首轮开放时间。"""
    if user["role"] != "doctor":
        raise HTTPException(status_code=403, detail="只有主管医生可以登记出院。")
    try:
        receipt = schedule_store.register_discharge(
            user["user_id"], patient_id, data.request_id, data.discharge_date, data.discharge_diagnosis,
            data.discharge_instructions or None, data.discharge_medications or None,
        )
    except schedule_store.DischargeConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if receipt is None:
        raise HTTPException(status_code=404, detail="患者不存在或无权登记出院。")
    return receipt


@app.patch("/patients/{patient_id}/discharges/{discharge_id}")
def supplement_discharge(patient_id: str, discharge_id: str, data: DischargeDetails, user: dict = Depends(current_user)):
    if user["role"] != "doctor":
        raise HTTPException(status_code=403, detail="只有主管医生可以补充出院资料。")
    changes = data.model_dump(exclude_unset=True)
    if not changes or not all(changes.values()):
        raise HTTPException(status_code=422, detail="请填写需要补充的医嘱或用药。")
    try:
        found = schedule_store.supplement_discharge(user["user_id"], patient_id, discharge_id, changes)
    except schedule_store.DischargeConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    if not found:
        raise HTTPException(status_code=404, detail="出院记录不存在或无权补充。")
    return {"message": "出院资料已补充。"}


# GET /tasks 没有身份参数，即使医生查任务也只用服务器确认的身份。
@app.get("/tasks")
def tasks(user: dict = Depends(current_user)):
    """读取登录患者或负责医生可以进入的任务；没有关联任务时返回空列表。"""
    # 查询和归属过滤写在 utils，路由只转交服务端已确认的身份。
    return list_followup_tasks(user["user_id"], user["role"])


@app.get("/tasks/{task_id}")
def task_detail(task_id: str, user: dict = Depends(current_user)):
    """输入任务编号，返回获授权的任务字典；找不到或无权查看均返回 404。

    例如患者 GET /tasks/F001 返回含 task_id、task_status 和本次负责人 doctor_phone 的字典。
    本接口只读数据库，不把任务改为进行中，也不创建 Agent。
    """
    # 查询时同时检查编号和登录身份，不能先取出任务再让前端判断权限。
    task = get_followup_task(user["user_id"], user["role"], task_id, include_contact=True)
    # 两种情况使用相同响应，不泄露其他账号的任务是否存在。
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在或无权访问。")
    # 只返回查询层明确选择的字段，不包含未审核报告或原始 JSON。
    return task


@app.get("/tasks/{task_id}/draft")
def task_draft(task_id: str, user: dict = Depends(current_user)):
    """读取当前患者的采集草稿；无草稿返回版本 0，越权返回 404，不调用模型。"""
    # 获取采集草稿只读 task_drafts，未保存的草稿以版本 0 返回。
    try:
        return get_task_draft(user["user_id"], user["role"], task_id)
    # 未获授权和编号不存在使用相同的响应，不能借编号猜测任务归属。
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
        # data.data 是字段模型，model_dump() 将其转换为草稿存储函数接受的字典。
        return save_task_draft(user["user_id"], user["role"], task_id,
                               data.expected_revision, data.data.model_dump())
    except PermissionError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    # 草稿版本或状态冲突时不覆盖他人在另一页面刚保存的值。
    except RuntimeError as error:
        return draft_error_response(409, str(error), risk)
    # 输入类型正确也可能有业务冲突，例如“尚未测量”却填了读数。
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
    # 失败时仅把已确定的高风险结果附在错误响应里，供页面单独显示。
    if risk["level"] == "high":
        content["risk"] = risk
        content["notification_saved"] = False  # 本次失败不能保证已保存通知，不能误导患者等待。
    # status_code 保留 409、422 或 503，避免前端将提醒误当保存回执。
    return JSONResponse(status_code=status_code, content=content)


@app.post("/tasks/{task_id}/submit")
def submit_followup(task_id: str, data: SubmitDraftData, background_tasks: BackgroundTasks,
                    user: dict = Depends(current_user)):
    """患者确认后写正式记录，先回传成功，再在后台按记录编号生成报告。

    输入例：{expected_revision: 2, confirmed: true}。缺项为 422，版本/状态冲突为 409。
    身份来自登录凭证；不存在或越权为 404。数据库写入和事务放在 utils 中。
    """
    try:
        # confirmed 必须为 True，expected_revision 必须匹配服务器所存草稿。
        result = submit_task_draft(user["user_id"], user["role"], task_id, data.expected_revision, data.confirmed)
    except PermissionError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    # 已提交任务的不同版本不能再次提交，应刷新查看最新状态。
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    # 缺项或未勾选确认由存储层报告为 ValueError，页面会显示可修改的问题。
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    # 重复点击只取原回执；生成失败的任务仍可由负责医生从待办中重试。
    if not result["already_submitted"]:
        identity = {"user_id": user["user_id"], "role": user["role"]}
        background_tasks.add_task(generate_submitted_report, identity, task_id, result["record_id"])
    return result


def generate_submitted_report(user, task_id, record_id):
    """在 HTTP 回执发出后生成报告。读已提交记录、请求模型并写报告；失败保留待办。"""
    try:
        report_store.generate_report(user, task_id, record_id)
    except Exception as error:
        # 此时患者的提交事务已完成，医生可重试。日志记录错误类型，不带病历正文。
        logger.warning("后台生成随访报告未完成，错误类型：%s", type(error).__name__)


@app.get("/notifications")
def doctor_notifications(user: dict = Depends(current_user)):
    """返回当前负责医生收到的风险通知列表。"""
    try:
        # SQL 同时检查接收人和任务当前负责医生，改派后旧医生无法读取。
        return list_doctor_notifications(user["user_id"], user["role"])
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error


@app.post("/notifications/{notice_id}/handle")
def handle_risk_notification(notice_id: int, user: dict = Depends(current_user)):
    try:
        notice = handle_doctor_notification(user["user_id"], user["role"], notice_id)
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    if notice is None:
        raise HTTPException(status_code=404, detail="风险通知不存在或无权处理。")
    return notice


@app.get("/notifications/schedule")
def schedule_notifications(user: dict = Depends(current_user)):
    """通过登录身份读取仍需本人的随访提醒，无待办时返回 []；只读，不更改通知。"""
    try:
        return schedule_store.list_schedule_notifications(user["user_id"], user["role"])
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error


@app.get("/tasks/{task_id}/report")
def task_report(task_id: str, user: dict = Depends(current_user)):
    """读取本次报告视图。患者只能读到最终发布正文，医生可查看原始记录和引用。"""
    return report_store.report_view(user, task_id)  # 同一个 URL，返回哪些字段由服务端决定。


@app.post("/tasks/{task_id}/report/generate")
def generate_report(task_id: str, data: report_store.GenerateRequest, user: dict = Depends(current_user)):
    """接收 record_id，成功仅返回编号回执；模型或检索失败返回可重试的 503。"""
    # 生成成功的回执只有 record_id，患者由此知道记录已受理。
    return report_store.generate_report(user, task_id, data.record_id)


@app.post("/tasks/{task_id}/report/review")
def review_report(task_id: str, data: report_store.ReviewRequest, user: dict = Depends(current_user)):
    """接收医生选择和所见版本，返回新版本；存储层在事务内核对归属、记录和状态。"""
    # 例如 action=publish 要求该版本已经审核通过；页面不能自行将状态改为完成。
    return report_store.review_report(user, task_id, data)


@app.post("/chat")
def chat(data: ChatData, user: dict = Depends(current_user)):
    """输入消息和可选任务编号，返回聊天事件流；任务无权访问时先返回 404。

    例如 {"message": "查看本次安排", "task_id": "F001"} 只进入该任务的聊天。
    每条消息重新核对归属；会话层在调用模型前执行规则并按需保存风险通知。
    聊天不能修改任务状态或提交正式随访记录。
    """
    # 已在页面进入任务也不能跳过检查，任务可能已被重新分配给其他医生。
    patient_id = data.patient_id  # 可能在档案页选定，也可能由被授权的任务补全。
    if data.task_id is not None:
        # 在建立事件流之前验证任务，权限失效可以用 HTTP 404 明确告知页面。
        task = get_followup_task(user["user_id"], user["role"], data.task_id)
        # 在开始流式回答、创建 Agent 之前拒绝越权请求。
        if task is None:
            raise HTTPException(status_code=404, detail="任务不存在或无权访问。")
        if data.patient_id is not None and task["patient_id"] != data.patient_id:
            raise HTTPException(status_code=404, detail="任务不存在或无权访问。")
        # 即使页面只传任务号，各工具也绑定这次实际查到的患者，改派后不会沿用旧患者记忆。
        patient_id = task["patient_id"]
    elif data.patient_id is not None:
        # 档案页没有选单次任务时，也必须验证患者关系。
        if not followup_store.list_patients(user["user_id"], user["role"], patient_id=data.patient_id):
            raise HTTPException(status_code=404, detail="患者不存在或无权访问。")
    if data.mode == "collect" and (user["role"] != "patient" or data.task_id is None):
        raise HTTPException(status_code=422, detail="采集模式需要先进入自己的随访任务。")

    def events():
        """把事件转为逐行 JSON；草稿校验保存由会话层调用 utils 完成。"""
        # 一行一个 JSON 事件，前端收到一行便可更新工具状态或回答。
        for event in stream_chat(
            user["session_key"], user["user_id"], user["role"], data.message,
            task_id=data.task_id, patient_id=patient_id, mode=data.mode,
        ):
            # 例如 {"type": "status", "content": "正在查询"} 后面加一个换行。
            yield json.dumps(event, ensure_ascii=False) + "\n"

    # 告诉客户端按行读取 NDJSON；后续模型错误由流中的 error 事件表示。
    return StreamingResponse(events(), media_type="application/x-ndjson")
