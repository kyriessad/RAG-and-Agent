# 令牌和密码使用不同算法；数据库里只保存它们的校验信息。
import hashlib
# compare_digest 使用恒定时间比较同类型摘要。
import hmac
# secrets 生成密码盐和登录令牌，两者都需要可靠随机性。
import secrets
# SQLite 连接负责参数化查询、事务以及外键校验。
import sqlite3
from contextlib import closing  # 注册事务结束后释放连接，包括校验失败的分支。
# 统一按 UTC 保存过期时刻；timedelta 表示有效时长。
from datetime import datetime, timedelta, timezone
# Path 把含中文或空格的本地路径转换成 SQLite 可读取的 URI。
from pathlib import Path
# uuid4 只用于生成账号编号，不承担登录鉴权。
from uuid import uuid4

from utils.path_tool import get_abs_path


# 所有账号和登录凭证都放在同一个医疗数据库里。
# DB_PATH 是字符串，例如 C:/项目/data/medical.db；测试时可替换为临时库路径。
DB_PATH = get_abs_path("data/medical.db")


def get_connection():
    """无参数，返回已开启外键校验的 SQLite 连接；打不开现库时抛 sqlite3.Error。"""
    # mode=rw 表示只打开已有数据库，路径写错时直接报错，不创建空数据库。
    database_uri = Path(DB_PATH).resolve().as_uri() + "?mode=rw"
    # uri=True 启用上面 mode=rw 的约束；遇到其他写事务时最多等待 10 秒。
    conn = sqlite3.connect(database_uri, uri=True, timeout=10)

    # conn 是数据库连接；Row 让结果支持 row["user_id"] 这样的列名读取。
    conn.row_factory = sqlite3.Row
    # SQLite 需要为每条新连接开启外键检查，才能拒绝没有对应任务的记录。
    conn.execute("PRAGMA foreign_keys = ON")
    # 调用者负责关闭这条连接；多个 HTTP 请求不共享同一个游标。
    return conn


# 哈希密码
def hash_password(password: str) -> str:
    """输入密码文本，输出带随机盐的 PBKDF2 存储串；只计算，不写库或发网络请求。"""
    # 每次新哈希都使用新的 16 字节随机盐，即使密码相同也得到不同存储串。
    salt = secrets.token_hex(16)  # 十六进制字符串
    digest = hashlib.pbkdf2_hmac(
        "sha256",  # 哈希算法
        password.encode("utf-8"),  # 用户密码
        bytes.fromhex(salt),  # 盐（需要转回字节）
        600_000,  # 迭代次数
    ).hex()  # 十六进制字符串

    # 存储格式带算法、迭代数和盐；以后验证无需保存原始密码。
    return f"pbkdf2_sha256$600000${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    """输入本次密码和数据库存储串，返回是否匹配；常见格式错误返回 False，不改数据。"""
    # 数据损坏时返回验证失败，不把异常的密码格式当成正常密码。
    if not isinstance(stored, str):
        return False

    try:
        # 例如拆分后得到算法名、字符串“600000”、盐和预期摘要。
        method, iterations, salt, expected = stored.split("$")

        # 只按本项目支持的格式验证，未知算法视为无效记录。
        if method != "pbkdf2_sha256":
            return False
        # 重新计算哈希
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            # 验证时复用原来的盐，绝不能在这里重新生成随机盐。
            bytes.fromhex(salt),
            # 存储串里的迭代数是文本，计算哈希时要转为整数。
            int(iterations),
        ).hex()
        # 恒定时间比较函数，无论两个字符串差异在哪里，它花费的时间都是固定的
        return hmac.compare_digest(actual, expected)

    # 格式损坏时返回 False，让上层按一次失败登录处理。
    except (ValueError, TypeError):
        return False


# 保留原来的函数名，启动时只检查现有表，不创建另一套账号表。
def init_db():
    """只读检查 users、sessions 所需列；正常返回 None，缺库或缺列抛 SQLite 异常。"""
    # 使用现有库检查结构；路径错误会报错，避免另建空库掩盖问题。
    conn = get_connection()
    try:
        # LIMIT 0 不读取账号内容，只确认登录所需的表和列确实存在。
        conn.execute("SELECT user_id, name, password_hash, role FROM users LIMIT 0")
        # 同样核对登录令牌表，注册、登录和鉴权都连接这一库。
        conn.execute("SELECT token_hash, user_id, expires_at FROM sessions LIMIT 0")
    finally:
        # finally 确保即使缺少表列也会释放连接。
        conn.close()


def list_registration_doctors() -> list[dict]:
    """无参数，只读公开的医生选项，返回医生编号、姓名、科室和职称；无人可选时返回空列表。"""
    # 登录前的名录只展示选择主管医生需要的资料；联系方式在授权档案中获取。
    with closing(get_connection()) as conn:
        rows = conn.execute(
            """SELECT d.doctor_id, d.name, d.department, d.title
               FROM doctors d JOIN users u ON u.user_id=d.user_id
               WHERE u.role='doctor' ORDER BY d.department, d.name, d.doctor_id"""
        )
        doctors = []
        for row in rows:
            doctors.append(dict(row))
        return doctors


def register_user(username: str, password: str, *, role="patient", profile=None) -> bool:
    """将账号与注册表单中的档案一起写库。旧调用只传姓名和密码，仍可仅建立患者账号。

    role 由路由固定。患者 profile 含 age、gender、doctor_id；医生 profile 必须有 department。
    成功返回 True，重名返回 False，主管医生已失效抛 LookupError，必填项为空抛 ValueError。
    """
    # 姓名在账号和档案中始终一致；密码保留用户实际输入的空格。
    username = username.strip()
    if not username or not password:
        raise ValueError("姓名和密码不能为空。")
    if role not in ("doctor", "patient"):
        raise ValueError("请选择患者或医生注册。")
    if role == "doctor":
        department = ""
        if profile is not None:
            department = (profile.get("department") or "").strip()  # 全空格的科室也不能用于建档。
        if not department:
            raise ValueError("请填写医生所在科室。")
    # 编号由服务端分配；耗时的密码哈希在获取写锁前完成。
    user_id = "U" + uuid4().hex
    password_hash = hash_password(password)
    with closing(get_connection()) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")  # 锁内校验医生，防止点提交时主管医生已被改派角色或删除。
        created = conn.execute(
            """INSERT INTO users(user_id,name,password_hash,role) VALUES(?,?,?,?)
               ON CONFLICT(name) DO NOTHING""",
            (user_id, username, password_hash, role),
        )
        if created.rowcount == 0:  # 两个请求同时注册同名时，后到的请求不能覆盖原账号。
            return False
        if profile is None:
            return True  # 兼容只传账号密码的旧患者客户端。
        phone = (profile.get("phone") or "").strip() or None  # 联系卡将未登记的号码显示为空。
        if role == "doctor":
            title = (profile.get("title") or "").strip() or None
            conn.execute(
                """INSERT INTO doctors(doctor_id,user_id,name,department,title,phone)
                   VALUES(?,?,?,?,?,?)""",
                ("D" + uuid4().hex, user_id, username, department, title, phone),
            )
        else:
            # 下拉框保存的是医生档案编号；提交时核对当前账号仍有医生身份。
            doctor = conn.execute(
                """SELECT d.doctor_id FROM doctors d JOIN users u ON u.user_id=d.user_id
                   WHERE d.doctor_id=? AND u.role='doctor'""", (profile["doctor_id"],),
            ).fetchone()
            if doctor is None:
                raise LookupError("所选主管医生已不可选，请重新选择。")  # 撤销本次刚创建的账号。
            conn.execute(
                """INSERT INTO patients(patient_id,user_id,name,age,gender,doctor_id,phone)
                   VALUES(?,?,?,?,?,?,?)""",
                ("P" + uuid4().hex, user_id, username, profile["age"], profile["gender"],
                 doctor["doctor_id"], phone),
            )
        return True  # 离开事务块后，账号与档案同时可见。


def login_user(username: str, password: str, role: str | None = None) -> dict | None:
    """按姓名和密码登录，若指定入口则同时核对角色。成功时写会话并返回凭证，失败返回 None。"""
    conn = get_connection()
    try:
        # fetchone() 取一行或 None；不会把 users 全表放进内存。
        row = conn.execute(
            "SELECT user_id, name AS username, password_hash, role FROM users WHERE name = ?",
            # 单个 SQL 参数也需要写成元组，末尾逗号有实际作用。
            (username.strip(),),
        ).fetchone()

        # 角色直接读取数据库，不相信页面传来的角色名称。
        if row is None or row["role"] not in ("patient", "doctor"):
            return None
        # 入口只用来核对已有身份，角色不匹配时不签发登录凭证。
        if role is not None and row["role"] != role:
            return None
        # 用数据库中的算法、盐和迭代数验证用户刚提交的密码。
        if not verify_password(password, row["password_hash"]):
            return None

        # token 是发给前端的随机登录凭证；数据库只保存它的 SHA256 哈希。
        token = secrets.token_urlsafe(32)
        # 只保存 token 的 SHA256 十六进制摘要，响应中才带原始令牌。
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        # 有效期是从本次登录起的 24 小时，带 UTC 时区。
        expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
        conn.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            # isoformat() 生成可还原时区的文本，供 get_current_user 比较。
            (token_hash, row["user_id"], expires_at.isoformat()),
        )
        # 成功写入会话后才返回令牌，避免发出数据库无法识别的凭证。
        conn.commit()

        # user 是普通字典，不包含密码哈希，可安全交给页面显示用户名和角色。
        user = {"user_id": row["user_id"], "username": row["username"], "role": row["role"]}
        # 页面只需保存 access_token 和 user，不需要保存密码。
        return {"access_token": token, "user": user}
    finally:
        conn.close()


def get_current_user(token: str) -> dict | None:
    """根据登录凭证查询当前身份；凭证不存在或过期时返回 None。"""
    # 空令牌直接拒绝，避免执行无意义的数据库查询。
    if not token:
        return None

    # 相同令牌在登录和鉴权时得到相同摘要，用它匹配 sessions。
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    conn = get_connection()
    try:
        # 每次请求重新关联 users，用户角色以数据库中的当前值为准。
        row = conn.execute(
            """
            SELECT u.user_id, u.name AS username, u.role, s.expires_at
            FROM sessions AS s
            JOIN users AS u ON u.user_id = s.user_id
            WHERE s.token_hash = ?
            """,
            (token_hash,),
        # 一条会话对应一个账号，取出角色和过期时间后即可关闭连接。
        ).fetchone()
    finally:
        conn.close()

    # 登录后角色可能已变动，不能一直信任前端上一次拿到的角色。
    if row is None or row["role"] not in ("patient", "doctor"):
        return None

    # 新日期带 UTC 时区；旧数据若没写时区，也按 UTC 解释。
    # 例如 2026-11-17T08:00:00+00:00，解析失败就视为无效凭证。
    try:
        expires_at = datetime.fromisoformat(row["expires_at"])
        # 旧库可能只有无时区的时间字符串；先统一时区再比较。
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None

    # 当前时刻到达过期点就失效，不能将边界那一秒算作有效。
    if expires_at <= datetime.now(timezone.utc):
        return None

    # 返回普通字典，供 /me 和每个受保护接口复用。
    return {"user_id": row["user_id"], "username": row["username"], "role": row["role"]}


def logout_user(token: str):
    """退出时删除当前登录凭证，其它设备的登录不受影响。"""
    # 使用当前令牌的摘要定点删除，用户在其他设备的会话仍独立。
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    conn = get_connection()
    try:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
        # 提交撤销操作后，下一个请求将找不到对应 sessions 行。
        conn.commit()
    finally:
        conn.close()
