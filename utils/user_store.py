import hashlib
import hmac
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from utils.path_tool import get_abs_path


# 所有账号和登录凭证都放在同一个医疗数据库里。
# DB_PATH 是字符串，例如 C:/项目/data/medical.db；测试时可替换为临时库路径。
DB_PATH = get_abs_path("data/medical.db")


def get_connection():
    # mode=rw 表示只打开已有数据库，路径写错时直接报错，不创建空数据库。
    database_uri = Path(DB_PATH).resolve().as_uri() + "?mode=rw"
    conn = sqlite3.connect(database_uri, uri=True, timeout=10)

    # conn 是数据库连接；Row 让结果支持 row["user_id"] 这样的列名读取。
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# 哈希密码
def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)  # 十六进制字符串
    digest = hashlib.pbkdf2_hmac(
        "sha256",  # 哈希算法
        password.encode("utf-8"),  # 用户密码
        bytes.fromhex(salt),  # 盐（需要转回字节）
        600_000,  # 迭代次数
    ).hex()  # 十六进制字符串

    return f"pbkdf2_sha256$600000${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    # 数据损坏时返回验证失败，不把异常的密码格式当成正常密码。
    if not isinstance(stored, str):
        return False

    try:
        method, iterations, salt, expected = stored.split("$")

        if method != "pbkdf2_sha256":
            return False
        # 重新计算哈希
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt),
            int(iterations),
        ).hex()
        # 恒定时间比较函数，无论两个字符串差异在哪里，它花费的时间都是固定的
        return hmac.compare_digest(actual, expected)

    except (ValueError, TypeError):
        return False


# 保留原来的函数名，启动时只检查现有表，不创建另一套账号表。
def init_db():
    conn = get_connection()
    try:
        # LIMIT 0 不读取账号内容，只确认登录所需的表和列确实存在。
        conn.execute("SELECT user_id, username, password_hash, role FROM users LIMIT 0")
        conn.execute("SELECT token_hash, user_id, expires_at FROM sessions LIMIT 0")
    finally:
        conn.close()


def register_user(username: str, password: str) -> bool:
    """注册患者账号；成功返回 True，用户名重复返回 False。"""
    username = username.strip()
    if not username or not password:
        raise ValueError("用户名和密码不能为空。")

    # 医疗库的 user_id 是文本，例如 U 加上一串唯一编号，不再使用自增整数。
    user_id = "U" + uuid4().hex
    password_hash = hash_password(password)
    conn = get_connection()
    try:
        # 注册页不能选择医生角色；医生账号由已有医疗账号体系提供。
        # 这里只创建登录账号，不自动编造患者姓名、年龄或病历。
        conn.execute(
            "INSERT INTO users (user_id, username, password_hash, role) VALUES (?, ?, ?, 'patient')",
            (user_id, username, password_hash),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError as error:
        conn.rollback()
        # 只把用户名重复解释为“注册失败”，其它数据库错误交给后端处理。
        if error.sqlite_errorname == "SQLITE_CONSTRAINT_UNIQUE":
            return False
        raise
    finally:
        conn.close()


def login_user(username: str, password: str) -> dict | None:
    """登录成功返回 token 和用户信息；账号、密码或角色不符合时返回 None。"""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT user_id, username, password_hash, role FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()

        # 角色直接读取数据库，不相信页面传来的角色名称。
        if row is None or row["role"] not in ("patient", "doctor"):
            return None
        if not verify_password(password, row["password_hash"]):
            return None

        # token 是发给前端的随机登录凭证；数据库只保存它的 SHA256 哈希。
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
        conn.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (token_hash, row["user_id"], expires_at.isoformat()),
        )
        conn.commit()

        # user 是普通字典，不包含密码哈希，可安全交给页面显示用户名和角色。
        user = {"user_id": row["user_id"], "username": row["username"], "role": row["role"]}
        return {"access_token": token, "user": user}
    finally:
        conn.close()


def get_current_user(token: str) -> dict | None:
    """根据登录凭证查询当前身份；凭证不存在或过期时返回 None。"""
    if not token:
        return None

    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    conn = get_connection()
    try:
        # 每次请求重新关联 users，用户角色以数据库中的当前值为准。
        row = conn.execute(
            """
            SELECT u.user_id, u.username, u.role, s.expires_at
            FROM sessions AS s
            JOIN users AS u ON u.user_id = s.user_id
            WHERE s.token_hash = ?
            """,
            (token_hash,),
        ).fetchone()
    finally:
        conn.close()

    if row is None or row["role"] not in ("patient", "doctor"):
        return None

    # 新日期带 UTC 时区；旧数据若没写时区，也按 UTC 解释。
    # 例如 2026-11-17T08:00:00+00:00，解析失败就视为无效凭证。
    try:
        expires_at = datetime.fromisoformat(row["expires_at"])
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None

    if expires_at <= datetime.now(timezone.utc):
        return None

    return {"user_id": row["user_id"], "username": row["username"], "role": row["role"]}


def logout_user(token: str):
    """退出时删除当前登录凭证，其它设备的登录不受影响。"""
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    conn = get_connection()
    try:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
        conn.commit()
    finally:
        conn.close()
