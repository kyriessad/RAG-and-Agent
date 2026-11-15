import hashlib
import hmac
import secrets
import sqlite3

from utils.path_tool import get_abs_path


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


# 初始化数据库，连接/创表
def init_db():
    # 连接数据库
    conn = sqlite3.connect(get_abs_path("data/app.db"))

    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                created_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    finally:
        conn.close()