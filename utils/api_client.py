# 浏览器一般打开 http://localhost:8501，这是 Streamlit 的页面地址。
# 本文件默认请求 http://127.0.0.1:8000，这是 FastAPI 的接口地址。
# requests 是第三方 HTTP 库；这里通过它发送 GET、POST 等请求。
import requests
from utils.config_handler import agent_conf


# 接受前端的请求，发给后端执行，再把后端结果返回给前端
def request_api(method, path, token=None, data=None, stream=False):
    # eg: path='/auth/login'，url为 http://127.0.0.1:8000/auth/login。
    # method = "GET" 或 "POST"
    url = agent_conf["backend_url"].rstrip("/") + path
    # headers 是 dict，携带身份、正文类型等 HTTP 附加信息。
    # 初始为空字典 {}，是因为登录和注册请求还没有登录凭证可带。
    headers = {}
    # 确认身份的凭证/令牌
    if token:
        # headers = {"Authorization": "Bearer example-token"}。
        headers["Authorization"] = "Bearer " + token
    # 向后端发起请求并接受结果
    response = requests.request(
        method, url, headers=headers, json=data,
        stream=stream, timeout=(5, 180),
    )
    try:
        # raise_for_status() 检查这类状态码，但不会检查 JSON 的字段或模型生成内容。
        response.raise_for_status()
    except requests.HTTPError:
        # 例如后端返回 HTTP 401   .content正文为 {"detail": "登录已失效，请重新登录。"}。
        try:
            # 读取正文，并缓存到对象内部。
            response.content
        finally:
            response.close()
        # 不带参数的 raise 表示重新抛出正在处理的 HTTPError，交回 app.py 的 except。
        raise
    return response
