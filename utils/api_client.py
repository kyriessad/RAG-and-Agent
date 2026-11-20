# 浏览器一般打开 http://localhost:8501，这是 Streamlit 的页面地址。
# 本文件默认请求 http://127.0.0.1:8000，这是 FastAPI 的接口地址。
# requests 是第三方 HTTP 库；这里通过它发送 GET、POST 等请求。
import requests
# 从 agent.yml 读取服务地址；部署改变端口时不用逐页改 URL。
from utils.config_handler import agent_conf


# 接受前端的请求，发给后端执行，再把后端结果返回给前端
def request_api(method, path, token=None, data=None, stream=False):
    # eg: path='/auth/login'，url为 http://127.0.0.1:8000/auth/login。
    # method 是请求方法，例如 GET 读取、PUT 保存完整草稿、POST 确认或审核。
    # rstrip 去掉地址末尾斜杠，再接 /tasks/F001/report 等路径。
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
        # data 是普通字典；json=data 负责序列化，并设置相应请求头。
        method, url, headers=headers, json=data,
        # 连接最多等待 5 秒，连续读取数据的等待上限为 180 秒，流式聊天可逐行消费。
        stream=stream, timeout=(5, 180),
    )
    try:
        # raise_for_status() 检查这类状态码，但不会检查 JSON 的字段或模型生成内容。
        response.raise_for_status()
    # 捕获 HTTP 错误再抛回页面，因为 401 和 409 需要不同的提示或恢复动作。
    except requests.HTTPError:
        # 例如后端返回 HTTP 401   .content正文为 {"detail": "登录已失效，请重新登录。"}。
        try:
            # 读取正文，并缓存到对象内部。
            response.content
        finally:
            # 即使读取错误正文失败也释放连接，避免连接池被占满。
            response.close()
        # 不带参数的 raise 表示重新抛出正在处理的 HTTPError，交回 app.py 的 except。
        raise
    # 返回 Response；普通接口由页面 .json() 取字典，聊天则用 .iter_lines()。
    return response
