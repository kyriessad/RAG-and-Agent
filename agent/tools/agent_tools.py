import csv
import json
import os
from langchain_core.tools import tool
from rag.rag_service import RagService
import random
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.config_handler import agent_conf
from langchain.tools import ToolRuntime

rag = RagService()
weather = ["天气为多云，气温22摄氏度，空气湿度65%，东风2级，AQI45，最近6小时降雨概率较低",
           "天气为小雨，气温15摄氏度，空气湿度85%，北风3级，AQI30，最近6小时降雨概率较高",
           "天气为雷阵雨，气温30摄氏度，空气湿度90%，西南风4级，AQI55，最近6小时降雨概率极高",
           "天气为晴天，气温35摄氏度，空气湿度25%，西北风1级，AQI78，最近6小时降雨概率极低",
           "天气为阴天，气温18摄氏度，空气湿度55%，东北风2级，AQI60，最近6小时降雨概率中等",
           "天气为大雪，气温零下5摄氏度，空气湿度70%，北风5级，AQI25，最近6小时降雨概率极高",
           "天气为雾霾，气温12摄氏度，空气湿度75%，无持续风向1级，AQI150，最近6小时降雨概率极低",
           "天气为阵雨，气温20摄氏度，空气湿度80%，东南风3级，AQI40，最近6小时降雨概率较高",
           "天气为晴转多云，气温25摄氏度，空气湿度40%，南风2级，AQI35，最近6小时降雨概率极低",
           "天气为多云，气温8摄氏度，空气湿度45%，西北风6级，AQI85，最近6小时降雨概率较低",
           "天气为晴天，气温26摄氏度，空气湿度50%，南风1级，AQI21，最近6小时降雨概率极低"]

month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", ]
external_data = {}
"""
    external_data = {
        "user_1001": {
            "month":{"特征":xxx, "效率":xxx...},
            "month":{"特征":xxx, "效率":xxx...},
            "month":{"特征":xxx, "效率":xxx...},
            ...
        }
        "user_1002": {
            "month":{"特征":xxx, "效率":xxx...},
            "month":{"特征":xxx, "效率":xxx...},
            "month":{"特征":xxx, "效率":xxx...},
            ...
        }
        "user_1003": {
            "month":{"特征":xxx, "效率":xxx...},
            "month":{"特征":xxx, "效率":xxx...},
            "month":{"特征":xxx, "效率":xxx...},
            ...
        }
    }
"""


@tool(description="从向量存储中检索参考资料")
def rag_search(query: str) -> str:
    return rag.rag_search(query)


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    result_weather = random.choice(weather)
    return f"{city}:{result_weather}"


@tool(description="获取当前登录用户有使用记录的月份")
def get_available_months(runtime: ToolRuntime) -> str:
    generate_external_data.invoke({})
    user_id = runtime.context["user_id"]
    user_records = external_data.get(user_id, {})
    months = sorted(user_records.keys())
    if not months:
        return "当前用户暂无使用记录。"
    return "可查询的记录月份：" + "、".join(months)


@tool(description="导入外部数据(用户使用记录)")
def generate_external_data():
    if external_data:  # 如果已经导入了数据，直接返回
        return
    path = get_abs_path(agent_conf["external_data_path"])
    records = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:  # 防止表头出现\ufeff乱码
        reader = csv.DictReader(f)  # 把每行读成以表头为键的字典
        required_fields = {"用户ID", "特征", "清洁效率", "耗材", "对比", "时间"}
        if not required_fields.issubset(reader.fieldnames or []):  # issubset 检查 6 个列是否齐全
            raise ValueError("使用记录文件缺少必要列")
        for row in reader:  # row = {"用户ID": "1001", "特征": "客厅,卧室", "时间": "2025-01",...}
            # 一行数据比表头多出字段时，DictReader 会使用 None 作为键。
            if None in row:
                raise ValueError(f"第 {reader.line_num} 行字段太多")
            # 一行数据缺少某些字段时，对应的值可能是 None。
            for field in required_fields:
                if row.get(field) is None:
                    raise ValueError(f"第 {reader.line_num} 行缺少字段：{field}")
            user_id = row["用户ID"].strip()
            month = row["时间"].strip()
            if not user_id or not month:
                raise ValueError(f"第 {reader.line_num} 行缺少用户ID或月份")
            if user_id not in records:
                records[user_id] = {}
            records[user_id][month] = {
                "特征": row["特征"],
                "效率": row["清洁效率"],
                "耗材": row["耗材"],
                "对比": row["对比"],
            }

    # 文件全部读取成功之后，再更新全局数据。
    external_data.update(records)


@tool(description="查询当前登录用户在指定月份的使用记录，YYYY-MM")
def fetch_external_data(month: str, runtime: ToolRuntime) -> str:
    generate_external_data.invoke({})

    user_id = runtime.context["user_id"]
    data: dict = external_data.get(user_id, {}).get(month)
    if data is None:
        return f"当前用户在{month}没有使用记录。"
    return json.dumps(data, ensure_ascii=False)


@tool(description="标记是否要切换提示词，没入参，返回值没作用")
def fill_context_for_report():
    return "fill_context_for_report已调用"


# if __name__ == '__main__':
#
#     print(get_weather("重庆"))
