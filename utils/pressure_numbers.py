"""解析患者口述的明确血压整数，供采集校验和模型前风险提醒共用。"""

import re


CN_DIGITS = dict(zip("零一二三四五六七八九两", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2)))
NUMBER = r"(?:[-+\d.]+|[零〇一二两三四五六七八九十百千万点]+)"
PRESSURE_PAIR = re.compile(rf"({NUMBER})\s*[/／、比]\s*({NUMBER})")
_DIGIT = "[一二两三四五六七八九]"
_CHINESE_INTEGER = re.compile(rf"(?:(?P<h>{_DIGIT})百)?(?:(?P<t>{_DIGIT})?十(?P<u>{_DIGIT})?|零(?P<z>{_DIGIT}))?")


def pressure_number(token: str) -> int | None:
    """只转换可以唯一解读的整数字词；“一百四”不猜 104 或 140。"""
    if not token:
        return None
    if token.isdecimal():
        return int(token) if len(token) <= 3 else None
    if token in CN_DIGITS:
        return CN_DIGITS[token]
    match = _CHINESE_INTEGER.fullmatch(token)
    if not match or (match["z"] and not match["h"]):
        return None
    hundreds = CN_DIGITS.get(match["h"], 0) * 100
    tens = CN_DIGITS.get(match["t"], 1) * 10 if "十" in token else 0
    return hundreds + tens + CN_DIGITS.get(match["u"] or match["z"], 0)
