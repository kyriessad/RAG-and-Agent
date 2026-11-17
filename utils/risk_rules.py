"""独立风险筛查规则：只读输入，返回提醒；不调用模型、网络或数据库。

规则用于决定随访流程是否需要升级，不能据此诊断或排除疾病。
血压依据：美国心脏协会说明超过 180/120 时应及时处置；本项目将边界
也纳入提醒，即大压 >=180 或小压 >=120，这是保守的项目策略。
https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings/when-to-call-911-for-high-blood-pressure
症状依据：NHS 的卒中症状说明；近期症状即使缓解也不能据此排除风险。
https://www.nhs.uk/conditions/stroke/symptoms/
胸痛和呼吸困难参考下列官方说明；本项目以更宽的词表升级给医生，
不根据单个词诊断病因，也不宣称所有词表命中都是临床急症。
https://www.nhs.uk/symptoms/chest-pain/
https://www.nhs.uk/symptoms/shortness-of-breath/
文字识别是有限的中文词表和否定过滤，不是完整的自然语言理解。
"""

import re  # 只用于拆分患者原话、寻找明确数值和固定词，不依赖外部服务。


RULE_VERSION = "followup-risk-v1"  # 保存到记录中，便于解释当时采用哪版规则。
SYSTOLIC_LIMIT = 180  # 单位 mmHg；等于边界也提醒，不自动诊断高血压急症。
DIASTOLIC_LIMIT = 120  # 小压单独达到边界也触发，不要求两个数值同时超限。

# 每项依次为规则码、提醒原因、患者可能使用的常见说法。
RED_FLAGS = (
    ("chest_pain", "出现胸痛相关描述", ("胸痛", "胸疼", "胸口痛", "胸口疼")),
    ("severe_headache", "出现剧烈头痛相关描述", (
        "剧烈头痛", "剧烈的头痛", "头痛得厉害", "头疼得厉害", "头痛欲裂",
        "头疼得很厉害", "头痛得很厉害", "头痛很厉害", "头疼很厉害",
    )),
    ("speech_difficulty", "出现说话不清相关描述", ("言语不清", "说话不清", "说不清话", "说话含糊", "口齿不清")),
    ("breathing_difficulty", "出现呼吸困难相关描述", ("呼吸困难", "喘不上气", "喘不过气")),
    ("one_sided_weakness", "出现一侧肢体无力或麻木相关描述", (
        "一侧肢体无力", "一侧手脚无力", "一侧手脚麻木", "左侧手脚无力", "右侧手脚无力", "半边身子没劲",
        "左手突然无力", "右手突然无力", "左腿突然无力", "右腿突然无力",
        "左手突然没劲", "右手突然没劲", "突然左手无力", "突然右手无力",
    )),
    ("vision_change", "出现突发视力变化相关描述", ("突然视物模糊", "突然看不清", "突然视力下降")),
)


def split_clauses(text):
    """输入原话字符串，返回短句列表；无副作用，空输入返回空列表。

    例：“没有胸痛，但是喘不上气”拆成两句，前一句的否定不覆盖后一句。
    顿号和“和”保留，让“没有胸痛、呼吸困难”仍能共用前面的否定。
    """
    clauses = []  # 普通字符串列表，不改变患者输入。
    # “我爸爸没事我现在胸痛”也能分开；(?=...) 在这些明确自述词前切开并保留词本身。
    pattern = r"[，,。；;！!？?\n]|但是|可是|不过|而是|但|(?=我(?:现在|目前|刚才|今天))"
    for part in re.split(pattern, text):
        part = part.strip()  # 清理首尾空格，方便识别只有数值的短句。
        if part:
            clauses.append(part)
    return clauses


def describes_current_patient(clause):
    """输入一段原话，返回是否适合按本患者情况检查；不推断患者没有症状。

    明确的假设、科普引用、他人情况或远期病史不作为当前阳性依据。
    “刚才/今天/昨天发生，现在好了”仍检查，不能把近期缓解当作安全。
    """
    excluded = ("如果", "假如", "假设", "要是", "比如", "例如", "万一", "科普", "资料说", "网上说", "医生说出现", "医生说如果")
    for word in excluded:  # “如果胸痛怎么办”不是已经胸痛。
        if word in clause:
            return False
    people = ("我爸爸", "我妈妈", "我父亲", "我母亲", "我爱人", "我老公", "我老婆", "我朋友", "别人", "邻居", "他有", "她有")
    for word in people:  # 不把别人描述写成本患者风险。
        if word in clause:
            return False
    history_words = ("去年", "前年", "上个月", "上周", "以前", "曾经", "小时候", "病史", "多年前")
    current_words = ("现在", "刚才", "刚刚", "今天", "目前", "此刻", "又出现", "又开始")
    for word in history_words:
        if word in clause:
            for current_word in current_words:
                if current_word in clause:  # 混合时态仍检查，避免漏掉复发描述。
                    return True
            return False  # 只有明确远期病史时不按当前发作处理。
    return True


def is_positive_mention(clause, start, end):
    """输入短句及匹配词的位置，返回这处是否明确肯定；不请求网络或写库。

    例如“没有胸痛”返回 False，“胸痛怎么办”返回 True。
    不确定及否定说法不伪装为阳性；调用方的未命中结果也不能称作安全。
    """
    if not describes_current_patient(clause):  # 假设、他人和远期病史先排除。
        return False
    before = clause[:start]  # 检查同一句中该症状前的否定，兼容并列症状。
    after = clause[end:]  # 用于“胸痛没有”等后置否定。
    # “没有发烧伴胸痛”中，伴/还有引出另一项阳性；“不伴胸痛”仍保留否定。
    for marker in ("还有", "同时有", "伴有", "伴", "另有"):
        position = before.rfind(marker)
        if position >= 0:
            prefix = before[:position].strip()
            if not prefix.endswith(("不", "没", "未", "没有", "无")):
                before = before[position + len(marker):]
    uncertain = ("不确定", "不知道", "是不是", "是否", "会不会", "担心会", "可能会", "不记得")
    for word in uncertain:
        if word in before:
            return False
    # “没有……但现在……”已由 split_clauses 拆开；常见的无标点转折也单独处理。
    for marker in ("现在", "目前", "刚才", "又出现", "开始出现"):
        position = before.rfind(marker)
        if position >= 0:
            before = before[position + len(marker):]
    # 双重否定表示不能说没有，不按“没有症状”处理。
    for phrase in ("不是没有", "并非没有", "不能说没有"):
        before = before.replace(phrase, "")
    # 不能用单字“不”否定“说话不清”，只匹配患者放在症状前的完整否定表达。
    for word in ("没有", "没出现", "未出现", "不伴", "未伴", "没伴", "否认", "并非", "不是", "未有", "无"):
        if word in before:
            return False
    # “胸痛没有/不存在”是否定；“刚才胸痛现在好了”仍应保留近期警示。
    if re.fullmatch(r"\s*(?:没有|不存在|未出现|是没有的|为无)\s*", after):
        return False
    return True


def add_reason(result, code, reason):
    """把一个唯一规则加入结果字典；返回 None，仅修改传入的内存结果。

    同一句里反复说“胸痛”只产生一个规则码，不重复列出同一原因。
    """
    if code not in result["rule_codes"]:  # 顺序保持稳定，方便保存和测试。
        result["rule_codes"].append(code)
        result["reasons"].append(reason)


def check_pressure(result, high, low):
    """检查两个可选整数，命中时更新内存结果；不补另一项、不改原始字段。

    输入 high=180、low=None 也会提醒；字符串或 bool 不当作有效血压。
    数值 1～999 仅是已有表单的格式范围，不表示这些数值临床合理。
    """
    if type(high) is int and SYSTOLIC_LIMIT <= high <= 999:
        add_reason(result, "systolic_ge_180", "收缩压达到或超过 180 mmHg")
    if type(low) is int and DIASTOLIC_LIMIT <= low <= 999:
        add_reason(result, "diastolic_ge_120", "舒张压达到或超过 120 mmHg")


def check_text(result, text):
    """检查一段原话中的红旗症状和明确血压，更新结果；无网络或存储操作。

    例：“血压190/120，胸口疼”直接命中，不必等待模型提取或完整随访提交。
    只识别本函数列出的数值格式；不把药物剂量、日期或裸单个数字当血压。
    """
    for clause in split_clauses(text):
        for code, reason, words in RED_FLAGS:
            for word in words:
                for match in re.finditer(re.escape(word), clause):
                    if is_positive_mention(clause, match.start(), match.end()):
                        add_reason(result, code, reason)
        # 成对数字需有血压/测量语境，或整句本身仅为 190/120，避免误读其他比值。
        for match in re.finditer(r"(?<![\d.])(\d{2,3})\s*[/／]\s*(\d{2,3})(?![\d.])", clause):
            before = clause[:match.start()]
            has_context = bool(re.search(r"血压|大压|高压|刚量|测量|量了|测了", before))
            if re.fullmatch(r"\d{2,3}\s*[/／]\s*\d{2,3}\s*(?:mmHg)?", clause, flags=re.IGNORECASE):
                has_context = True
            if has_context and is_positive_mention(clause, match.start(), match.end()):
                check_pressure(result, int(match.group(1)), int(match.group(2)))
        # 单独上报其中一项也及时检查；数值不推断另一项，不保存成草稿。
        pressure_words = (("收缩压|高压|大压", "high"), ("舒张压|低压|小压", "low"))
        for labels, kind in pressure_words:
            pattern = r"(?:" + labels + r")\s*(?:是|为|有|达到|到了|测得|[:：])?\s*(\d{2,3})(?![\d.])"
            for match in re.finditer(pattern, clause):
                if is_positive_mention(clause, match.start(), match.end()):
                    value = int(match.group(1))
                    if kind == "high":
                        check_pressure(result, value, None)
                    else:
                        check_pressure(result, None, value)


def evaluate_risk(data=None, message=None):
    """输入部分结构化字段及可选当前原话，输出一份可保存的规则结果字典。

    例：evaluate_risk({"systolic_bp": 185}) 的 level 为 high。
    data 只能为字典或 None，message 只能为字符串或 None，否则抛 ValueError。
    本函数不改输入，不读写数据库，不调用模型；模型失败时仍可独立运行。
    high 表示触发升级；unassessed 表示信息不足；no_alert 仅表示未命中预设规则。
    """
    if data is None:
        data = {}  # 无结构化数据时仍可检查当前聊天原话。
    if not isinstance(data, dict):
        raise ValueError("风险检查的数据必须是字段字典。")
    if message is not None and not isinstance(message, str):
        raise ValueError("风险检查的原话必须是文字。")
    result = {
        "level": "unassessed", "reasons": [], "rule_codes": [],
        "patient_message": "当前信息还不足以完成规则检查；没有提醒不代表排除风险。",
        "rule_version": RULE_VERSION,
    }
    high = data.get("systolic_bp")  # 缺值为 None；不使用历史记录或默认数值补齐。
    low = data.get("diastolic_bp")
    check_pressure(result, high, low)  # 结构化字段已有一项超限即可升级。
    for field in ("symptoms", "adverse_effects"):
        text = data.get(field)
        if isinstance(text, str):
            check_text(result, text)  # 否定和时态检查同样适用于手工输入。
    if message:
        check_text(result, message)  # 不依赖 Agent 是否成功或是否调用提议工具。
    symptoms = data.get("symptoms")  # 不确定的症状文字不能仅因为非空就认为已回答。
    symptoms_known = isinstance(symptoms, str) and bool(symptoms.strip())
    if symptoms_known:
        for word in ("不确定", "不知道", "不清楚", "是不是", "是否", "会不会", "说不准", "记不清"):
            if word in symptoms:
                symptoms_known = False  # 有真实红旗时下方仍优先 high；否则保持信息不足。
    if result["rule_codes"]:
        result["level"] = "high"
        result["patient_message"] = "血压数值触发及时联系医生的提醒。请尽快联系医生核对；如同时有胸痛、呼吸困难、说话不清等不适，请立即就医或联系当地急救服务，不要等待在线回复。不要自行调整药物。"
        for code in result["rule_codes"]:
            if code not in ("systolic_ge_180", "diastolic_ge_120"):
                result["patient_message"] = "你描述了需要及时处理的不适。请立即就医或联系当地急救服务，不要等待随访报告或在线回复。不要自行调整药物。"
                break
    elif (data.get("bp_measurement_status") == "measured"
          and type(high) is int and type(low) is int and 0 < low < high <= 999
          and symptoms_known):
        result["level"] = "no_alert"  # 不称“低风险/安全”，也不作临床风险分类。
        result["patient_message"] = "本轮信息未命中预设的升级规则；这不代表排除风险。如有新发或加重的不适，请及时联系医生。"
    return result
