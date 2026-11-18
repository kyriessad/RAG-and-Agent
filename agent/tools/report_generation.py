"""从本次正式记录和检索原文组装报告。模型只选择已有内容，不能自行提供来源。"""

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from utils.path_tool import get_abs_path


class ReportSelection(BaseModel):
    """模型输出示例 {"summary_order":[1,2,3],"abnormal_points":[],"references":[1]}。"""

    model_config = ConfigDict(extra="forbid", strict=True)  # 拒绝模型添加诊断、风险或任意来源字段。
    summary_order: list[int]  # 用数字排列确认事实，程序检查每条事实都只用一次。
    abnormal_points: list[int]  # 需要医生关注的事实编号，可以为空。
    references: list[int] = Field(min_length=1, max_length=3)  # 从本轮资料中选一至三段。


def record_facts(record):
    """把正式记录转为带标签的中文事实列表。不读库、不推算测量时间；兼容旧记录。"""
    data = record["data"]  # 如 {"measured_at":"今天早上","adverse_effects":"吃药后没有不舒服"}。
    facts = []  # 列表中的文字也将原样展示给医生。
    # 已测量时只用正式记录里成对保存的读数，单位固定为 mmHg。
    if record["bp_measurement_status"] == "measured":
        facts.append(f"本次血压为 {record['systolic_bp']}/{record['diastolic_bp']} mmHg。")
    else:
        facts.append("本次未测量血压。")
    # 仅当患者提供过测量时间才列出，不把记录提交时间当成测量时间。
    if data.get("measured_at"):
        facts.append("测量时间（自述）：" + data["measured_at"])
    # 直接保留症状原话，尤其要保留“没有”“曾经”等限定词。
    if record.get("symptoms"):
        facts.append("症状（自述）：" + record["symptoms"])
    adherence = record.get("medication_adherence")
    # 新记录用枚举，旧记录可能存整段自述，展示时都要保留。
    if adherence:
        labels = {"regular": "按时服药", "missed": "有漏服", "not_prescribed": "未开药"}
        facts.append("服药情况：" + labels.get(adherence, adherence))  # 历史记录可能直接存中文。
    for name, label in (("missed_dose_reason", "漏服原因"), ("adverse_effects", "服药后反应"),
                        ("missed_reason", "原记录漏服原因"), ("adverse_reactions", "原记录服药后反应")):
        # 未填的原因不进入已确认事实，避免把“没有提供”改写成“没有漏服”。
        if data.get(name):
            facts.append(label + "：" + data[name])
    return facts


def select_sources(documents):
    """把检索返回的 Document 转成带稳定编号的原文候选列表。没有正文时返回空列表。"""
    choices = []  # 每个候选保留原始片段和可引用的完整段落，方便医生核对上下文。
    # 同一文档的相邻片段可能重叠，用文件名和引用文字一起去重。
    seen = set()
    for doc in documents:
        name = Path(doc.metadata.get("source") or "").name  # 不把部署机器的绝对路径发给页面。
        if not name:
            continue
        chunk = doc.page_content
        # 同名文档的片段内容变了，摘要也会变化；重跑同一原文则稳定。
        digest = hashlib.sha256((name + "\n" + chunk).encode()).hexdigest()
        for line in chunk.splitlines():
            # 仅去掉行首尾空白，正文不会交给模型改写，便于逐字比对。
            quote = line.strip()
            if len(quote) < 12 or quote.startswith(("【", "资料", "适用范围", "整理日期", "主要来源", "来源链接", "http")):
                continue  # 标题、元数据、空白行不能独立作为建议。
            if (name, quote) in seen:
                continue  # 相邻片段重叠时，只给模型一个对应选项。
            seen.add((name, quote))
            # quote 是要引用的一段；excerpt 保存这段所在的完整检索片段。
            choices.append({"source": name, "chunk_id": digest, "excerpt": chunk, "quote": quote})
    return choices


def build_report(record):
    """输入授权的正式记录字典，调用 Chroma 和聊天模型，返回可持久保存的报告字典。

    生成失败或模型给出越界/重复编号会抛异常；函数不写医疗库，调用者可原样重试。
    """
    from agent.tools.chat_session import ready_knowledge  # 延迟到生成时加载向量库和模型。
    from model.factory import chat_model

    facts = record_facts(record)  # 模型只能重排或选取这些已确认事实。
    # 公共知识检索使用主题词，检索词中无需发送账号、姓名等身份数据。
    query = "高血压出院随访 家庭血压测量记录 服药注意事项"
    # 如果正式记录已经触发升级，就优先检索相关资料，风险等级本身保持原值。
    if record["risk_level"] == "high":
        query = "血压明显升高 危险症状 紧急就医"
    elif record["medication_adherence"] == "missed":
        query = "出院后漏服药物 记录 联系负责医生"
    # 先取回这次真实命中的 Document，再为可引用的段落编号。
    choices = select_sources(ready_knowledge().retriever_docs(query))
    # 没有来源时无法生成带依据的报告，交给上层保留记录并提示重试。
    if not choices:
        raise ValueError("知识库未返回可引用的正文。")
    paragraphs = []  # 给模型编号及段落；完整原文由程序保管。
    for number, source in enumerate(choices, 1):
        paragraphs.append([number, source["quote"]])
    numbered_facts = list(enumerate(facts, 1))  # 第一条事实为 [1, "本次血压为 ..."]。
    # 传给模型的只有本次事实、既有规则和编号候选，不传登录凭证。
    context = {"facts": numbered_facts, "rule_risk": record["risk_level"], "references": paragraphs}
    prompt = Path(get_abs_path("prompts/report_selection.txt")).read_text(encoding="utf-8")
    messages = [("system", prompt), ("human", json.dumps(context, ensure_ascii=False))]
    # 这是报告生成的网络调用点；要求 JSON 后还须在本地核对字段和编号。
    reply = chat_model.bind(response_format={"type": "json_object"}).invoke(messages)
    selection = ReportSelection.model_validate_json(reply.content)  # 不接受多余正文、字符串编号或未知字段。
    return assemble_report(record, facts, choices, selection)


def assemble_report(record, facts, choices, selection):
    """核对模型给出的数字并生成报告字典。只复制证据原文，越界或重复编号抛 ValueError。"""
    # 排序后恰好是 1 到 N 才表示每条事实都选中且仅出现一次。
    if sorted(selection.summary_order) != list(range(1, len(facts) + 1)):
        raise ValueError("事实排列不完整。")
    summary = []
    for number in selection.summary_order:
        summary.append(facts[number - 1])  # 编号从 1 开始，Python 列表下标从 0 开始。
    abnormal = []
    # 关注项可以为空，模型只能选事实编号，不能另写一条猜测。
    for number in selection.abnormal_points:
        if number < 1 or number > len(facts) or facts[number - 1] in abnormal:
            raise ValueError("关注项编号无效。")
        abnormal.append(facts[number - 1])
    if record.get("risk_reason"):
        abnormal.insert(0, record["risk_reason"])  # 明确规则提醒不会被模型的空选择吞掉。
    sources = []
    recommendations = []
    # 引用编号来自这次检索，不能凭模型给出的文件名临时补找证据。
    for number in selection.references:
        if number < 1 or number > len(choices) or choices[number - 1] in sources:
            raise ValueError("引用编号无效。")
        source = choices[number - 1]
        sources.append(source)
        recommendations.append(f"{source['quote']}〔{len(sources)}〕")  # 来源序号由程序生成。
    # 风险直接复制正式记录，列表中每条参考建议都与保存的一段原文相连。
    return {"summary": "\n".join(summary), "risk_level": record["risk_level"],
            "abnormal_points": abnormal, "recommendations": recommendations, "sources": sources}
