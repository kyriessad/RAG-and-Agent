"""按账号和业务范围恢复聊天，在消息内展示办理入口。"""

import json
from urllib.parse import urlencode
from uuid import uuid4

import requests
import streamlit as st

from utils import api_client
from utils.ui_chat_cards import show_chat_card, show_task_actions
from utils.ui_components import show_contact_phone, show_risk_notice, task_doctor, task_name, task_state_label
from utils.ui_conversations import show_home_reminders
from utils.ui_session import EDITABLE_STATES, draft_has_changes, get_view, show_request_error


PATIENT_QUESTIONS = {
    "查我的随访计划": "我有哪些随访计划？请列出日期和状态。",
    "看出院记录": "请帮我查看自己的出院记录。",
    "看最新报告": "我最近发布的随访报告写了什么？",
    "问健康知识": "请查公共知识库，告诉我在家测血压时怎样做记录。",
}
DOCTOR_QUESTIONS = {
    "查待审核报告": "有哪些随访报告等我审核或发布？",
    "登记患者出院": "我要登记患者出院，请帮我准备登记资料。",
    "看逾期随访": "哪些由我负责的随访已经逾期了？",
    "处理风险提醒": "请列出我还没有处理的风险提醒。",
}


def show_chat_sources(sources):
    if sources is None:
        return
    if not sources:
        st.caption("知识库未返回可核对的原文。")
        return
    with st.expander(f"查看本轮检索原文（{len(sources)} 段）"):
        for entry in sources:
            st.caption(entry["source"])
            st.text(entry["excerpt"])


def show_message(token, message):
    with st.chat_message("user" if message["role"] == "user" else "assistant"):
        receipt = message.get("receipt")
        if receipt:
            st.success(message["content"])
            for plan in receipt.get("schedule", []):
                st.write(f"{task_name(plan)}　{plan['scheduled_at']}")
        elif message.get("content"):
            st.markdown(message["content"])
        show_chat_sources(message.get("sources"))
        if message.get("risk"):
            show_risk_notice(message["risk"])
        if message.get("status") == "interrupted":
            st.caption("这条提问尚无完整答复，可稍后刷新或重新提问。")
        for index, card in enumerate(message.get("cards") or []):
            show_chat_card(token, card, f"msg_{message['id']}_{index}", message_id=message["id"], card_index=index)


def send_turn(token, data):
    with st.chat_message("user"):
        st.write(data["message"])
    with st.chat_message("assistant"):
        answer_box = st.empty()
        chunks = []
        completed = False
        try:
            with api_client.request_api("POST", "/chat", token=token, data=data, stream=True) as response:
                for line in response.iter_lines(chunk_size=1):
                    if not line:
                        continue
                    event = json.loads(line)
                    kind, content = event["type"], event["content"]
                    if kind == "reset":
                        chunks.clear()
                        answer_box.empty()
                    elif kind == "delta":
                        chunks.append(content)
                        answer_box.markdown("".join(chunks) + " ▌")
                    elif kind == "risk":
                        show_risk_notice(content)
                    elif kind == "answer":
                        answer_box.markdown(content)
                        completed = True
                    elif kind == "error":
                        answer_box.empty()
                        st.error(content)
                        return
        except requests.RequestException as error:
            answer_box.empty()
            show_request_error(error)
            st.caption("重新打开本对话即可查看后端已保存的消息。")
            return
        except (ValueError, KeyError):
            answer_box.empty()
            st.error("回复数据异常，请重新打开对话。")
            return
        if not completed:
            answer_box.empty()
            st.warning("连接已结束，请重新打开对话查看保存结果。")
            return
    st.rerun()


def show_chat(token, mode, task_id=None, patient_id=None):
    scope = {}
    if task_id:
        scope["task_id"] = task_id
    elif patient_id:
        scope["patient_id"] = patient_id
    suffix = "?" + urlencode(scope) if scope else ""
    history = get_view(token, "/chat/history" + suffix)
    if history is None:
        if st.button("重新加载对话"):
            st.rerun()
        return
    for message in history["messages"]:
        show_message(token, message)
    disabled = mode == "collect" and draft_has_changes()
    if disabled:
        st.caption("请先保存核对单上的修改，再继续采集。")
    placeholder = "说说最近测的血压、服药情况或身体感受" if mode == "collect" else "输入你想查询或办理的事"
    choices = DOCTOR_QUESTIONS if st.session_state["user"]["role"] == "doctor" else PATIENT_QUESTIONS
    question = None
    if not task_id:
        with st.popover("助手能做什么", icon=":material/apps:"):
            for label, prompt in choices.items():
                if st.button(label, key="chat_example_" + label):
                    question = prompt
    typed = st.chat_input(placeholder, max_chars=4000, disabled=disabled, key="chat_prompt")
    question = (typed or question or "").strip()
    if question and not disabled:
        send_turn(token, dict(scope, mode=mode, message=question, request_id=str(uuid4())))


def show_assistant(token, doctor, current_task, detail):
    if current_task:
        task_id = current_task["task_id"]
        label = task_name(current_task)
        if doctor:
            label = current_task["patient_name"] + " · " + label
        st.title(label)
        st.caption(f"{current_task['followup_date']}　·　{task_state_label(current_task)}")
        if not doctor:
            with st.popover("联系本次负责医生", icon=":material/call:"):
                st.write(task_doctor(current_task))
                show_contact_phone(current_task.get("doctor_phone"))
        conversation = st.container()
        show_task_actions(token, current_task, doctor)
        mode = "collect" if not doctor and current_task["task_status"] in EDITABLE_STATES else "query"
        with conversation:
            if mode == "collect":
                st.caption("最近一次是什么时候测的血压？也可以告诉我还没量。")
            show_chat(token, mode, task_id=task_id)
    elif detail:
        patient = detail["patient"]
        st.title(patient["patient_name"] + "的对话")
        st.caption(f"患者编号　{patient['patient_id']}")
        with st.popover("联系患者", icon=":material/call:"):
            show_contact_phone(patient.get("phone"))
        show_chat(token, "query", patient_id=patient["patient_id"])
    else:
        st.title("随访助手")
        show_home_reminders(token, doctor)
        show_chat(token, "query")
