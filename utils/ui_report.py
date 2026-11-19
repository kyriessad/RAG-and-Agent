"""报告阅读、审核和发布。"""

import json

import requests
import streamlit as st

from utils import api_client
from utils.ui_session import EDITABLE_STATES, go_home, show_request_error
from utils.ui_components import doctor_label, report_doctor_label, show_record


# 将已授权的报告字典绘成摘要、关注项和引用
def display_report(content):
    # 不同版本的规则代码含义不同，采用报告保存时的含义和相应的视觉提示。
    levels = {"high": ("已触发升级", "red"), "unassessed": ("待评估", "gray"),
              "no_alert": ("未触发预设告警", "blue"), "no_flag": ("未标记异常", "blue"),
              "review": ("需要复核", "orange")}
    level = content.get("risk_level") or "unassessed"
    label, color = levels.get(level, (level, "gray"))
    st.badge("规则评估 · " + label, color=color, icon=":material/monitor_heart:")
    st.markdown("#### 随访摘要")
    if content.get("legacy_report"):
        # 已签发的旧版 Markdown 保留其段落，省去与阅读页标题重复的通用标题。
        summary = content["summary"].removeprefix("# 随访报告\n")
        st.markdown(summary)
    else:
        st.text(content["summary"])  # 患者原话按纯文本换行，符号不会成为网页标记。
    if content.get("abnormal_points"):
        st.markdown("#### 需要关注")
        for point in content["abnormal_points"]:
            # 部分历史记录在单个关注项中保存了 JSON 字符串列表，阅读时按原顺序换行。
            if isinstance(point, str) and point.startswith("["):
                try:
                    items = json.loads(point)
                    if isinstance(items, list) and items:
                        point = "\n".join(items)  # 混入非字符串时抛 TypeError，仍展示原文。
                except (ValueError, TypeError):
                    pass
            st.text(point)
    if content.get("recommendations"):
        st.markdown("#### 参考建议")
        for recommendation in content["recommendations"]:
            st.text(recommendation)  # 保留建议文字原有的〔1〕等引用编号。
    if content.get("sources"):
        st.markdown("#### 引用依据")
        for number, source in enumerate(content["sources"], 1):
            with st.expander(f"〔{number}〕{source['source']} · 查看原文"):
                # 只读报告生成时保存的片段，知识库后续更新不改变这里的依据。
                st.text(source["excerpt"])
                st.caption("片段编号 " + source["chunk_id"])


# 绘制医生草稿或患者终稿，标题由调用方根据状态提供
def show_report_sheet(view, title):
    # 只作用于报告单，正文自动折行，手机可上下阅读；背景叠加透明色以适应深色主题。
    st.html("""
    <style>
      .st-key-report_sheet {
        border-top: 4px solid #27867d !important;
        border-radius: 14px;
        background: linear-gradient(150deg, rgba(53,151,137,.09), transparent 30%);
        box-shadow: 0 5px 18px rgba(20,69,64,.06);
        padding: clamp(1rem, 2.2vw, 2rem) !important;
      }
      .st-key-report_sheet h3 { letter-spacing: .012em; }
      .st-key-report_sheet h4 { padding: .8rem 0 .22rem; font-size: 1.05rem; }
      .st-key-report_byline { border-inline-start: 3px solid #78a99e; padding-inline-start: .85rem; }
      .st-key-report_sheet [data-testid="stText"],
      .st-key-report_sheet [data-testid="stText"] pre {
        font-family: inherit; white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.8; max-width: 76ch;
      }
      .st-key-report_review_actions { border-radius: 14px; }
    </style>
    """)
    with st.container(key="report_sheet", border=True):
        st.subheader(title)
        # 当次负责医生由报告快照或可核实的历史登记给出，与当前任务联系人分别展示。
        with st.container(key="report_byline", gap="xxsmall"):
            st.write("本次负责医生", report_doctor_label(view.get("responsible_doctor"),
                                                    view.get("responsible_doctor_source")))
            if view.get("reviewer"):
                signed = "审核医生 · " + doctor_label(view["reviewer"])
                if view.get("reviewed_at"):
                    signed += "　｜　" + view["reviewed_at"]
                st.caption(signed)
            elif view.get("reviewed_at"):
                st.caption("审核时间　" + view["reviewed_at"])
        display_report(view["content"])
        if view.get("note"):
            st.markdown("#### 医生意见")
            st.text(view["note"])  # 署名和意见均来自当前报告版本。


# 展示一个任务的报告进度。
def show_report_workflow(token, task_id, doctor=False):
    # 同一任务的 GET、生成和审核操作共用此前缀，例如 /tasks/F001/report。
    path = f"/tasks/{task_id}/report"
    try:
        view = api_client.request_api("GET", path, token=token).json()  # 每轮重新授权，页面不缓存报告正文。
        # 等待异步生成、审核或发布时可刷新；已发布的阅读页直接呈现报告正文。
        can_refresh = view["task_status"] in ("submitted", "in_review") or (doctor and view["task_status"] == "needs_info")
        if can_refresh and st.button(
                "刷新报告状态", key="refresh_report", type="tertiary", icon=":material/refresh:"):
            st.rerun()
        # 记录已提交而任务仍在 submitted，说明草稿生成尚未完成。
        if view["task_status"] == "submitted":
            st.info("随访记录已保存，正在等待整理报告。")
            if doctor and st.button("生成报告草稿", key="generate_report", type="primary"):
                with st.spinner("正在检索资料并生成报告……"):
                    # 只发送当前正式记录编号，即使重试多次也不能创建第二条随访记录。
                    api_client.request_api("POST", path + "/generate", token=token, data={"record_id": view["record_id"]})
                st.rerun()  # 成功后重新读取；页面重复刷新不会再次生成。
        # 患者和医生使用不同的页面控件，后端仍对每次 GET 与 POST 独立鉴权。
        if not doctor:
            if view["task_status"] == "needs_info":
                if view.get("note"):
                    st.warning("医生请你补充", icon=":material/edit_note:")
                    st.write(view["note"])  # 退回时明确展示要补的内容，同时保护草稿正文。
                else:
                    st.info("医生请你补充本次随访信息，当前没有可查看的具体意见。")
            # 仅在接口确实返回 content 时显示正文。发布前患者响应里没有这个字段。
            if "content" in view:
                show_report_sheet(view, "我的最终随访报告")
                if st.button("返回我的随访", key="report_back_home"):
                    go_home()
            # 已通过但尚未发布时也走等待分支，医生仍需点击发布按钮。
            elif view["task_status"] == "in_review":
                st.info("报告正在等待医生审核或发布。")
            elif view["task_status"] == "completed":
                st.info("本次随访已结束，目前没有可查看的已发布报告。")
            return
        # 医生可以在报告旁核对同一个 record_id 的提交版本和原始读数。
        if view.get("record"):
            # 封面给出实际提交的血压；展开可逐项比对测量时间、症状、服药和原始提交版本。
            record = view["record"]
            heading = "本次提交的原始记录"
            if record.get("systolic_bp") is not None and record.get("diastolic_bp") is not None:
                heading += f"　·　血压 {record['systolic_bp']}/{record['diastolic_bp']} mmHg"
            with st.expander(heading, expanded="content" not in view):
                show_record(view["record"])  # 使用患者确认时的正式版本，中文展示原始信息。
        if "content" not in view:
            if view["task_status"] in EDITABLE_STATES:
                st.info("正在等待患者填写并确认本次随访信息。")
            return  # 模型失败时仍可看原始记录、重新生成。
        report_states = {"pending_review": "待审核报告", "approved": "待发布报告",
                         "returned": "已退回的报告", "published": "已发布报告"}
        title = report_states.get(view.get("status"), "随访报告")
        if view.get("status") not in ("pending_review", "approved"):
            show_report_sheet(view, title)
            return  # 已发布或已退回版本留作阅读，不绘制可提交的审核操作。
        # 宽屏把决定区放在草稿旁，窄屏由 Streamlit 自动堆叠，先阅读再提交。
        paper, decision = st.columns([2, 1], gap="medium")
        with paper:
            show_report_sheet(view, title)
        panel = decision.container(key="report_review_actions", border=True)
        # expected_version 是打开此报告时读到的版本，不在页面自行加一。
        payload = {"report_id": view["report_id"], "expected_version": view["version"]}
        # key 包含账号、报告和版本。换患者或另一位医生登录时，不能沿用之前的审核输入。
        key = f"{st.session_state['user']['user_id']}_{view['report_id']}_{view['version']}"
        if view["status"] == "pending_review":
            # 四种动作中的前三种用于审核；发布只会在服务端返回 approved 后出现。
            options = {"通过": "approve", "修改后通过": "edit", "退回补充": "return"}
            panel.subheader("审核决定")
            choice = panel.radio("审核操作", list(options), key="choice_" + key)
            payload["action"] = options[choice]  # 例如“退回补充”提交给 API 的 action 是 return。
            if choice == "修改后通过":
                # 编辑器初值取数据库内原摘要，医生可直接更正和补充本次情况。
                payload["summary"] = panel.text_area("修改后的摘要", view["content"]["summary"],
                                                  max_chars=8000, key="summary_" + key)
                # 一行代表一个关注项；写回列表保留条目界限，空行会略过。
                points = panel.text_area("需关注情况（每行一项）", "\n".join(view["content"]["abnormal_points"]),
                                      key="points_" + key)
                payload["abnormal_points"] = []
                for line in points.splitlines():
                    if line.strip():
                        payload["abnormal_points"].append(line.strip())
                payload["keep_sources"] = []  # 保留建议及其对应引用一起提交，不能改写知识库原文。
                for number, source in enumerate(view["content"]["sources"], 1):
                    if panel.checkbox(f"保留建议〔{number}〕 {source['quote']}", value=True,
                                   key=f"keep_{key}_{number}"):
                        payload["keep_sources"].append(number)
            note_label = "给患者的审核意见（选填）"
            button_label = "确认通过"
            if choice == "退回补充":
                note_label = "请写明需要患者补充什么（必填）"
                button_label = "退回给患者补充"
            elif choice == "修改后通过":
                button_label = "保存修改并通过"
            payload["note"] = panel.text_area(note_label, max_chars=2000, key="note_" + key)
            if panel.button(button_label, key="review_" + key, type="primary", width="stretch"):
                # 必填项留空时在决定区提示，保留其他填写内容，不发送无效审核。
                if choice == "退回补充" and not payload["note"].strip():
                    panel.warning("请先填写需要患者补充的内容。")
                    return
                if choice == "修改后通过" and not payload["summary"].strip():
                    panel.warning("请先填写修改后的摘要。")
                    return
                api_client.request_api("POST", path + "/review", token=token, data=payload)
                st.rerun()
        elif view["status"] == "approved":
            panel.subheader("发布报告")
            panel.write("审核已通过，发布后患者即可查看。")
            if panel.button("发布给患者", key="publish_" + key, type="primary", width="stretch"):
                payload["action"] = "publish"
                api_client.request_api("POST", path + "/review", token=token, data=payload)
                st.rerun()
    except requests.RequestException as error:
        show_request_error(error)  # 保留可修改的输入；409 时刷新版本，503 时可重新生成。
