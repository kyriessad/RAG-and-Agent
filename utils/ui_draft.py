"""患者核对、修改和确认随访记录。"""

import requests
import streamlit as st

from utils import api_client
from utils.ui_session import (
    DRAFT_CHOICES,
    DRAFT_TEXT_FIELDS,
    clear_task_selection,
    draft_display_values,
    draft_has_changes,
    load_draft_fields,
    remember_draft_field,
    show_request_error,
)
from utils.ui_components import show_risk_notice


# 读取当前任务已收到的规则结果并展示，返回 None
def show_cached_task_risk(task_id):
    displayed = []  # 同一风险只展示一次，但通知保存失败提示仍需保留。
    # 一个缓存来自流式事件，另一个来自读取持久草稿，两者可能描述同一提醒。
    for key in ("draft_risk_event", "draft_loaded_risk"):
        content = st.session_state.get(key)
        if content is None or content.get("task_id") != task_id:
            continue  # 每份缓存明确绑定任务，不依赖页面当前显示的位置。
        risk = content.get("risk", {})
        if risk not in displayed:
            show_risk_notice(content)
            # 记录本轮已经呈现过的规则结果，防止后续缓存重复展示。
            displayed.append(risk)


# 从草稿保存失败的响应提取风险提醒，返回 None
def show_failed_draft_risk(error, task_id):
    response = error.response
    if response is None or response.status_code not in (409, 422, 503):
        return  # 401/403/404 必须走权限处理，不能据此恢复旧提醒。
    if st.session_state.get("user", {}).get("role") != "patient":
        return  # 仅患者任务草稿保存调用这里，不把它扩展成所有接口的通用缓存。
    if st.session_state.get("selected_task", {}).get("task_id") != task_id:
        return  # 保存失败期间任务已切换时，不能混入另一项任务的信息。
    try:
        # 后端可能同时返回 detail 和 risk，保存失败也有独立的风险信息。
        payload = response.json()
    except ValueError:
        return  # HTML 等非 JSON 错误没有可用的规则结果。
    if not isinstance(payload, dict) or not isinstance(payload.get("risk"), dict):
        return  # 兼容原有仅含 detail 的错误，不凭错误类型生成就医结论。
    if payload["risk"].get("level") != "high":
        return
    # 失败提示绑定 task_id，切换任务后不会误显示上一位患者的提醒。
    content = {"task_id": task_id, "risk": payload["risk"], "notification_saved": False}
    st.session_state["draft_risk_event"] = content  # 写库失败仍保留服务端已经检查出的危险信号。
    show_risk_notice(content)  # 当轮立即显示，不能等下一次 GET 或模型回答。


# 让患者确认并提交已保存的草稿，返回 None；仅点击按钮时发送 POST。
def show_patient_submit(token, task_id, latest, base):
    values = {}  # 当前显示内容也是确认依据；编辑后必须重新确认。
    for field in draft_display_values(base["data"]):
        # 将当前可见的每个输入都纳入核对，改一个数字也应取消既有勾选。
        values[field] = st.session_state.get("draft_field_" + field)
    signature = {"task_id": task_id, "revision": base["revision"],
                 "server_revision": latest["revision"], "editable": latest["editable"],
                 "values": values}  # 普通字典比较，明确区分服务器版本和当前显示。
    # 比较本次核对依据与上一次；内容、版本、可编辑状态发生变化都算新确认。
    if st.session_state.get("draft_confirm_signature") != signature:
        st.session_state["draft_confirm"] = False  # 内容或版本变动使以前的确认失效。
        st.session_state["draft_confirm_signature"] = signature
    # pop 读取后删除，网络错误引发的重置只需处理一次。
    if st.session_state.pop("draft_reset_confirmation", False):
        st.session_state["draft_confirm"] = False  # 提交失败后，下一轮先取消旧确认。
    # 完整性只是第一关，下一行还会检查未保存改动和版本过期。
    can_submit = latest["editable"] and not latest["missing_fields"]
    if draft_has_changes() or latest["revision"] != base["revision"] or latest.get("returned_draft"):
        can_submit = False  # 页面不把尚未保存的内容说成已经提交。
    if not can_submit:
        st.session_state["draft_confirm"] = False  # 禁用时不保留可能过期的勾选。
    st.divider()
    st.write("确认本次随访信息")
    if latest.get("returned_draft"):
        # 后端核对了最近提交的草稿版本；手工补充或对话保存出新版本后即可再次确认。
        st.info("医生退回了这份记录。请按医生意见告诉助手补充内容，或在核对单中修改后点「保存草稿」，再确认提交。")
    confirmed = st.checkbox("我已核对上面的信息，确认提交本次随访", key="draft_confirm",
                            disabled=not can_submit)  # 默认不勾选，不代替患者确认。
    # 只有患者主动勾选并点击按钮才发出 POST；普通页面重跑不会提交。
    if st.button("确认提交随访", key="draft_submit", type="primary", disabled=not can_submit or not confirmed):
        try:
            result = api_client.request_api("POST", f"/tasks/{task_id}/submit", token=token, data={
                "expected_revision": base["revision"], "confirmed": True,
            }).json()  # 服务端独立检查归属、版本、缺项和重复请求。
        except requests.RequestException as error:
            # 保留输入，但下一轮要求患者重新核对，避免误用之前的确认。
            st.session_state["draft_reset_confirmation"] = True
            show_request_error(error)  # 输入保持原样，不把错误当成成功。
            st.warning("请重新加载确认提交结果；连接中断时，后端可能已经保存，请勿反复点击。")
        else:
            st.session_state["draft_submit_notice"] = result  # 下一轮展示成功并读取实时状态。
            st.session_state["draft_reload"] = True
            # 接口已接收正式记录，下一轮读取实际处理进度，不让患者再次触发模型。
            st.rerun()


# 显示当前已保存的核对单，供患者提交前逐项核对
def show_saved_draft_summary(data):
    values = draft_display_values(data)  # 与手工输入框使用完全相同的选项和读数。
    pressure = values["bp_measurement_status"]
    if values["systolic_bp"] or values["diastolic_bp"]:
        pressure = f"{values['systolic_bp'] or '—'} / {values['diastolic_bp'] or '—'} mmHg"
    with st.container(border=True, key="saved_draft_summary"):
        st.markdown("**请核对本次随访信息**")
        lines = ["血压　" + pressure]
        for key, label in (("measured_at", "测量时间"), ("symptoms", "身体情况"),
                           ("medication_adherence", "服药情况"), ("missed_dose_reason", "漏服原因"),
                           ("adverse_effects", "服药后感受")):
            if values[key]:
                lines.append(label + "　" + values[key])  # 按原文展示患者已确认前的描述。
        st.text("\n".join(lines))


# 读取、展示并修改当前患者草稿，返回 None
def show_patient_draft(token, task_id):
    path = f"/tasks/{task_id}/draft"  # 例如 /tasks/F001/draft，任务归属仍由后端校验。
    try:
        latest = api_client.request_api("GET", path, token=token).json()  # 重进页面可恢复数据库里的草稿。
    except requests.RequestException as error:
        # 登录失效或归属改变都要丢弃缓存；暂时断线则允许恢复之前的草稿编辑。
        if error.response is not None and error.response.status_code in (401, 403, 404):
            clear_task_selection()  # 权限失效后不能继续展示旧草稿。
        else:
            show_cached_task_risk(task_id)  # 数据库暂不可读时，已收到的就医提醒仍然有效。
        show_request_error(error)
        if st.button("重新加载草稿", key="draft_retry"):
            st.rerun()
        st.stop()  # 请求失败不等于空草稿，也不开放可能覆盖旧数据的编辑。

    # latest 是本轮 GET 返回的版本，base 是用户开始编辑时的版本。
    base = st.session_state.get("draft_base")
    if base is None or base["task_id"] != task_id or st.session_state.pop("draft_reload", False):
        load_draft_fields(latest)  # 首次进入或明确保存成功后，采用服务器返回值。
    elif latest["revision"] != base["revision"] and not draft_has_changes():
        load_draft_fields(latest)  # 没有手工改动时，安全展示 Agent 或另一个页面保存的新版本。
    base = st.session_state["draft_base"]
    if latest.get("risk", {}).get("level") == "high":
        # 及时记录刚读到的高风险，以便下次网络断开时仍能呈现必要提醒。
        st.session_state["draft_loaded_risk"] = {"task_id": task_id, "risk": latest["risk"]}
    else:
        st.session_state.pop("draft_loaded_risk", None)  # 当前已读结果更新，历史提醒仍由 alerts 保存。
    show_risk_notice({"risk": latest.get("risk", {})})  # 规则结果不是模型回答，不会被聊天 reset 清空。
    for alert in latest.get("alerts", []):  # 读取持久通知，重新登录后仍能看到已触发提醒。
        st.warning("此前触发的提醒：" + alert["message"])
        st.caption("提醒时间：" + alert["created_at"])
    recent_risk = st.session_state.get("draft_risk_event")  # 通知保存失败时也保留刚收到的就医提示。
    # 保留刚才未保存成功的通知状态，即使内容与规则快照相同也要显示失败提示。
    if recent_risk and recent_risk.get("task_id") == task_id and (recent_risk.get("risk") != latest.get("risk") or recent_risk.get("notification_saved") is False):
        show_risk_notice(recent_risk)
    # pop 让提交回执只显示一次，刷新页面不会触发第二次提交。
    submitted = st.session_state.pop("draft_submit_notice", None)
    if submitted is not None:
        st.success("本次随访已提交，记录已保存。")  # 不把提交说成医生已经审核或发布报告。
    ready = not latest["missing_fields"]
    if not ready and not st.session_state.get("draft_manual") and not draft_has_changes():
        if st.button("手工填写", key="open_manual_form", type="tertiary"):
            st.session_state["draft_manual"] = True
            st.rerun()
        return
    for note in latest.get("notes", []):  # 例如旧记录只写“已服药”，尚不能确定是否按时。
        st.info(note)  # 只展示核对提示，输入框仍保持未说明，直到患者自己确认。
    if st.session_state.pop("draft_saved_notice", False):
        st.success("草稿已保存，可继续补充或修改。")
    # 如果后端已经保存新版本，提示核对；当前页面的个人编辑先保留。
    if latest["revision"] != base["revision"]:
        st.warning("草稿已有更新，当前输入仍保留。请先核对，再放弃当前修改并重新加载，避免覆盖新内容。")
    if draft_has_changes():
        st.caption("当前有尚未保存的修改。可以先问助手，回到这里继续；采集和提交前请先保存。")
    # 已经齐全时先呈现可读的摘要；只有需要修正时才展开整张表单。
    if ready:
        show_saved_draft_summary(latest["data"])
    elif latest["revision"]:
        # 恢复部分草稿时，患者不用展开整张表格就能知道还要告诉助手哪些信息。
        st.write("还需补充　" + "、".join(latest["missing_fields"]))
    with st.expander("核对随访信息 · 手工填写或修改", expanded=draft_has_changes() or not ready):
        for field, value in st.session_state["draft_edits"].items():
            st.session_state["draft_field_" + field] = value
        st.caption("还没说到的信息可以留空。修改后请先保存，再确认提交。")
        can_discard = draft_has_changes() or latest["revision"] != base["revision"]
        if can_discard and st.button("放弃当前修改并重新加载", key="draft_discard"):
            load_draft_fields(latest)  # 用户已明确选择放弃，才替换输入及版本。
            st.rerun()
        if latest["missing_fields"] and not latest["revision"]:
            # 缺项由服务器按“已测量”“有漏服”等条件计算，页面直接列给患者。
            st.write("已保存草稿还需补充：" + "、".join(latest["missing_fields"]))
        editable = latest["editable"]  # 即使控件不可编辑，后端保存接口仍要独立校验状态。
        if not editable:
            st.info("这项任务当前不能修改草稿，下面仅供查看。")
        st.selectbox("这次测过血压了吗？", list(DRAFT_CHOICES["bp_measurement_status"]),
                     key="draft_field_bp_measurement_status", disabled=not editable,
                     on_change=remember_draft_field, args=("bp_measurement_status",))
        for field, label in DRAFT_TEXT_FIELDS.items():
            # 每个文本框用字段名作 key，保存时可以精确取回相应输入。
            st.text_input(label, key="draft_field_" + field, disabled=not editable,
                          on_change=remember_draft_field, args=(field,))
        st.selectbox("最近服药情况", list(DRAFT_CHOICES["medication_adherence"]),
                     key="draft_field_medication_adherence", disabled=not editable,
                     on_change=remember_draft_field, args=("medication_adherence",))
        if st.button("保存草稿", key="draft_save", disabled=not editable):
            data = {}  # 空输入转换为 None；接口 JSON 中对应 null。
            try:
                for field in DRAFT_TEXT_FIELDS:
                    # 去掉首尾空格再判断是否为空，患者可以有意清空之前填过的字段。
                    value = st.session_state["draft_field_" + field].strip()
                    data[field] = None
                    if value:
                        data[field] = value  # 时间、症状和原因保留文字，不自行推算日期或医学结论。
                        if field in ("systolic_bp", "diastolic_bp"):
                            data[field] = int(value)  # 无法转成整数时明确报错，不写成 0。
                for field, choices in DRAFT_CHOICES.items():
                    # 例如把页面选择“还没测”转换为 not_measured。
                    data[field] = choices[st.session_state["draft_field_" + field]]
            except ValueError:
                st.error("高压和低压请填整数；还不知道时留空即可。")
                return  # 输入仍保留在控件中，患者可以修改。
            try:
                # PUT 提交完整可编辑表单；与聊天只提议本轮提到的字段有所区别。
                api_client.request_api("PUT", path, token=token, data={
                    "expected_revision": base["revision"], "data": data,
                })  # 例如 revision=2；如果服务器已是 3，必须拒绝覆盖。
            except requests.RequestException as error:
                if error.response is not None and error.response.status_code in (401, 403, 404):
                    clear_task_selection()  # 写入阶段也可能撤权，立即清除旧草稿和风险缓存。
                    show_request_error(error)
                    st.stop()
                show_failed_draft_risk(error, task_id)  # 规则提醒独立于写库成功，不被保存失败吞掉。
                show_request_error(error)
                # 409 提醒用户重新核对版本，不能自动用过时输入覆盖服务器值。
                if error.response is not None and error.response.status_code == 409:
                    st.warning("未覆盖服务器草稿。当前输入仍保留，请核对后重新加载。")
            else:
                st.session_state["draft_reload"] = True  # 下一轮控件创建前再更新，避免修改已创建的控件。
                st.session_state["draft_saved_notice"] = True
                st.rerun()
        if not ready:
            show_patient_submit(token, task_id, latest, base)  # 缺项时在编辑区说明提交条件。
    if ready:
        show_patient_submit(token, task_id, latest, base)  # 齐全后直接在摘要下方确认，仍需勾选。
