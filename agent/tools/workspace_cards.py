"""随访问答中的业务卡片。卡片仅指向服务端已查询的对象，实际办理走原业务接口。"""

from datetime import date, timedelta
import re
from uuid import uuid4

from utils import followup_store, report_store, schedule_store
from utils.risk_store import list_doctor_notifications


def visible_rows(context, query):
    return query(user_id=context["user_id"], role=context["role"],
                 patient_id=context.get("patient_id"), task_id=context.get("task_id"))


def card(kind, row, *fields):
    keys = ("patient_id", "patient_name", "task_id", "discharge_id", *fields)
    return {"kind": kind, **{key: row[key] for key in keys if row.get(key) is not None}}


def add_cards(context, cards):
    # 工具可能并行执行。相同对象同一种操作只保留本轮最新的一张。
    with context["draft_lock"]:
        collected = context.setdefault("cards", {})
        for item in cards:
            key = tuple(item.get(field) for field in
                        ("kind", "patient_id", "task_id", "discharge_id", "notice_id"))
            collected[key] = item


def task_card(task, kind="task", role="patient"):
    item = card(kind, task, "task_type", "followup_date", "task_status", "report_status",
                "scheduled_at", "opened_at", "published_at", "is_overdue",
                "draft_needs_clarification", "draft_revision")
    item["available"] = role == "doctor" or not task.get("scheduled_at") or bool(task.get("opened_at"))
    return item


def selected_people(context, name):
    people = visible_rows(context, followup_store.list_patients)
    if name:
        people = [person for person in people if person["patient_name"] == name]
    return people


def choose_patient(context, people):
    choices = [card("patient", person, "age", "gender") for person in people]
    add_cards(context, choices)
    message = "请医生点选具体患者后继续。" if choices else "当前范围内查不到匹配的患者。"
    return {"choose_patient": choices, "message": message}


def patient_scope(context, patient_name):
    if patient_name:
        people = selected_people(context, patient_name)
        if len(people) != 1:
            return context, choose_patient(context, people)
        return {**context, "patient_id": people[0]["patient_id"]}, None
    return context, None


def task_cards(context, selection, patient_name):
    context, choices = patient_scope(context, patient_name)
    if choices is not None:
        return choices
    rows = visible_rows(context, followup_store.list_followup_plans)
    if selection == "review":
        rows = [row for row in rows if row["task_status"] in ("submitted", "in_review")]
    elif selection == "overdue":
        rows = [row for row in rows if row.get("is_overdue")]
    add_cards(context, [task_card(row, role=context["role"]) for row in rows])
    return rows


def reports(context, patient_name, task_id, latest, action, note, summary):
    context, choices = patient_scope(context, patient_name)
    if choices is not None:
        return choices
    rows = visible_rows(context, followup_store.list_followup_tasks)
    if task_id:
        rows = [task for task in rows if task["task_id"] == task_id]
    if context["role"] == "patient" or (latest and not action):
        if action:
            return {"message": "只有负责医生可以审核或发布报告。"}
        rows = [task for task in rows if task["report_status"] == "published"]
    elif action:
        state = "approved" if action == "publish" else "pending_review"
        rows = [task for task in rows if task["report_status"] == state]
    elif not task_id and not context.get("task_id"):
        rows = [task for task in rows if task["report_status"]]
    if not rows:
        return {"reports": [], "message": "当前范围内没有符合条件的报告。"}
    ordered = sorted(rows, key=lambda t: (t.get("published_at") or t.get("report_date")
                                          or t["followup_date"], t["task_id"]), reverse=True)
    if len(ordered) > 1 and (not latest or action):
        choices = [task_card(task, "report", context["role"]) for task in ordered]
        add_cards(context, choices)
        return {"reports": choices, "message": "请点选要查看或办理的随访报告。"}
    task = ordered[0]
    view = report_store.report_view(context, task["task_id"])
    item = task_card(task, "report", context["role"])
    if view.get("report_id"):
        item.update(report_id=view["report_id"], expected_version=view["version"])
    if action:
        fields = {"action": action}
        for key, value in (("note", note), ("summary", summary)):
            if value and value in context["patient_message"]:
                fields[key] = value
        item["fields"] = fields
    add_cards(context, [item])
    return {**view, "task_id": task["task_id"], "patient_name": task["patient_name"]}


def notifications(context, kind, status="all"):
    query = list_doctor_notifications if kind == "risk" else schedule_store.list_schedule_notifications
    rows = query(context["user_id"], context["role"])
    rows = [row for row in rows if all(context.get(key) in (None, row.get(key))
                                        for key in ("task_id", "patient_id"))]
    if kind == "risk":
        if status == "unhandled":
            rows = [row for row in rows if not row["handled_at"]]
        elif status == "handled":
            rows = [row for row in rows if row["handled_at"]]
        add_cards(context, [dict(card("risk_notice", row, "message", "created_at", "handled_at"),
                                 notice_id=row["id"]) for row in rows])
    else:
        by_id = {task["task_id"]: task for task in visible_rows(context, followup_store.list_followup_tasks)}
        add_cards(context, [task_card(by_id[row["task_id"]], role=context["role"])
                            for row in rows if row["task_id"] in by_id])
    return rows


def stated_date(value, statement):
    """只转换医生本轮明确给出的出院日期。"""
    if not value or value not in statement:
        return None
    today = schedule_store.beijing_now().date()
    relative = {"今天": 0, "昨天": 1, "前天": 2}
    if value in relative:
        return (today - timedelta(days=relative[value])).isoformat()
    match = re.fullmatch(r"(\d{4})[-年/.](\d{1,2})[-月/.](\d{1,2})日?", value)
    if match:
        try:
            return date(*map(int, match.groups())).isoformat()
        except ValueError:
            return None
    return None


def discharge_cards(context, operation, patient_name, discharge_id, proposed, new_registration=False):
    if context["role"] != "doctor":
        return {"message": "此操作需要主管医生办理。"}
    message = context["patient_message"]
    new_fields = {key: value for key, value in proposed.items()
                  if key != "discharge_date" and value and value in message}
    normalized = stated_date(proposed.get("discharge_date"), message)
    if normalized:
        new_fields["discharge_date"] = normalized
    if operation == "supplement_discharge":
        new_fields = {k: v for k, v in new_fields.items()
                      if k in ("discharge_instructions", "discharge_medications")}
    unassigned = context.get("unassigned_discharge") or {}
    if unassigned.get("operation") == operation and not new_registration:
        new_fields = {**unassigned["fields"], **new_fields}
    previous = context.get("previous_cards", [])
    previous = [item for item in previous if item["kind"] == operation]
    if not patient_name and not context.get("patient_id"):
        names = {item["patient_name"] for item in previous}
        if len(names) != 1:
            context["awaiting_patient"] = {"operation": operation, "fields": new_fields}
            return {"missing": ["患者姓名"], "message": "请先询问要为哪位患者办理。"}
        patient_name = names.pop()
    if new_registration:
        previous = []
    people = selected_people(context, patient_name)
    if not people:
        return {"message": "未找到当前可办理的患者。"}
    matching = [p for p in people if p["is_primary"]]
    if not matching:
        return {"message": "此项业务仅限该患者的主管医生办理。"}
    items = []
    for person in matching:
        base = card(operation, person, "age", "gender")
        targets = [base]
        if operation == "supplement_discharge":
            scoped = {**context, "patient_id": person["patient_id"]}
            targets = [{**base, **card(operation, row, "discharge_date")}
                       for row in visible_rows(scoped, followup_store.list_discharges)
                       if (len(people) > 1 or not discharge_id or row["discharge_id"] == discharge_id)
                       and any(not (row[key] or "").strip() for key in
                               (new_fields or ("discharge_instructions", "discharge_medications")))]
        for item in targets:
            prior = next((p for p in previous if p["patient_id"] == item["patient_id"]
                          and p.get("discharge_id") == item.get("discharge_id")), {})
            if operation == "register_discharge" and prior:
                existing = followup_store.list_discharges(context["user_id"], "doctor", patient_id=item["patient_id"])
                if any(row["discharge_id"] == "DIS-" + prior["request_id"] for row in existing):
                    prior = {}
            item["fields"] = {**prior.get("fields", {}), **new_fields}
            if operation == "register_discharge":
                item["request_id"] = prior.get("request_id") or str(uuid4())
            items.append(item)
    if not items:
        return {"message": "没有符合条件且可以补充的出院记录。"}
    add_cards(context, items)
    missing = []
    if operation == "register_discharge":
        for key, label in (("discharge_date", "出院日期"), ("discharge_diagnosis", "出院诊断")):
            if any(not item["fields"].get(key) for item in items):
                missing.append(label)
    elif not any(item["fields"] for item in items):
        missing.append("要补充的医嘱或用药原文")
    return {"missing": missing, "choose_target": len(people) > 1 or len(items) > 1,
            "patients": [p["patient_name"] for p in matching], "message": "待医生核对卡片并确认。"}
