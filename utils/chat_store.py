"""账号对话、原始答复及业务回执。数据库范围是总对话、单患者或单任务。"""
import json
from contextlib import closing
from uuid import uuid4

from utils.user_store import get_connection


class ChatAccessError(PermissionError):
    """会话或其关联病历已不属于当前访问范围。"""


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def ensure_table(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS conversation_turns (
        id INTEGER PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(user_id),
        scope_key TEXT NOT NULL, patient_id TEXT, task_id TEXT, mode TEXT NOT NULL,
        turn_key TEXT NOT NULL, question TEXT, answer TEXT, metadata TEXT NOT NULL DEFAULT '{}',
        permissions TEXT NOT NULL, draft_start INTEGER, draft_end INTEGER,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
        UNIQUE(user_id,turn_key))""")
    conn.execute("""CREATE INDEX IF NOT EXISTS conversation_history
                    ON conversation_turns(user_id,scope_key,id)""")


def migrate_chat():
    """旧 chat_messages 只有任务号和发言角色，消息与登录账号之间没有可核验的关联。"""
    with closing(get_connection()) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        ensure_table(conn)


def permitted_scope(conn, user_id, role, patient_id=None, task_id=None):
    """在同一次 SQLite 快照里核对账号、主管关系及当前任务归属。"""
    if role not in ("doctor", "patient"):
        raise ChatAccessError("当前账号无权查看对话。")
    args = {"uid": user_id, "role": role, "pid": patient_id, "tid": task_id}
    task_rows = conn.execute("""SELECT t.task_id,d.patient_id,d.discharge_id FROM followup_tasks t
        JOIN discharges d ON d.discharge_id=t.discharge_id JOIN patients p ON p.patient_id=d.patient_id
        JOIN doctors dr ON dr.doctor_id=t.doctor_id JOIN users u ON u.user_id=:uid AND u.role=:role
        WHERE ((:role='patient' AND p.user_id=:uid AND (t.scheduled_at IS NULL OR t.opened_at IS NOT NULL))
               OR (:role='doctor' AND dr.user_id=:uid))
          AND (:pid IS NULL OR d.patient_id=:pid) AND (:tid IS NULL OR t.task_id=:tid)""", args).fetchall()
    permissions = {("task", *tuple(row)) for row in task_rows}
    if task_id:
        if not task_rows:
            raise ChatAccessError("任务不存在或无权访问。")
        permissions.add(("role", role))
        return "task:" + task_id, task_rows[0]["patient_id"], permissions
    people = conn.execute("""SELECT p.patient_id FROM patients p
        LEFT JOIN doctors dr ON dr.doctor_id=p.doctor_id
        JOIN users u ON u.user_id=:uid AND u.role=:role
        WHERE ((:role='patient' AND p.user_id=:uid) OR (:role='doctor' AND dr.user_id=:uid))
        AND (:pid IS NULL OR p.patient_id=:pid)""", args)
    permissions.update(("primary", row["patient_id"]) for row in people)
    if patient_id and not permissions:
        raise ChatAccessError("患者不存在或无权访问。")
    permissions.add(("role", role))
    return "patient:" + patient_id if patient_id else "general", patient_id, permissions


def visible(row, permissions):
    return all(tuple(entry) in permissions for entry in json.loads(row["permissions"]))


def has_history_table(conn):
    return conn.execute("""SELECT 1 FROM sqlite_master
                           WHERE type='table' AND name='conversation_turns'""").fetchone() is not None


def rows_to_messages(rows):
    messages = []
    for row in rows:
        meta = json.loads(row["metadata"])
        common = {"created_at": row["created_at"], "sources": None, "cards": [], "risk": None, "receipt": None}
        if row["question"] is not None:
            messages.append(dict(common, id=f"{row['id']}-u", role="user", content=row["question"],
                                 status="complete" if row["answer"] else "interrupted"))
        if row["answer"] is not None or meta.get("risk"):
            messages.append({**common, **meta, "id": f"{row['id']}-a", "role": "assistant",
                             "content": row["answer"] or "", "status": "complete" if row["answer"] else "interrupted"})
    return messages


def history(user_id, role, patient_id=None, task_id=None):
    with closing(get_connection()) as conn:
        conn.execute("BEGIN")
        scope, _, permissions = permitted_scope(conn, user_id, role, patient_id, task_id)
        if not has_history_table(conn):
            return []
        rows = conn.execute("""SELECT * FROM conversation_turns WHERE user_id=? AND scope_key=?
                               ORDER BY created_at,id""", (user_id, scope))
        return rows_to_messages(row for row in rows if visible(row, permissions))


def conversations(user_id, role):
    with closing(get_connection()) as conn:
        conn.execute("BEGIN")
        _, _, permissions = permitted_scope(conn, user_id, role)
        if not has_history_table(conn):
            return []
        rows = conn.execute("""SELECT m.*,p.name,t.task_type,t.followup_date FROM conversation_turns m
            LEFT JOIN patients p ON p.patient_id=m.patient_id LEFT JOIN followup_tasks t ON t.task_id=m.task_id
            WHERE m.user_id=? ORDER BY m.created_at DESC,m.id DESC""", (user_id,))
        found = {}
        for row in rows:
            key = row["scope_key"]
            if key in found or not visible(row, permissions):
                continue
            kind = key.split(":", 1)[0]
            title = "总对话"
            if kind == "patient":
                title = row["name"] or "患者对话"
            if kind == "task":
                task_type = row["task_type"] or "随访"
                if task_type.startswith("D") and task_type[1:].isdigit():
                    task_type = f"第 {task_type[1:]} 天随访"
                title = f"{row['name']} · {row['followup_date']} · {task_type}"
            found[key] = dict(id=key, scope=kind, title=title, patient_id=row["patient_id"],
                              task_id=row["task_id"], updated_at=row["created_at"])
        return list(found.values())


def start_turn(user_id, role, patient_id, task_id, mode, text, request_id):
    with closing(get_connection()) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        ensure_table(conn)
        scope, pid, permissions = permitted_scope(conn, user_id, role, patient_id, task_id)
        key = "chat:" + str(request_id or uuid4())
        row = conn.execute("SELECT * FROM conversation_turns WHERE user_id=? AND turn_key=?", (user_id, key)).fetchone()
        if row:
            if row["scope_key"] != scope or row["question"] != text or row["mode"] != mode or not visible(row, permissions):
                raise ValueError("这条消息标识已使用，请重新发送。")
            if row["answer"] is None:
                raise ValueError("这条消息尚无完整答复，请刷新对话后重新发送。")
            events = [{"type": name, "content": value} for name, value in json.loads(row["metadata"]).items()
                      if name in ("sources", "risk", "cards") and value is not None]
            return row["id"], events + [{"type": "answer", "content": row["answer"], "message_id": f"{row['id']}-a"}]
        cursor = conn.execute("""INSERT INTO conversation_turns
            (user_id,scope_key,patient_id,task_id,mode,turn_key,question,permissions)
            VALUES (?,?,?,?,?,?,?,?)""", (user_id, scope, pid, task_id, mode, key, text, encode(sorted(permissions))))
        return cursor.lastrowid, None


def save_risk_event(turn_id, metadata):
    with closing(get_connection()) as conn, conn:
        conn.execute("UPDATE conversation_turns SET metadata=? WHERE id=?", (encode(metadata), turn_id))


def finish_turn(conn, user_id, role, turn_id, metadata, answer, draft_start, draft_end):
    row = conn.execute("SELECT * FROM conversation_turns WHERE id=? AND user_id=?", (turn_id, user_id)).fetchone()
    if row is None:
        raise ChatAccessError("对话已无法访问。")
    _, _, access = permitted_scope(conn, user_id, role, row["patient_id"], row["task_id"])
    if not visible(row, access):
        raise ChatAccessError("查看范围已变更，请重新打开对话。")
    conn.execute("""UPDATE conversation_turns SET metadata=?,answer=?,draft_start=?,draft_end=?,permissions=?
        WHERE id=?""", (encode(metadata), answer, draft_start, draft_end, encode(sorted(access)), turn_id))


def recent_turns(user_id, role, patient_id, task_id, mode, before_id, draft_revision):
    with closing(get_connection()) as conn:
        conn.execute("BEGIN")
        scope, _, permissions = permitted_scope(conn, user_id, role, patient_id, task_id)
        rows = conn.execute("""SELECT * FROM conversation_turns WHERE user_id=? AND scope_key=?
            AND (mode=? OR (?='query' AND mode='receipt')) AND id<? AND answer IS NOT NULL
            ORDER BY id DESC LIMIT 12""", (user_id, scope, mode, mode, before_id))
        recent = []
        for row in rows:
            if not visible(row, permissions) or (mode == "collect" and row["draft_end"] != draft_revision):
                break
            recent.append(row)
            draft_revision = row["draft_start"]
        return permissions, rows_to_messages(reversed(recent))


def save_receipt(conn, user_id, role, action, key, content, *, patient_id=None, task_id=None, **details):
    """随业务事务一并提交。相同办理标识只追加一条回执。"""
    ensure_table(conn)
    scope, pid, permissions = permitted_scope(conn, user_id, role, patient_id, task_id)
    receipt = dict(action=action, patient_id=pid, task_id=task_id, **details)
    conn.execute("""INSERT INTO conversation_turns
        (user_id,scope_key,patient_id,task_id,mode,turn_key,answer,permissions,metadata)
        VALUES (?,?,?,?,'receipt',?,?,?,?) ON CONFLICT(user_id,turn_key) DO NOTHING""",
        (user_id, scope, pid, task_id, f"{action}:{key}", content,
         encode(sorted(permissions)), encode({"receipt": receipt})))


def select_card(user_id, role, message_id, index):
    """从医生已完成的答复中选择一位患者，把服务端保存的那张预填卡接入专属对话。"""
    denied = "该卡片不存在或已无法访问。"
    if role != "doctor":
        raise ChatAccessError(denied)
    with closing(get_connection()) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        ensure_table(conn)
        row = conn.execute("""SELECT * FROM conversation_turns
            WHERE user_id=? AND id=? AND answer IS NOT NULL""", (user_id, message_id)).fetchone()
        if not row:
            raise ChatAccessError(denied)
        _, _, access = permitted_scope(conn, user_id, role, row["patient_id"], row["task_id"])
        cards = json.loads(row["metadata"]).get("cards", [])
        if not visible(row, access) or not 0 <= index < len(cards):
            raise ChatAccessError(denied)
        card = cards[index]
        pid = card.get("patient_id")
        if not pid or card.get("kind") not in ("register_discharge", "supplement_discharge"):
            raise ChatAccessError(denied)
        scope, _, permissions = permitted_scope(conn, user_id, role, pid)
        if ("primary", pid) not in permissions:
            raise ChatAccessError(denied)
        key = f"card:{message_id}:{index}"
        conn.execute("""INSERT INTO conversation_turns
            (user_id,scope_key,patient_id,mode,turn_key,answer,permissions,metadata)
            VALUES (?,?,?,'query',?,?,?,?) ON CONFLICT(user_id,turn_key) DO NOTHING""",
            (user_id, scope, pid, key, "已选定患者，请核对出院资料。",
             encode(sorted(permissions)), encode({"cards": [card]})))
        selected = conn.execute("SELECT id FROM conversation_turns WHERE user_id=? AND turn_key=?", (user_id, key)).fetchone()
        return {"patient_id": pid, "message_id": f"{selected['id']}-a"}
