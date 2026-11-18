"""首轮随访排期、开放和逾期通知；只读写本地业务库，不调用模型。"""

from contextlib import closing
from datetime import date, datetime, time, timedelta, timezone
import logging
from uuid import uuid4

from utils.user_store import get_connection


BEIJING = timezone(timedelta(hours=8))  # 业务约定以北京时间的日历日期排期。
OPEN_HOUR = 9
FILL_WINDOW = timedelta(hours=48)
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"  # SQLite 内统一用本地时间，固定宽度可直接比较。


def beijing_now(now=None):
    """返回北京时间。可传带时区的 datetime 复现扫描，传入无时区的值抛 ValueError；不读写库。"""
    if now is None:
        return datetime.now(BEIJING)
    if not isinstance(now, datetime) or now.utcoffset() is None:
        raise ValueError("排期时钟需要带时区的日期时间。")
    return now.astimezone(BEIJING)


def beijing_time_text(now=None):
    """把当前时钟或指定的带时区 datetime 转成库内时间文本；非法时刻抛 ValueError，不读写库。"""
    return beijing_now(now).strftime(TIME_FORMAT)


def migrate_schedule():
    """无入参、无返回值。首次升级给旧出院记录划定边界；重复运行保留边界，失败时整体回滚。"""
    with closing(get_connection()) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")  # 其他写入须等到新旧记录已区分后才能开始。
        columns = conn.execute("SELECT name FROM pragma_table_info('discharges') WHERE name='auto_schedule'").fetchone()
        if columns is None:
            # 新登记的出院记录默认启用，升级时库里已有的记录一次性排除。
            conn.execute("ALTER TABLE discharges ADD COLUMN auto_schedule INTEGER NOT NULL DEFAULT 1 CHECK(auto_schedule IN (0,1))")
            conn.execute("UPDATE discharges SET auto_schedule=0")
        task_fields = set()
        for row in conn.execute("PRAGMA table_info(followup_tasks)"):
            task_fields.add(row["name"])
        if "scheduled_at" not in task_fields:
            conn.execute("ALTER TABLE followup_tasks ADD COLUMN scheduled_at TEXT")  # NULL 代表既有流程。
        if "opened_at" not in task_fields:
            conn.execute("ALTER TABLE followup_tasks ADD COLUMN opened_at TEXT")  # 到点并真正发布后才填写。
        if "window_revision" not in task_fields:
            conn.execute("ALTER TABLE followup_tasks ADD COLUMN window_revision INTEGER NOT NULL DEFAULT 0")


def plan_new_discharges(conn):
    """在扫描事务里为新出院记录添加第七天任务，返回新增项数；缺医生留待下轮，日期错误单独记录。"""
    rows = conn.execute(
        """SELECT h.discharge_id,h.discharge_date,p.doctor_id
           FROM discharges h JOIN patients p ON p.patient_id=h.patient_id
           JOIN doctors d ON d.doctor_id=p.doctor_id
           JOIN users du ON du.user_id=d.user_id AND du.role='doctor'
           JOIN users pu ON pu.user_id=p.user_id AND pu.role='patient'
           WHERE h.auto_schedule=1 AND NOT EXISTS
             (SELECT 1 FROM followup_tasks t WHERE t.discharge_id=h.discharge_id AND t.task_type='D7')"""
    ).fetchall()
    count = 0
    for row in rows:
        try:
            discharged = date.fromisoformat(row["discharge_date"])  # 出院当天为第 0 天。
            day_seven = discharged + timedelta(days=7)
            scheduled = datetime.combine(day_seven, time(OPEN_HOUR), BEIJING)
            expected_deadline = scheduled + FILL_WINDOW
        except (TypeError, ValueError, OverflowError):
            logging.getLogger(__name__).warning("出院日期无法排期，记录编号 %s", row["discharge_id"])
            continue  # 一条坏日期不阻碍其他患者；修正后下一轮会重新处理。
        conn.execute(
            """INSERT INTO followup_tasks(task_id,discharge_id,doctor_id,task_type,followup_date,
                                          scheduled_at,window_revision,expires_at)
               VALUES(?,?,?,'D7',?,?,1,?)""",
            (str(uuid4()), row["discharge_id"], row["doctor_id"], day_seven.isoformat(),
             scheduled.strftime(TIME_FORMAT), expected_deadline.strftime(TIME_FORMAT)),
        )  # 原表要求截止时间非空；任务实际开放时再按实际时钟写入完整的 48 小时。
        count += 1
    return count


def save_notice(conn, kind, task_id, revision, recipient, message, created_at):
    """在调用方事务里写一条站内通知，首次成功返回 True。唯一键按类型、任务、轮次和接收人去重。"""
    event_key = f"schedule:{kind}:{task_id}:{revision}:{recipient}"  # 医生改派后新接收人有独立通知。
    cursor = conn.execute(
        """INSERT INTO notifications(user_id,task_id,message,event_key,created_at)
           SELECT ?,?,?,?,? WHERE NOT EXISTS (SELECT 1 FROM notifications WHERE event_key=?)""",
        (recipient, task_id, message, event_key, created_at, event_key),
    )
    return cursor.rowcount == 1  # 已有事件不再分配通知编号。库异常交由调用方整体回滚。


def scan_schedule(now=None):
    """扫描一次并写库。可传带时区时钟，返回本次新增 planned/opened/overdue 计数；库错误回滚。"""
    with closing(get_connection()) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")  # 与提交、退回和其他扫描串行，以拿到锁后的实际开放时间为准。
        current = beijing_now(now)
        stamp = current.strftime(TIME_FORMAT)
        deadline = (current + FILL_WINDOW).strftime(TIME_FORMAT)
        result = {"planned": plan_new_discharges(conn), "opened": 0, "overdue": 0}
        # 同一患者有多次住院：按任务关联的出院记录取患者账号，按任务关联的医生取负责账号。
        assignments = """FROM followup_tasks t
            JOIN discharges h ON h.discharge_id=t.discharge_id
            JOIN patients p ON p.patient_id=h.patient_id
            JOIN doctors d ON d.doctor_id=t.doctor_id
            JOIN users pu ON pu.user_id=p.user_id AND pu.role='patient'
            JOIN users du ON du.user_id=d.user_id AND du.role='doctor'"""
        rows = conn.execute(
            "SELECT t.task_id,t.window_revision,p.user_id AS patient " + assignments +
            " WHERE t.scheduled_at<=? AND t.opened_at IS NULL AND t.task_status='pending'", (stamp,),
        ).fetchall()
        for row in rows:
            task_id = row["task_id"]
            conn.execute("UPDATE followup_tasks SET opened_at=?,expires_at=? WHERE task_id=?", (stamp, deadline, task_id))
            message = "出院后首次随访已开放，请在" + deadline + "前完成填写。"
            save_notice(conn, "opened", task_id, row["window_revision"], row["patient"], message, stamp)
            result["opened"] += 1  # 开放与通知一起提交，进程中断后也可重试。
        late_rows = conn.execute(
            "SELECT t.task_id,t.window_revision,p.user_id AS patient,p.name,d.user_id AS doctor " + assignments +
            """ WHERE t.scheduled_at IS NOT NULL AND t.opened_at IS NOT NULL AND t.expires_at<=?
                AND t.task_status IN ('pending','in_progress','needs_info')""", (stamp,),
        ).fetchall()
        for row in late_rows:
            task_id = row["task_id"]
            revision = row["window_revision"]
            save_notice(conn, "overdue", task_id, revision, row["patient"], "本次随访已逾期，仍可继续填写并补交。", stamp)
            message = row["name"] + "的随访已逾期未交，可打开档案联系患者。"
            if save_notice(conn, "overdue", task_id, revision, row["doctor"], message, stamp):
                result["overdue"] += 1  # 只数本次产生的医生待办，不按每分钟重复计数。
        return result


def reset_return_window(conn, task_id, now=None):
    """医生退回后在同一审核事务内重置自动任务的期限并通知患者；旧任务跳过，不单独提交。"""
    row = conn.execute(
        """SELECT t.window_revision,p.user_id FROM followup_tasks t
           JOIN discharges h ON h.discharge_id=t.discharge_id JOIN patients p ON p.patient_id=h.patient_id
           WHERE t.task_id=? AND t.scheduled_at IS NOT NULL AND t.opened_at IS NOT NULL
             AND t.task_status='needs_info'""", (task_id,),
    ).fetchone()
    if row is None:
        return  # 医生审核入口已检查状态和版本，这里只给已开放的自动任务开始补充期。
    current = beijing_now(now)
    revision = row["window_revision"] + 1  # 原通知保留，读取当前待办时将排除旧轮次。
    deadline = (current + FILL_WINDOW).strftime(TIME_FORMAT)
    conn.execute("UPDATE followup_tasks SET window_revision=?,expires_at=? WHERE task_id=?", (revision, deadline, task_id))
    message = "医生已退回本次随访，请查看补充意见，在" + deadline + "前重新提交。"
    save_notice(conn, "returned", task_id, revision, row["user_id"], message, current.strftime(TIME_FORMAT))


def list_schedule_notifications(user_id, role):
    """按已登录的身份读取本轮待办，各任务只返回一条。提交后隐藏，非法角色抛 PermissionError。"""
    if role not in ("patient", "doctor"):
        raise PermissionError("只有医生或患者可以查看随访提醒。")
    with closing(get_connection()) as conn:
        conn.execute("BEGIN")  # 账号、归属和状态使用同一份快照。
        account = conn.execute("SELECT role FROM users WHERE user_id=?", (user_id,)).fetchone()
        if account is None or account["role"] != role:
            raise PermissionError("无权查看随访提醒。")
        rows = conn.execute(
            """SELECT n.id,n.task_id,n.event_key,n.message,n.created_at,t.window_revision,p.patient_id,p.name AS patient_name
               FROM notifications n JOIN followup_tasks t ON t.task_id=n.task_id
               JOIN discharges h ON h.discharge_id=t.discharge_id JOIN patients p ON p.patient_id=h.patient_id
               JOIN doctors d ON d.doctor_id=t.doctor_id
               WHERE n.user_id=:user AND n.event_key LIKE 'schedule:%' AND t.opened_at IS NOT NULL
                 AND t.task_status IN ('pending','in_progress','needs_info')
                 AND ((:role='patient' AND p.user_id=:user) OR (:role='doctor' AND d.user_id=:user))
               ORDER BY n.id DESC""", {"user": user_id, "role": role},
        ).fetchall()
        notices = []
        visited = set()
        for row in rows:
            notice = dict(row)
            task_id = notice["task_id"]
            if task_id in visited:
                continue  # 例如逾期通知已经覆盖原来的“已开放”通知。
            event_key = notice.pop("event_key")
            revision = notice.pop("window_revision")
            kind = event_key.split(":", 2)[1]  # 只识别本模块使用的三种待办。
            if kind not in ("opened", "returned", "overdue"):
                continue
            if event_key != f"schedule:{kind}:{task_id}:{revision}:{user_id}":
                continue  # 旧补充期的通知仍留库，只展示当前轮次和当前接收人的事件。
            notice["kind"] = kind
            notices.append(notice)
            visited.add(task_id)
        return notices
