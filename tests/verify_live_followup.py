"""手动验收入口：python tests/verify_live_followup.py。会实际请求配置的模型并产生调用费用。

从主库只读表结构，独立目录创建三个随访任务。启动真正的 Uvicorn 子进程，
AppTest 运行 app.py 的真实控件，页面只经本地 HTTP 操作后端。退出时删除临时库。
"""

import logging
import os
from pathlib import Path
import secrets
import socket
import sqlite3
import subprocess
import sys
import tempfile
from threading import Thread
import time
from contextlib import closing, contextmanager

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import user_store
from utils.config_handler import agent_conf, chroma_conf, rag_conf


def seed(directory, password):
    """复制表结构，建两名医生、三项任务及一名无任务患者，只在临时库写入。"""
    database = directory / "medical.db"
    with closing(sqlite3.connect(Path(user_store.DB_PATH).as_uri() + "?mode=ro", uri=True)) as source:
        schema = source.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
    with closing(sqlite3.connect(database)) as conn, conn:
        for table in schema:
            conn.execute(table[0])
        digest = user_store.hash_password(password)
        conn.executemany("INSERT INTO users(user_id,name,password_hash,role) VALUES(?,?,?,?)",
                         [("U1", "林宇", digest, "patient"), ("U2", "陈怡", digest, "doctor"),
                          ("U3", "何远", digest, "doctor"), ("U4", "周宁", digest, "patient")])
        conn.executemany("INSERT INTO doctors(doctor_id,user_id,name,department,title) VALUES(?,?,?,?,?)",
                         [("D1","U2","陈怡","全科","主任医师"), ("D2","U3","何远","心内科","主治医师")])
        conn.execute("""INSERT INTO patients(patient_id,user_id,name,age,gender,diagnosis,doctor_id)
                        VALUES('P1','U1','林宇',50,'male','高血压','D1')""")
        conn.execute("""INSERT INTO patients(patient_id,user_id,name,age,gender,diagnosis,doctor_id)
                        VALUES('P2','U4','周宁',54,'female','高血压','D1')""")
        conn.execute("""INSERT INTO discharges(discharge_id,patient_id,discharge_date,discharge_diagnosis)
                        VALUES('H1','P1','2026-10-01','高血压')""")
        for n, stage in enumerate(("D7", "D14", "D30"), 1):
            # 第二项任务交由另一位医生负责，使两份已发布报告能核对不同的医生。
            doctor = "D1"
            if stage == "D14":
                doctor = "D2"
            conn.execute("""INSERT INTO followup_tasks(task_id,discharge_id,doctor_id,followup_date,task_type,expires_at)
                         VALUES(?,'H1',?,'2026-11-18',?,'2026-11-24')""", (f"F00{n}", doctor, stage))


@contextmanager
def server(directory, port, unavailable=False):
    """启动本脚本专属的真实 FastAPI 服务，使用独立端口和数据库，结束时只关闭本次子进程。"""
    args = [sys.executable, "-B", __file__, "--serve", str(directory), str(port)]
    if unavailable:
        args.append("--unavailable")  # 此阶段把模型地址指向本机关闭的端口，实际触发连接失败。
    # Chroma 的 SQLite 引擎在 Windows 还需能写临时文件；上传或索引缓存都留在本次目录。
    environment = os.environ.copy()
    for key in ("TEMP", "TMP", "TMPDIR", "SQLITE_TMPDIR"):
        environment[key] = str(directory)
    process = subprocess.Popen(args, cwd=ROOT, env=environment)
    try:
        for _ in range(120):
            if process.poll() is not None:
                raise RuntimeError("验收服务启动失败")
            try:
                if requests.get(f"http://127.0.0.1:{port}/openapi.json", timeout=1).ok:
                    break
            except requests.ConnectionError:
                pass
            time.sleep(0.25)
        else:
            raise RuntimeError("验收服务启动超时")
        yield
    finally:
        stop_file = directory / "stop-server"
        stop_file.touch()  # 由子进程主动退出，Windows 虚拟环境的启动器也能收回所有文件句柄。
        process.wait(timeout=40)
        stop_file.unlink()


def serve():
    """子进程只使用命令行指定的验收库与向量目录，模型凭证沿用环境变量。"""
    import uvicorn
    directory = Path(sys.argv[2])
    user_store.DB_PATH = str(directory / "medical.db")
    chroma_conf["persist_directory"] = str(directory / "chroma")
    # 记录这个独立进程的工具状态，避免把模型凭常识作答误判为读取了本地资料。
    from utils.logger_handler import logger
    logger.addHandler(logging.FileHandler(directory / "service.log", encoding="utf-8"))
    if "--unavailable" in sys.argv:
        rag_conf["BASE_URL"] = "http://127.0.0.1:1/v1"
    service = uvicorn.Server(uvicorn.Config("backend:app", host="127.0.0.1", port=int(sys.argv[3]), access_log=False))
    def wait_for_stop():
        while not (directory / "stop-server").exists():
            time.sleep(0.2)
        service.should_exit = True
    Thread(target=wait_for_stop, daemon=True).start()
    service.run()


def run():
    """通过真实页面和 HTTP 核对正常发布、退回重交、提前升级、重复请求及故障恢复。"""
    from streamlit.testing.v1 import AppTest

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    agent_conf["backend_url"] = base  # 仅当前验收进程的页面指向此服务，不改 YAML。
    password = secrets.token_urlsafe(24)
    def login_page(account, role):
        """输入姓名和角色标签，通过真实页面登录；失败时显示页面错误。"""
        page = AppTest.from_file(str(ROOT / "app.py"), default_timeout=200).run()
        page.radio[0].set_value(role)
        page.text_input[0].input(account)
        page.text_input[1].input(password)
        page.button[0].click().run()
        assert not page.error, screen_text(page, "error")
        assert page.session_state["user"]["username"] == account
        return page

    def api(page, method, path, data=None):
        headers = {"Authorization": "Bearer " + page.session_state["token"]}
        response = requests.request(method, base + path, headers=headers, json=data, timeout=200)
        response.raise_for_status()
        return response.json()

    def click(page, key):
        page.button(key=key).click().run()
        assert not page.exception, page.exception

    def screen_text(page, kind):
        return " ".join(item.value for item in page.get(kind))

    def chat(page, message):
        page.chat_input[0].set_value(message).run()
        assert not page.exception, page.exception
        assert not page.error, screen_text(page, "error")
        assert page.session_state["message"][-1]["role"] == "assistant"

    def doctor_select(page, task):
        """从医生待办点入一项已提交或待审核任务，只读取资料。"""
        click(page, "return_tasks")
        page.radio(key="doctor_section").set_value("待办随访").run()
        click(page, "enter_task_" + task)

    def review(page, task, action, note="", summary=""):
        view = api(page, "GET", f"/tasks/{task}/report")
        key = f"{page.session_state['user']['user_id']}_{view['report_id']}_{view['version']}"
        page.radio(key="choice_" + key).set_value(action).run()
        if summary:
            page.text_area(key="summary_" + key).input(summary)
        page.text_area(key="note_" + key).input(note)
        click(page, "review_" + key)
        return view

    def submit(page, task):
        page.checkbox(key="draft_confirm").check().run()
        assert not page.button(key="draft_submit").disabled
        click(page, "draft_submit")
        # 页面先收到提交回执；模型在后端后台生成报告，轮询真实进度。
        deadline = time.monotonic() + 180
        while True:
            view = api(page, "GET", f"/tasks/{task}/report")
            if view["task_status"] != "submitted" or time.monotonic() >= deadline:
                break
            time.sleep(0.5)
        assert view["task_status"] == "in_review", (view["task_status"], screen_text(page, "warning"))
        assert "content" not in view
        return view

    with tempfile.TemporaryDirectory(prefix="followup-live-") as folder:
        directory = Path(folder)
        seed(directory, password)
        with server(directory, port):
            patient = login_page("林宇", "患者")
            doctor = login_page("陈怡", "医生")
            peer = login_page("何远", "医生")
            empty_patient = login_page("周宁", "患者")
            # 无任务可问助手，医生从名册进入同一份档案；另一位医生仅见自己负责的任务。
            assert "没有随访安排" in screen_text(empty_patient, "info")
            doctor.selectbox(key="patient_filter").select("暂无任务").run()
            click(doctor, "patient_P2")
            assert "周宁" in screen_text(doctor, "title")
            click(doctor, "open_assistant")
            chat(doctor, "请查询当前患者的基本档案，告诉我她的姓名和主管医生。")
            assert "周宁" in doctor.session_state["message"][-1]["content"]
            assert api(doctor, "GET", "/patients/P2")["tasks"] == []
            peer_view = api(peer, "GET", "/patients/P1")
            assert [row["task_id"] for row in peer_view["tasks"]] == ["F002"]
            for path in ("/patients/P2", "/tasks/F001/report"):
                denied = requests.get(base + path, headers={"Authorization": "Bearer " + peer.session_state["token"]}, timeout=5)
                assert denied.status_code == 404
            click(empty_patient, "open_assistant")
            chat(empty_patient, "请检索健康知识库，告诉我家庭血压记录应包含什么，以及资料名称。")
            answer = empty_patient.session_state["message"][-1]["content"]
            assert "血压" in answer and "测量" in answer
            assert "工具rag_search调用成功" in (directory / "service.log").read_text(encoding="utf-8")
            print("通过：无任务患者和医生档案助手可用，两位医生的患者、任务权限独立", flush=True)
            click(patient, "enter_task_F001")
            chat(patient, "我来做这次随访，请问我需要告诉你什么？")
            chat(patient, "今天早上量的，大压128，小压78。")
            first = api(patient, "GET", "/tasks/F001/draft")
            assert first["data"]["systolic_bp"] == 128 and first["data"]["diastolic_bp"] == 78
            assert first["missing_fields"] and "？" in patient.session_state["message"][-1]["content"]
            click(patient, "open_assistant")
            chat(patient, "我刚量的血压是135/85，请查询本次随访由哪位医生负责。")
            assert "陈怡" in patient.session_state["message"][-1]["content"]
            assert api(patient, "GET", "/tasks/F001/draft")["revision"] == first["revision"]
            click(patient, "resume_context")
            chat(patient, "最近没有不舒服。每天都按时服药。吃药后没有不舒服。")
            # 患者核对并修正一次已填数值；仅在模型遗漏时按原话补齐输入。
            patient.text_input(key="draft_field_systolic_bp").input("126")
            patient.text_input(key="draft_field_measured_at").input("今天早上八点")
            patient.text_input(key="draft_field_symptoms").input("最近没有不舒服")
            patient.text_input(key="draft_field_adverse_effects").input("吃药后没有不舒服")
            patient.selectbox(key="draft_field_medication_adherence").select("按时服药")
            click(patient, "draft_save")
            saved = api(patient, "GET", "/tasks/F001/draft")
            submitted = submit(patient, "F001")
            again = api(patient, "POST", "/tasks/F001/submit", {"confirmed": True, "expected_revision": saved["revision"]})
            assert again["already_submitted"] and again["record_id"] == submitted["record_id"]
            doctor_select(doctor, "F001")
            pending = review(doctor, "F001", "通过")
            assert pending["record"]["systolic_bp"] == 126
            assert pending["content"]["sources"] and pending["content"]["recommendations"]
            for source in pending["content"]["sources"]:
                assert source["quote"] in source["excerpt"] and (ROOT / "data" / source["source"]).is_file()
            assert "content" not in api(patient, "GET", "/tasks/F001/report")
            approved = api(doctor, "GET", "/tasks/F001/report")
            click(doctor, f"publish_U2_{approved['report_id']}_{approved['version']}")
            patient.run()
            assert "我的最终随访报告" in screen_text(patient, "subheader")
            print("通过：真实模型分轮采集、患者修改、重复提交、审核隔离、正常发布", flush=True)

            click(patient, "return_tasks")
            click(patient, "enter_task_F002")
            chat(patient, "我今天还没量血压。最近没有不舒服。出院时没有开药。")
            # 人工核对路径也可用于真实模型没有识别到的字段。
            patient.selectbox(key="draft_field_bp_measurement_status").select("还没测")
            patient.selectbox(key="draft_field_medication_adherence").select("没有开药")
            patient.text_input(key="draft_field_symptoms").input("最近没有不舒服")
            click(patient, "draft_save")
            submit(patient, "F002")
            doctor_select(peer, "F002")
            old = review(peer, "F002", "退回补充", note="请补充最近一次测量的时间和读数。")
            patient.run()
            assert "请补充最近一次" in screen_text(patient, "markdown")
            assert "content" not in api(patient, "GET", "/tasks/F002/report")
            patient.selectbox(key="draft_field_bp_measurement_status").select("测过了")
            patient.text_input(key="draft_field_measured_at").input("午饭前")
            patient.text_input(key="draft_field_systolic_bp").input("130")
            patient.text_input(key="draft_field_diastolic_bp").input("82")
            click(patient, "draft_save")
            submit(patient, "F002")
            doctor_select(peer, "F002")
            revised = review(peer, "F002", "修改后通过", summary="患者补充午饭前测量为 130/82 mmHg。")
            assert revised["record"]["revision"] == 2 and old["record_id"] != revised["record_id"]
            approved = api(peer, "GET", "/tasks/F002/report")
            click(peer, f"publish_U3_{approved['report_id']}_{approved['version']}")
            assert api(patient, "GET", "/tasks/F002/report")["content"]["summary"] == "患者补充午饭前测量为 130/82 mmHg。"
            # 从历史入口分别阅读两份已发布报告，核对当时负责的不同医生和职称。
            for task, name, title in (("F001", "陈怡", "主任医师"), ("F002", "何远", "主治医师")):
                click(patient, "return_tasks")
                assert patient.button(key="enter_task_" + task).label == "阅读报告"
                click(patient, "enter_task_" + task)
                detail = api(patient, "GET", f"/tasks/{task}/report")
                assert detail["responsible_doctor"]["name"] == name and detail["reviewer"]["title"] == title
                assert name in screen_text(patient, "markdown") and title in screen_text(patient, "markdown")
            print("通过：真实退回、补充、第二版提交、修改后通过、发布", flush=True)

        with server(directory, port, unavailable=True):
            click(patient, "return_tasks")
            click(patient, "enter_task_F003")
            patient.chat_input[0].set_value("我现在胸痛。").run()
            alerts = api(doctor, "GET", "/notifications")
            assert len(alerts) == 1 and alerts[0]["task_id"] == "F003"
            assert len(patient.error) >= 2  # 独立红旗提醒与聊天失败分别显示为两个红色提示。
            assert "胸痛" in screen_text(patient, "markdown")  # 规则触发原因在模型失败后仍保留。
            click(doctor, "return_tasks")
            doctor.radio(key="doctor_section").set_value("风险通知").run()
            assert "胸痛" in screen_text(doctor, "warning")
            patient.selectbox(key="draft_field_bp_measurement_status").select("还没测")
            patient.selectbox(key="draft_field_medication_adherence").select("没有开药")
            patient.text_input(key="draft_field_symptoms").input("我现在胸痛")
            click(patient, "draft_save")
            patient.checkbox(key="draft_confirm").check().run()
            click(patient, "draft_submit")
            failed = api(patient, "GET", "/tasks/F003/report")
            assert failed["task_status"] == "submitted" and "content" not in failed
            print("通过：模型连接失败仍先升级，医生收到通知，报告失败保留提交", flush=True)

        with server(directory, port):
            doctor_select(doctor, "F003")
            click(doctor, "generate_report")
            recovered = api(doctor, "GET", "/tasks/F003/report")
            assert recovered["record_id"] == failed["record_id"] and recovered["status"] == "pending_review"
            assert recovered["content"]["risk_level"] == "high"
            api(patient, "POST", "/tasks/F003/report/generate", {"record_id": failed["record_id"]})
            assert len(api(doctor, "GET", "/notifications")) == 1
            doctor_select(doctor, "F003")
            review(doctor, "F003", "通过", note="已核对本次上报及升级通知。")
            approved = api(doctor, "GET", "/tasks/F003/report")
            click(doctor, f"publish_U2_{approved['report_id']}_{approved['version']}")
            assert api(patient, "GET", "/tasks/F003/report")["content"]["risk_level"] == "high"
            with closing(sqlite3.connect(directory / "medical.db")) as conn:
                assert conn.execute("SELECT count(*) FROM reports").fetchone()[0] == 4
                assert conn.execute("SELECT count(*) FROM followup_records").fetchone()[0] == 4
                assert conn.execute("SELECT count(*) FROM audit_logs WHERE action='report_generated'").fetchone()[0] == 4
            print("通过：恢复模型后同一记录生成成功，重复点击没有新增报告或审计", flush=True)


if __name__ == "__main__":
    if "--serve" in sys.argv:
        serve()
    else:
        run()
