# 高血压出院随访助手

医生登记出院后，患者会在约定日期收到站内提醒。患者可以用日常语言完成随访，核对血压、症状和服药情况后提交。系统生成带知识库出处的报告，医生审核、修改并发布后，患者就能阅读终稿。

这是个人学习项目。患者档案和知识资料均用于展示，请勿录入真实病历。使用对话和知识检索功能时，相应内容会发往配置的阿里云百炼服务。

## 本地启动

在项目根目录使用 Python 3.11 安装依赖。

```powershell
py -3.11 -m venv venv
.\venv\Scripts\python -m pip install -r requirements.txt
```

将百炼密钥保存在环境变量 `DASHSCOPE_API_KEY` 中，让启动后端的终端读取该变量。模型名称和百炼服务地址位于 `config/rag.yml`。分别在两个终端运行以下命令。

```powershell
.\venv\Scripts\python -m uvicorn backend:app --host 127.0.0.1 --port 8000
```

```powershell
.\venv\Scripts\python -m streamlit run app.py
```

浏览器打开 [本地随访助手](http://localhost:8501)。前端的 API 地址由 `config/agent.yml` 指定。后端使用项目中的 `data/medical.db`，启动时检查表结构、补扫到期任务，运行期间每分钟扫描一次。知识文件放在 `data` 目录，支持 TXT、PDF 和 DOCX。向量索引保存在 `chroma_db`。

## 体验一次完整随访

1. 注册医生并登记联系号码，再注册患者，选择这位医生为主管医生。患者可在侧栏「我的资料」更新自己的电话和既往病史。
2. 医生登录后在总对话输入“帮我登记患者出院”，核对预填卡并确认。系统安排出院后第 7、14、30 天的随访，北京时间上午 9 点开放。登记八天前的出院日期可以体验已到期的首轮任务。
3. 患者在「助手能做什么」选择「查我的随访计划」，从任务卡进入专属对话，填写或修改后确认提交。已保存的聊天与资料可在再次登录后继续查看。
4. 医生在聊天首页查看风险提醒和待审事项，在任务对话里核对原始记录与引用，选择通过、修改后通过或退回补充。审核通过后点击发布，患者可在总对话查询已发布报告。

启动、双账号登录、每一步的按钮和预期结果见[对话首页手工验收](docs/验收/对话首页手工验收.md)。

## 验证

自动化测试使用临时数据库与模型替身，可在项目根目录运行。

```powershell
.\venv\Scripts\python -m unittest discover -s tests -p "test_*.py"
```

真实模型端到端验收使用独立数据库和端口，会产生百炼调用费用。

```powershell
.\venv\Scripts\python tests/verify_live_followup.py
```
