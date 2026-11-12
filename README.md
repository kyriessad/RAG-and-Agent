# 智能扫地机器人客服 Agent

基于 Streamlit、LangChain 和 Chroma 的中文客服演示项目，支持知识库问答和模拟设备使用报告。模型通过阿里云百炼的 OpenAI 兼容接口调用。

天气、位置、用户 ID 和月份工具当前使用固定值或随机值；数据：`data/external/records.csv` 。

## 环境与安装

使用 Python 3.11。当前依赖中的 LangChain 和 Streamlit 要求 Python 至少为 3.11。

在项目根目录打开 PowerShell。创建独立环境并安装依赖：

```powershell
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip check
```

## API key

代码在 `model/factory.py` 中读取环境变量 `DASHSCOPE_API_KEY`。

在Windows 环境变量中设置 `DASHSCOPE_API_KEY`，使用自己的密钥。模型名称和服务地址位于 `config/rag.yml`，应与该密钥有权使用的服务匹配。

## 激活虚拟环境
```powershell
venv\Scripts\activate
```

## 导入知识库和启动

在项目根目录执行以下命令。首次运行、知识库文本更新后，需要导入资料：

```powershell
python.exe -m rag.vector_store
```

这会读取 `data/` 中的知识库文件，调用嵌入模型并生成本地 Chroma 数据库及 `md5.txt`。可在控制台或 `logs/`检查。

启动页面：

项目根目录执行
```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

始终从项目根目录启动。若需要重建知识库，停用应用，删除 `chroma_db/` 和 `md5.txt`，然后重新导入即可。
只清理数据库却保留 MD5 记录，会让导入程序错误地跳过资料。



