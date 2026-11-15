# 智能扫地机器人客服 Agent
基于 Streamlit、LangChain 和 Chroma 的中文客服演示项目，支持知识库问答和模拟设备使用报告。模型通过阿里云百炼的 OpenAI 兼容接口调用。
天气、位置、用户 ID 和月份工具当前使用固定值或随机值；数据：`data/external/records.csv` 。

## 安装依赖
```powershell
pip install -r requirements.txt
```
## API key
代码在 `model/factory.py` 中读取环境变量 `DASHSCOPE_API_KEY`。
在Windows 环境变量中设置 `DASHSCOPE_API_KEY`，使用自己的密钥。模型名称和服务地址位于 `config/rag.yml`，应与该密钥有权使用的服务匹配。

## 启动
在项目根目录(虚拟环境)执行以下命令:
```powershell
streamlit run app.py
```
这会读取 `data/` 中的知识库文件，调用嵌入模型并生成本地 Chroma 数据库。可在控制台或 `logs/`检查。

