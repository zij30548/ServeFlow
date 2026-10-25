# 智应 ServeFlow

面向复杂客服任务的多Agent协作系统。用户通过对话咨询，系统识别诉求、选择处理角色，按需查阅政策或读取业务工具，并展示处理结果。前端只有**对话**与**监控**两个工作界面。

当前项目已替换原TripPilot旅行应用。它是客服开发／演示项目，尚不具备直接公开给真实客户使用的完整生产安全条件。

## 功能

- 意图识别：LLM、本地模板相似度和关键词融合。
- 角色协作：通用客服、技术支持、账单售后、人工升级；复合问题支持主辅路由和结果合并。
- 知识检索：Chroma中的演示政策，按需查询改写、召回和重排。
- 对话记忆：Redis工作记忆、Chroma历史摘要与画像。
- 工具与Skills：明确工具白名单和参数校验；Skills提供处理规范。
- 监控：读取角色执行、工具、耗时及失败状态；技术执行成功率不等于回答准确率。
- 可选MySQL演示：10笔合成订单，支持核对拆单、权益和退款状态；只读、不执行真实退款。

模型负责理解和组织回复；政策资料与订单记录的职责分开。未知、错误和不确定结果应如实保留，不能将模型回复当作真实交易凭证。

## 目录

```text
backend/
  api/             HTTP接口
  agents/          角色与编排、业务工具
  core/            意图、Skills及通用边界
  tooling/         检索、工具治理与MCP客户端
  memory/          工作记忆与历史记忆
  monitor/         运行指标
  evaluation/      评测实现
  skills/          运行规则
  knowledge/       合成演示政策
  demo_business/   可选MySQL演示
  tests/
frontend/
  src/             Vue界面
  tests/
```

## 本地运行

需要Python **3.12**、Node.js **20.19+或22.12+**、npm，以及可用的Redis。后端沿用Anthropic SDK，通过百炼官方Anthropic兼容接口调用千问；不是依赖Anthropic模型。

### 后端

```sh
cd backend
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt -r requirements-dev.txt
cp .env.example .env
```

在本机`.env`中填写：

- `ANTHROPIC_API_KEY`：百炼密钥，仅供服务端使用。
- `ANTHROPIC_BASE_URL`：北京地域官方Anthropic兼容地址，例如`https://dashscope.aliyuncs.com/apps/anthropic`；业务空间地址按控制台／官方说明填写。
- `ANTHROPIC_MODEL`：当前使用`qwen3.7-flash-2026-07-15`。
- `REDIS_URL`：自己的Redis连接。开发默认回环6379，也可配置私有Unix socket。
- `CHROMA_PERSIST_DIRECTORY`：私有本地数据目录；默认`./data/chroma`。

如果没有Redis实例，可在独立终端启动本机开发实例：

```sh
redis-server --bind 127.0.0.1 --port 6379
```

不要覆盖已有Redis配置或使用它清空其他应用的数据。后端启动：

```sh
.venv/bin/python -B run.py
```

后端仅监听`127.0.0.1:8816`。外部MCP默认关闭，不会启动时自动访问示例服务器。首次使用Chroma默认向量模型可能需要下载模型；后续本地向量化不代表付费Embedding调用。

### 前端

```sh
cd frontend
npm ci
npm run dev
```

打开 **http://127.0.0.1:5173/**。浏览器通过同源`/api/python`访问后端；开发代理默认指向回环8816。如需更改，使用可信服务端变量`SERVEFLOW_BACKEND_ORIGIN`，不要把密钥写入前端变量或运行时配置。

选择快捷问题只填写输入框，主动发送才进入模型链路。分类、Agent、检索改写／重排和后台记忆都可能产生模型费用；没有完整产品级硬额度，部署者须另行配置费用保护。

### 可选MySQL演示

```sh
cd backend
uv pip install --python .venv/bin/python -r requirements-demo.txt
```

按[示例配置](backend/.env.demo.example)创建**独立、空的演示库**及受限账号。初始化CLI仅读取当前进程环境，不会自动加载示例文件；在独立管理终端通过私有配置提供`SERVEFLOW_DEMO_ADMIN_MYSQL_URL`，不要把管理员连接交给Web进程，然后显式执行：

```sh
.venv/bin/python -B -m demo_business init
.venv/bin/python -B -m demo_business verify
```

初始化使用专用管理员连接；Web连接只授予该库SELECT。配置签名密钥文件及精确本机来源后，页面显示可折叠的演示数据入口。没有配置则功能关闭，不会自动建库。已有数据只验证，不覆盖或重置；DDL失败／提交结果不明时停止检查，不盲目重跑。

合成客户A／B及演示Cookie只用于本机样例，不能代替真实客户登录与业务授权。演示记录具有固定快照时间，不能声称查到实时商家物流。

## 检查

普通检查不应调用真实模型或业务接口：

```sh
cd backend
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests

cd ../frontend
npm test
npm run build
```

真实MySQL集成测试需要显式设置独立测试库，否则跳过；不得使用用户正式库。当前前端没有单独的lint／typecheck脚本。

## 当前限制

- 非思考兼容修复已完成离线检查，完整真实问答仍待复验；尚不能保证查单、退款和政策回复在任意表述下稳定正确。
- 示例政策和订单均非真实企业资料；当前知识覆盖很小，不能保证任意客服问题准确回答。
- 当前默认开发模式、演示身份和进程内监控不构成公网多用户隔离、完整成本控制或生产运维方案。请勿直接暴露开发服务或管理接口到公网。
- 当前npm完整审计有7项依赖告警（6 high、1 low），尚未逐项完成专项补修；告警数不等于已证实可利用漏洞数。
- 模型已发送后的取消不能保证上游停止处理或计费；失败不会自动重试。

## 来源与发布状态

本项目使用[MIT许可证](LICENSE)，来源及授权说明见[NOTICE](NOTICE.md)。
