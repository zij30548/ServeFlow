# 来源与许可

ServeFlow基于以下客服项目进行适配，保留其原有意图识别、路由、工具、知识检索、记忆、监控和评测方法：

- 后端：[Biscuit-AI531/EchoMind](https://github.com/Biscuit-AI531/EchoMind)，来源提交`ef847eb5b719ef98be9d5673f3429e62cd88109e`。
- 前端：[Biscuit-AI531/EchoMindFrontend](https://github.com/Biscuit-AI531/EchoMindFrontend)，来源提交`2ec9d14ed094203d455ceae311ebb5fc52911674`。

本项目的ServeFlow品牌、演示政策、合成MySQL业务数据及只读工具、界面调整和兼容修复由`zij30548`在上述基础上维护。原框架不作为全部原创成果声明。

维护者已确认持有本次修改及公开再分发所需的完整授权。本仓库以MIT许可证发布；以上保留源代码的来源元信息，依赖包各自的许可证继续适用。

受保护演示镜像缓存Chroma官方分发的`all-MiniLM-L6-v2` ONNX文件，用于本地向量化；对应[Sentence Transformers模型](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)标注Apache-2.0。镜像附带[Apache-2.0许可文本](deploy/demo/licenses/APACHE-2.0.txt)，模型文件不因项目MIT许可证而改换许可。基础镜像和安装依赖仍保留各自的许可文件。
