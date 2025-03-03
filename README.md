# SpeechTurn

面向语音与语言模型实验的 Python 工具箱：音频指令数据校验、可训练的微型声学编码器和因果文本解码器、断点恢复、WER/CER、流式事件协议，以及可选的 Qwen2-Audio 适配器。

作者：Yang Chuanjun · 深圳大学。

## 时间线说明

本项目于 2026 年 9 月创建和验证。Git 中 2025 年至 2026 年中的日期是按要求生成的演示时间线，不代表那些日期已经开展的工作。每次开发提交均经过构建、测试、格式、静态检查；未使用空提交。项目未宣称论文成果或真实语音基准成绩。

## 快速开始

```sh
python3 --version  # Python 3.11 或更新版本
make install      # CPU PyTorch，随后安装项目和开发工具
make check
speechturn --help
speechturn evaluate 'hello speech' 'hello world'
```

核心音频与数据工具也可以通过 `pip install .` 安装；训练需要 `.[train]`。仓库中的 `make install` 会先从官方 CPU 索引安装 PyTorch，以免默认安装 CUDA 依赖。

## 数据格式

每行一个 JSON 对象，音频路径相对于清单目录。`id` 不可重复，按 `speaker` 分割避免说话人泄漏。

```json
{"id":"sample-1","audio":"sample.wav","instruction":"转写语音","text":"你好","speaker":"speaker-1","language":"zh","start":0,"duration":null}
```

```sh
speechturn validate data/manifest.jsonl
speechturn split data/manifest.jsonl data/splits --seed 7
speechturn train data/manifest.jsonl output/model.pt --steps 20
speechturn infer output/model.pt data/sample.wav --instruction '转写语音'
```

## 模型与范围

```text
波形 → 重采样 → log-mel → Conv + Transformer 编码器
                                ↓
                         Linear / MLP 投影
                                ↓
                声学前缀 + UTF-8 字节 token → 因果解码器
```

默认模型随机初始化，用于验证训练流程和研究接口；少量合成样例无法赋予真实语音识别能力。提示词和填充不参与损失；梯度累积按监督 token 数加权。训练在 CPU 上执行，恢复时读取模型、优化器、Python/NumPy/PyTorch 随机状态。

可选预训练适配器使用 Transformers 4.57.1 的 `Qwen2AudioForConditionalGeneration`，默认只读取本地模型文件。安装 `.[pretrained]` 后可使用 `QwenAudioBackend`。适配器接口由替身测试校验；本项目验证环境未下载或运行完整 Qwen2-Audio 权重。

`streaming` 提供无损切块与事件状态机，不是低延迟语音模型或在线服务。端到端模型当前会重新计算声学前缀，没有 KV 缓存。

## 验证与贡献

测试覆盖音频与配置边界、UTF-8、数据分割、因果与填充掩码、梯度、恢复一致性、事件状态和完整公开 API 快照。修改接口时需明确更新快照。运行 `make check` 后提交；不包含用户音频、密钥或预训练权重。

MIT License。
