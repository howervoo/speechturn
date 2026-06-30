# 训练与恢复

## 模型

声学前端通过 log-mel、步长卷积和 Transformer 编码器构造音频 token。线性或 MLP 投影把它们转为文本模型维度，随后和 UTF-8 字节 token 一起送入因果解码器。

填充音频在卷积前清零；编码器、解码器都显式处理有效长度。文本位置按有效 token 累计，避免批次中长音频引入的填充改变短音频预测。默认模型随机初始化。

## 损失与冻结

提示词和填充标签为 -100，不参与交叉熵。每个梯度累积步骤按照实际监督 token 数加权。`freeze_encoder` / `freeze_decoder` 可冻结两侧骨干；投影层保持可训练。

## 命令

```sh
.venv/bin/speechturn train data/manifest.jsonl output/model.pt --steps 20
.venv/bin/speechturn train data/manifest.jsonl output/resumed.pt --steps 40 --resume output/model.pt
```

`--steps` 是最终总步数。恢复保持数据顺序、配置、batch size、梯度累积设置和软件环境不变；模型、优化器与 Python/NumPy/PyTorch 随机状态一起恢复。测试比较连续训练和中断恢复的逐个权重张量及损失，要求完全一致。

检查点使用张量和基础元数据，加载固定设置 `weights_only=True`，不会反序列化任意模型对象。格式标识是 `speechturn.checkpoint.v1`。

## 小型演示

```sh
.venv/bin/python examples/synthetic-demo.py --output /tmp/speechturn-demo
```

演示从两个合成音频开始，训练微型模型并重新加载检查点执行生成。输出 `result.json` 包含实际损失和生成文本。音频不是人声，结果不代表语音识别能力。
