# 可选预训练适配器

## 安装与调用

安装 CPU PyTorch 后执行 `pip install '.[pretrained]'`。接口固定使用 Transformers 4.57.1 的 Qwen2-Audio-Instruct 对话格式：

```python
from speechturn.backends import QwenAudioBackend

backend = QwenAudioBackend('/path/to/Qwen2-Audio-7B-Instruct')
print(backend.transcribe('speech.wav', 'Transcribe the speech.'))
```

默认 `local_files_only=True`、`trust_remote_code=False`。只有明确传入 `allow_download=True` 才允许下载模型。完整权重需要用户准备足够的内存或显存；此适配器默认在 CPU 加载，没有量化或多卡配置。

测试用替身核验模型/processor 参数、音频采样率、`audios` 输入和生成前缀裁剪。开发环境未下载或运行完整 Qwen2-Audio 权重，因此不报告该模型的速度或准确率。

## 参考

- [Transformers 4.57.1 Qwen2-Audio 官方接口](https://huggingface.co/docs/transformers/v4.57.1/model_doc/qwen2_audio)
- [PyTorch 官方检查点指南](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

本项目自行实现微型训练模型和工具接口，不把外部预训练模型作为本项目原创成果。
