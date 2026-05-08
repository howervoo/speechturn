"""Qwen processor integration without downloading pretrained weights."""

import sys
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch

from speechturn.backends import QwenAudioBackend


def test_qwen_adapter_contract(tmp_path, monkeypatch):
    calls = {}

    class Inputs(dict):
        def to(self, device):
            calls["device"] = device
            return self

    class Processor:
        feature_extractor = SimpleNamespace(sampling_rate=16000)

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls["processor_load"] = (name, kwargs)
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            calls["template"] = (messages, kwargs)
            return "audio prompt"

        def __call__(self, **kwargs):
            calls["process"] = kwargs
            return Inputs(input_ids=torch.tensor([[1, 10, 11]]))

        def batch_decode(self, tokens, **kwargs):
            calls["decode"] = (tokens.tolist(), kwargs)
            return ["recognized"]

    class Model:
        device = "cpu"

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls["model_load"] = (name, kwargs)
            return cls()

        def eval(self):
            return self

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            return torch.tensor([[1, 10, 11, 42, 2]])

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoProcessor=Processor, Qwen2AudioForConditionalGeneration=Model),
    )
    audio = tmp_path / "tone.wav"
    sf.write(audio, np.zeros(80), 8000)
    backend = QwenAudioBackend("local-model")
    assert backend.transcribe(audio, "Q", 8) == "recognized"
    assert (
        calls["processor_load"][1]
        == calls["model_load"][1]
        == {"local_files_only": True, "trust_remote_code": False}
    )
    assert set(calls["process"]) == {"text", "audios", "return_tensors", "padding"}
    assert calls["process"]["audios"][0].shape == (160,)
    assert calls["decode"][0] == [[42, 2]]
    assert calls["generate"]["max_new_tokens"] == 8
