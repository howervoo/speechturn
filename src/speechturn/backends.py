"""Optional local Qwen2-Audio inference without implicit model downloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from speechturn.audio import read_audio
from speechturn.config import positive_integer


class QwenAudioBackend:
    """Adapter for Transformers 4.57.1; install the pretrained extra first."""

    def __init__(self, model_path: str | Path, allow_download: bool = False) -> None:
        if type(allow_download) is not bool:
            raise ValueError("allow_download must be boolean")
        try:
            from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        except ImportError as error:
            raise ImportError("install speechturn[pretrained] to use Qwen2-Audio") from error
        self.processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=not allow_download, trust_remote_code=False
        )
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=not allow_download, trust_remote_code=False
        )
        self.model.eval()

    def transcribe(
        self,
        audio_path: str | Path,
        instruction: str = "Transcribe the speech.",
        max_new_tokens: int = 128,
    ) -> str:
        import torch

        positive_integer(max_new_tokens, "max_new_tokens")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be non-empty text")
        waveform = read_audio(audio_path, int(self.processor.feature_extractor.sampling_rate))
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": str(audio_path)},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(text=text, audios=[waveform], return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        continuation = output[:, inputs["input_ids"].shape[1] :]
        return str(
            self.processor.batch_decode(
                continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        )


__all__ = ["QwenAudioBackend"]
