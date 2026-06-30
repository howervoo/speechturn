"""Exercise audio preparation, CPU training, checkpoint loading and generation."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from speechturn.batching import prepare_example
from speechturn.checkpoint import load_checkpoint
from speechturn.config import SpeechConfig, TrainConfig
from speechturn.data import AudioExample, save_manifest
from speechturn.tokenizer import ByteTokenizer
from speechturn.training import train


def run(output: Path, steps: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    settings = SpeechConfig(encoder_dim=16, model_dim=16, num_heads=2)
    time = np.arange(1600, dtype=np.float32) / settings.sample_rate
    records = []
    for index, frequency in enumerate((220, 660)):
        name = f"tone-{index}.wav"
        sf.write(output / name, 0.2 * np.sin(2 * np.pi * frequency * time), settings.sample_rate)
        records.append(
            AudioExample(f"tone-{index}", name, "Q", chr(97 + index), f"synthetic-{index}")
        )
    save_manifest(records, output / "manifest.jsonl")
    examples = [prepare_example(record, output, settings) for record in records]
    checkpoint = output / "tiny-model.pt"
    result = train(examples, checkpoint, settings, TrainConfig(steps=steps, learning_rate=0.01))
    restored, _, restored_step = load_checkpoint(checkpoint)
    tokenizer = ByteTokenizer()
    features = torch.from_numpy(examples[0].features)[None]
    generated = restored.generate(
        features,
        torch.tensor([features.shape[1]]),
        torch.tensor([tokenizer.prompt("Q")]),
        max_new_tokens=4,
    )
    report = {
        "data": "synthetic tones, not human speech",
        "training_steps": restored_step,
        "first_loss": result.losses[0],
        "last_loss": result.losses[-1],
        "generated_text": tokenizer.decode(generated[0].tolist(), errors="replace"),
        "parameters": sum(parameter.numel() for parameter in restored.parameters()),
        "config": settings.to_dict(),
    }
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    run(args.output, args.steps)
