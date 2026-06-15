"""Command-line data checks, CPU training, generation and error measurement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from speechturn import __version__
from speechturn.config import SpeechConfig, TrainConfig
from speechturn.data import load_manifest, save_manifest, split_by_speaker
from speechturn.metrics import character_error_rate, word_error_rate


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        prog="speechturn", description="Reproducible speech-language experiments"
    )
    command.add_argument("--version", action="version", version=__version__)
    sub = command.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate a JSONL manifest and audio paths")
    validate.add_argument("manifest", type=Path)
    split = sub.add_parser("split", help="create speaker-disjoint train/dev/test manifests")
    split.add_argument("manifest", type=Path)
    split.add_argument("output", type=Path)
    split.add_argument("--seed", type=int, default=7)
    evaluate = sub.add_parser("evaluate", help="measure WER and CER for a text pair")
    evaluate.add_argument("reference")
    evaluate.add_argument("hypothesis")
    train = sub.add_parser("train", help="train the tiny reference model on CPU")
    train.add_argument("manifest", type=Path)
    train.add_argument("checkpoint", type=Path)
    train.add_argument("--config", type=Path)
    train.add_argument("--steps", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--resume", type=Path)
    infer = sub.add_parser("infer", help="generate text from a reference-model checkpoint")
    infer.add_argument("checkpoint", type=Path)
    infer.add_argument("audio", type=Path)
    infer.add_argument("--instruction", default="Transcribe the speech.")
    infer.add_argument("--max-new-tokens", type=int, default=32)
    return command


def main(argv: list[str] | None = None) -> int:
    command = parser()
    args = command.parse_args(argv)
    result: dict[str, Any]
    try:
        if args.command == "validate":
            records = load_manifest(args.manifest, check_audio=True)
            result = {
                "examples": len(records),
                "speakers": len({record.speaker for record in records}),
            }
        elif args.command == "split":
            records = load_manifest(args.manifest, check_audio=True)
            # Absolute paths remain valid when manifests move to another directory.
            from dataclasses import replace

            records = [
                replace(record, audio=str((args.manifest.parent / record.audio).resolve()))
                for record in records
            ]
            groups = split_by_speaker(records, seed=args.seed)
            args.output.mkdir(parents=True, exist_ok=True)
            for name, items in groups.items():
                save_manifest(items, args.output / f"{name}.jsonl")
            result = {name: len(items) for name, items in groups.items()}
        elif args.command == "evaluate":
            result = {
                "wer": word_error_rate(args.reference, args.hypothesis).to_dict(),
                "cer": character_error_rate(args.reference, args.hypothesis).to_dict(),
            }
        elif args.command == "train":
            import torch

            from speechturn.batching import prepare_example
            from speechturn.training import train

            torch.set_num_threads(1)
            config = (
                SpeechConfig.from_dict(json.loads(args.config.read_text()))
                if args.config
                else SpeechConfig()
            )
            records = load_manifest(args.manifest, check_audio=True)
            examples = [prepare_example(record, args.manifest.parent, config) for record in records]
            trained = train(
                examples,
                args.checkpoint,
                config,
                TrainConfig(steps=args.steps, batch_size=args.batch_size, seed=args.seed),
                args.resume,
            )
            result = {
                "step": trained.step,
                "losses": trained.losses,
                "checkpoint": str(args.checkpoint),
            }
        else:
            import torch

            from speechturn.audio import log_mel, read_audio
            from speechturn.checkpoint import load_checkpoint
            from speechturn.tokenizer import ByteTokenizer

            torch.set_num_threads(1)
            model, _, step = load_checkpoint(args.checkpoint)
            features = torch.from_numpy(
                log_mel(read_audio(args.audio, model.config.sample_rate), model.config)
            )[None]
            tokenizer = ByteTokenizer()
            prompt = torch.tensor([tokenizer.prompt(args.instruction)], dtype=torch.long)
            tokens = model.generate(
                features, torch.tensor([features.shape[1]]), prompt, args.max_new_tokens
            )
            result = {"text": tokenizer.decode(tokens[0].tolist(), errors="replace"), "step": step}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        command.exit(2, f"speechturn: {error}\n")


__all__ = ["main", "parser"]
