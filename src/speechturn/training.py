"""Deterministic CPU training with token-weighted gradient accumulation."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from speechturn.batching import PreparedExample, collate_examples
from speechturn.checkpoint import load_checkpoint, save_checkpoint, seed_everything
from speechturn.config import SpeechConfig, TrainConfig
from speechturn.model import SpeechLanguageModel, causal_loss


@dataclass(frozen=True)
class TrainResult:
    model: SpeechLanguageModel
    losses: list[float]
    step: int


def train(
    examples: Sequence[PreparedExample],
    checkpoint: str | Path,
    model_config: SpeechConfig | None = None,
    train_config: TrainConfig | None = None,
    resume: str | Path | None = None,
) -> TrainResult:
    """Train to the requested total step; resume includes optimizer and RNG state.

    To reproduce an interrupted run, keep the data, batch size, accumulation,
    optimizer settings and software environment unchanged. Execution is on CPU.
    """
    cfg = model_config or SpeechConfig()
    options = train_config or TrainConfig()
    if not examples:
        raise ValueError("training requires examples")
    # Validate every example before writing any output.
    for example in examples:
        collate_examples([example], cfg)
    if resume is None:
        seed_everything(options.seed)
        model = SpeechLanguageModel(cfg)
        state, start = None, 0
    else:
        model, state, start = load_checkpoint(resume, restore_rng=True)
        if model.config != cfg or state is None:
            raise ValueError("resume requires matching configuration and optimizer state")
    if start > options.steps:
        raise ValueError("checkpoint is beyond the requested total steps")
    # Validate checkpoint path is writable before starting training.
    checkpoint_path = Path(checkpoint)
    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ValueError(f"cannot create checkpoint directory: {error}") from error
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    if state is not None:
        optimizer.load_state_dict(state)
        if any(
            group["lr"] != options.learning_rate or group["weight_decay"] != options.weight_decay
            for group in optimizer.param_groups
        ):
            raise ValueError("resume optimizer settings differ from the checkpoint")
    model.train()
    losses = []
    for step in range(start + 1, options.steps + 1):
        batches = [
            collate_examples(
                [examples[random.randrange(len(examples))] for _ in range(options.batch_size)], cfg
            )
            for _ in range(options.gradient_accumulation)
        ]
        supervised = sum(int(batch.labels.ne(-100).sum()) for batch in batches)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for batch in batches:
            logits = model(
                batch.features, batch.feature_lengths, batch.input_ids, batch.attention_mask
            )
            loss = causal_loss(logits, batch.labels)
            weight = int(batch.labels.ne(-100).sum()) / supervised
            (loss * weight).backward()
            total_loss += float(loss.detach()) * weight
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), options.max_grad_norm, error_if_nonfinite=True
        )
        optimizer.step()
        losses.append(total_loss)
        if options.save_every and step % options.save_every == 0:
            save_checkpoint(checkpoint, model, optimizer, step)
    save_checkpoint(checkpoint, model, optimizer, options.steps)
    return TrainResult(model, losses, options.steps)


__all__ = ["TrainResult", "train"]
