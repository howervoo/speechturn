"""An acoustic encoder, trainable projector and causal text decoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from speechturn.config import SpeechConfig, finite_number, positive_integer


class AcousticEncoder(nn.Module):
    """Encode masked log-mel frames while preserving exact valid lengths."""

    def __init__(self, config: SpeechConfig) -> None:
        super().__init__()
        self.config = config
        self.convolution = nn.Conv1d(
            config.n_mels, config.encoder_dim, 3, stride=config.downsample, padding=1
        )
        self.positions = nn.Embedding(config.max_audio_tokens, config.encoder_dim)
        layer = nn.TransformerEncoderLayer(
            config.encoder_dim,
            config.num_heads,
            4 * config.encoder_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, config.encoder_layers, enable_nested_tensor=False
        )
        self.normalization = nn.LayerNorm(config.encoder_dim)

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or features.shape[0] == 0 or features.shape[2] != self.config.n_mels:
            raise ValueError("features must have shape (batch, frames, n_mels)")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("features must be finite floating point values")
        if lengths.shape != (features.shape[0],) or lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("lengths must be one integer per batch item")
        lengths = lengths.to(features.device)
        if torch.any(lengths <= 0) or torch.any(lengths > features.shape[1]):
            raise ValueError("audio lengths are outside the feature tensor")
        frame_mask = (
            torch.arange(features.shape[1], device=features.device)[None, :] < lengths[:, None]
        )
        cleaned = features.to(self.convolution.weight.dtype).masked_fill(~frame_mask[:, :, None], 0)
        encoded = F.gelu(self.convolution(cleaned.transpose(1, 2))).transpose(1, 2)
        if encoded.shape[1] > self.config.max_audio_tokens:
            raise ValueError(
                "audio exceeds max_audio_tokens; crop it or increase the configuration"
            )
        output_lengths = (lengths + self.config.downsample - 1) // self.config.downsample
        positions = torch.arange(encoded.shape[1], device=features.device)
        valid = positions[None, :] < output_lengths[:, None]
        encoded = encoded + self.positions(positions)[None, :, :]
        encoded = self.transformer(encoded, src_key_padding_mask=~valid)
        return self.normalization(encoded).masked_fill(~valid[:, :, None], 0), valid


class SpeechLanguageModel(nn.Module):
    """Condition next-token predictions on a projected acoustic prefix."""

    def __init__(self, config: SpeechConfig | None = None) -> None:
        super().__init__()
        self.config = config or SpeechConfig()
        cfg = self.config
        self.encoder = AcousticEncoder(cfg)
        self.projector: nn.Module
        if cfg.projection == "linear":
            self.projector = nn.Linear(cfg.encoder_dim, cfg.model_dim)
        else:
            self.projector = nn.Sequential(
                nn.Linear(cfg.encoder_dim, cfg.model_dim),
                nn.GELU(),
                nn.Linear(cfg.model_dim, cfg.model_dim),
            )
        self.tokens = nn.Embedding(cfg.vocab_size, cfg.model_dim, padding_idx=0)
        self.positions = nn.Embedding(cfg.max_audio_tokens + cfg.max_text_tokens, cfg.model_dim)
        layer = nn.TransformerEncoderLayer(
            cfg.model_dim,
            cfg.num_heads,
            4 * cfg.model_dim,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, cfg.decoder_layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(cfg.model_dim)
        self.lm_head = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False)
        if cfg.freeze_encoder:
            self.encoder.requires_grad_(False)
        if cfg.freeze_decoder:
            for module in (
                self.tokens,
                self.positions,
                self.decoder,
                self.output_norm,
                self.lm_head,
            ):
                module.requires_grad_(False)

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[0] != features.shape[0]:
            raise ValueError("input_ids must match the audio batch")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be integer tokens")
        if not 1 <= input_ids.shape[1] <= self.config.max_text_tokens:
            raise ValueError("text length is outside the configured positions")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
            raise ValueError("input token is outside the model vocabulary")
        text_valid = input_ids.ne(0) if attention_mask is None else attention_mask
        if text_valid.shape != input_ids.shape or text_valid.dtype != torch.bool:
            raise ValueError("attention_mask must be boolean and match input_ids")
        if not text_valid.any(dim=1).all():
            raise ValueError("each example needs at least one text token")
        encoded, audio_valid = self.encoder(features, lengths)
        prefix = self.projector(encoded)
        text = self.tokens(input_ids)
        combined = torch.cat((prefix, text), dim=1)
        valid = torch.cat((audio_valid, text_valid), dim=1)
        # Cumulative positions make right-padding in the acoustic prefix invisible.
        position_ids = (valid.long().cumsum(dim=1) - 1).clamp_min(0)
        combined = combined + self.positions(position_ids)
        size = combined.shape[1]
        causal = torch.ones((size, size), dtype=torch.bool, device=combined.device).triu(1)
        hidden = self.decoder(combined, mask=causal, src_key_padding_mask=~valid)
        return self.lm_head(self.output_norm(hidden[:, prefix.shape[1] :]))

    @torch.inference_mode()
    def generate(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        prompt_ids: torch.Tensor | None = None,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """Generate new tokens, stopping finished rows at EOS and padding them with zero."""
        positive_integer(max_new_tokens, "max_new_tokens")
        finite_number(temperature, "temperature")
        tokens = (
            prompt_ids
            if prompt_ids is not None
            else torch.ones((features.shape[0], 1), dtype=torch.long, device=features.device)
        )
        if tokens.ndim != 2 or tokens.shape[1] + max_new_tokens > self.config.max_text_tokens:
            raise ValueError("generation would exceed max_text_tokens")
        if tokens.shape[1] < 1:
            raise ValueError("prompt_ids must contain at least one token per row")
        if not tokens.ne(0).any(dim=1).all():
            raise ValueError("each prompt row needs at least one non-pad token")
        was_training = self.training
        self.eval()
        generated = []
        finished = torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
        try:
            for _ in range(max_new_tokens):
                logits = self(features, lengths, tokens)
                offsets = torch.arange(tokens.shape[1], device=tokens.device)[None, :].expand_as(
                    tokens
                )
                last = offsets.masked_fill(tokens.eq(0), -1).max(dim=1).values
                next_logits = logits[torch.arange(tokens.shape[0], device=tokens.device), last]
                next_logits[:, [0, 1, 3]] = -torch.inf
                if next_logits.shape[1] > 260:
                    next_logits[:, 260:] = -torch.inf
                if temperature == 0:
                    next_ids = next_logits.argmax(dim=-1)
                else:
                    next_ids = torch.multinomial(
                        torch.softmax(next_logits / temperature, dim=-1), 1
                    ).squeeze(-1)
                next_ids = next_ids.masked_fill(finished, 0)
                generated.append(next_ids)
                finished |= next_ids.eq(2)
                tokens = torch.cat((tokens, next_ids[:, None]), dim=1)
                if finished.all():
                    break
        finally:
            self.train(was_training)
        return torch.stack(generated, dim=1)


def causal_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute token-weighted loss while excluding prompt and padding labels."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or labels.dtype != torch.long:
        raise ValueError("labels must be int64 with shape (batch, text)")
    valid = labels.ne(-100)
    if (
        not valid.any()
        or torch.any(labels[valid] < 0)
        or torch.any(labels[valid] >= logits.shape[2])
    ):
        raise ValueError("labels need at least one valid supervised token")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
    )


__all__ = ["AcousticEncoder", "SpeechLanguageModel", "causal_loss"]
