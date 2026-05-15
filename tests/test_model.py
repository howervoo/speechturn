"""Model causality, padding, gradients, generation and loss contracts."""

import numpy as np
import torch

from speechturn.batching import PreparedExample, collate_examples
from speechturn.config import SpeechConfig
from speechturn.model import AcousticEncoder, SpeechLanguageModel


def config(**values):
    return SpeechConfig(
        n_mels=8,
        encoder_dim=8,
        model_dim=8,
        num_heads=2,
        max_audio_tokens=32,
        max_text_tokens=32,
        **values,
    )


def batch():
    return collate_examples(
        [
            PreparedExample("short", np.ones((5, 8), dtype=np.float32), "Q", "a"),
            PreparedExample("long", np.full((9, 8), 0.2, dtype=np.float32), "QQ", "bb"),
        ],
        config(),
    )


def test_batch_masks_and_shifted_labels():
    value = batch()
    assert value.features.shape == (2, 9, 8)
    assert value.feature_lengths.tolist() == [5, 9]
    assert value.input_ids.dtype == value.labels.dtype == torch.long
    assert torch.equal(value.attention_mask, value.input_ids.ne(0))
    assert torch.all(value.labels[~value.attention_mask] == -100)


def test_batch_device_copy_keeps_identifiers():
    value = batch()
    copy = value.to("cpu")
    assert copy.ids == value.ids
    assert copy.ids is not value.ids


def test_encoder_downsample_lengths():
    value = batch()
    encoded, valid = AcousticEncoder(config())(value.features, value.feature_lengths)
    assert valid.sum(dim=1).tolist() == [3, 5]
    assert encoded.shape == (2, 5, 8)
    assert torch.all(encoded[~valid] == 0)


def test_model_output_shape():
    value = batch()
    logits = SpeechLanguageModel(config())(value.features, value.feature_lengths, value.input_ids)
    assert logits.shape == (*value.input_ids.shape, 260)


def test_future_text_cannot_change_past_predictions():
    torch.manual_seed(4)
    model = SpeechLanguageModel(config()).eval()
    value = batch()
    changed = value.input_ids.clone()
    changed[:, -1] = 100
    with torch.no_grad():
        before = model(value.features, value.feature_lengths, value.input_ids)
        after = model(value.features, value.feature_lengths, changed)
    assert torch.allclose(before[:, :-1], after[:, :-1], atol=1e-06)
