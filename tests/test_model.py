"""Model causality, padding, gradients, generation and loss contracts."""

import numpy as np
import pytest
import torch

from speechturn.batching import PreparedExample, collate_examples
from speechturn.config import SpeechConfig
from speechturn.model import AcousticEncoder, SpeechLanguageModel, causal_loss


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


def test_audio_padding_cannot_change_predictions():
    model = SpeechLanguageModel(config()).eval()
    value = batch()
    changed = value.features.clone()
    changed[0, 5:] = 1000
    with torch.no_grad():
        before = model(value.features, value.feature_lengths, value.input_ids)
        after = model(changed, value.feature_lengths, value.input_ids)
    assert torch.equal(before, after)


def test_single_example_matches_padded_batch():
    model = SpeechLanguageModel(config()).eval()
    value = batch()
    with torch.no_grad():
        full = model(value.features, value.feature_lengths, value.input_ids)
        single = model(value.features[:1, :5], value.feature_lengths[:1], value.input_ids[:1, :4])
    assert torch.allclose(full[:1, :4], single, atol=2e-06)


def test_projector_receives_gradient_with_frozen_backbones():
    model = SpeechLanguageModel(config(freeze_encoder=True, freeze_decoder=True))
    value = batch()
    causal_loss(
        model(value.features, value.feature_lengths, value.input_ids), value.labels
    ).backward()
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(parameter.grad is None for parameter in model.decoder.parameters())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.projector.parameters()
    )


@pytest.mark.parametrize("projection", ["linear", "mlp"])
def test_both_projection_paths(projection):
    value = batch()
    model = SpeechLanguageModel(config(projection=projection))
    loss = causal_loss(model(value.features, value.feature_lengths, value.input_ids), value.labels)
    assert torch.isfinite(loss)


def test_loss_excludes_masked_labels():
    logits = torch.zeros(1, 2, 3, requires_grad=True)
    loss = causal_loss(logits, torch.tensor([[1, -100]]))
    loss.backward()
    assert loss.item() == pytest.approx(np.log(3))
    assert torch.all(logits.grad[0, 1] == 0)


def test_all_masked_loss_is_rejected():
    with pytest.raises(ValueError):
        causal_loss(torch.zeros(1, 2, 3), torch.full((1, 2), -100))


def test_greedy_generation_reproducible_and_restores_mode():
    model = SpeechLanguageModel(config())
    value = batch()
    first = model.generate(value.features, value.feature_lengths, max_new_tokens=3)
    assert model.training
    second = model.generate(value.features, value.feature_lengths, max_new_tokens=3)
    assert torch.equal(first, second)
    assert not torch.any((first == 1) | (first == 3))
