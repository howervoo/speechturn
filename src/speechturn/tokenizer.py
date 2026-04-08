"""A reversible UTF-8 byte tokenizer for experiments without external assets."""

from __future__ import annotations

import operator
from collections.abc import Iterable


class ByteTokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 2
    unk_id = 3
    vocab_size = 260

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        """Encode UTF-8 bytes, optionally adding boundary tokens."""
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        if type(bos) is not bool or type(eos) is not bool:
            raise ValueError("boundary settings must be booleans")
        return (
            ([self.bos_id] if bos else [])
            + [byte + 4 for byte in text.encode("utf-8")]
            + ([self.eos_id] if eos else [])
        )

    def decode(
        self, tokens: Iterable[int], skip_special: bool = True, errors: str = "strict"
    ) -> str:
        """Decode bytes; strict UTF-8 decoding exposes incomplete generated sequences."""
        if type(skip_special) is not bool or errors not in ("strict", "replace", "ignore"):
            raise ValueError("invalid decoding policy")
        specials = {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "<unk>"}
        parts: list[str] = []
        buffer = bytearray()
        for token in tokens:
            try:
                value = operator.index(token)
            except TypeError as error:
                raise ValueError("token identifiers must be integers") from error
            if isinstance(token, bool) or not 0 <= value < self.vocab_size:
                raise ValueError("token identifier is outside the byte vocabulary")
            if value >= 4:
                buffer.append(value - 4)
            elif not skip_special:
                parts.append(buffer.decode("utf-8", errors=errors))
                buffer.clear()
                parts.append(specials[value])
        parts.append(buffer.decode("utf-8", errors=errors))
        return "".join(parts)

    def prompt(self, instruction: str) -> list[int]:
        """Build the same instruction prefix used by training and generation."""
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be non-empty text")
        return self.encode(instruction + "\n", bos=True)

    def training_pair(self, instruction: str, answer: str) -> tuple[list[int], list[int]]:
        """Shift next-token targets and mask every instruction-only prediction."""
        prefix = self.prompt(instruction)
        sequence = prefix + self.encode(answer, eos=True)
        inputs = sequence[:-1]
        labels = sequence[1:]
        labels[: len(prefix) - 1] = [-100] * (len(prefix) - 1)
        return inputs, labels


__all__ = ["ByteTokenizer"]
