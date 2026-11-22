"""marbert_tokenizer - a self-contained WordPiece tokenizer for the
open-source MARBERT encoder (UBC-NLP/MARBERT, Apache-2.0). No
transformers dependency: vocab.txt + greedy longest-match WordPiece,
matching BertTokenizer's BasicTokenizer behavior for Arabic
(split on non-word chars, keep Arabic ranges, no lowercasing -
MARBERT's vocab is dialect tweet text and is cased for Latin).
Errors: none (OOV -> [UNK])."""
from __future__ import annotations

import re

_AR = re.compile(r"[\u0600-\u06FF]")
_PUNCT = re.compile(
    r"[^\w\u0600-\u06FF\s]|_", re.UNICODE)


def _basic_split(text: str) -> list[str]:
    """BERT BasicTokenizer: whitespace + punctuation split, keep
    digits/Latin/Arabic words whole, lowercase Latin. Errors: none."""
    out = []
    for chunk in text.split():
        for piece in _PUNCT.split(chunk):
            if piece:
                out.append(piece.lower())
    return out


class WordPiece:
    def __init__(self, vocab_path: str, max_chars: int = 100):
        self.vocab: dict[str, int] = {}
        for i, line in enumerate(open(vocab_path, encoding="utf-8")):
            self.vocab[line.rstrip("\n")] = i
        self.unk = self.vocab.get("[UNK]", 100)
        self.max_chars = max_chars

    def piece(self, token: str) -> list[str]:
        """Greedy longest-match-first WordPiece for one basic token.
        Errors: none."""
        if len(token) > self.max_chars:
            return ["[UNK]"]
        if token in self.vocab:
            return [token]
        out, start = [], 0
        while start < len(token):
            end = len(token)
            found = None
            while start < end:
                sub = token[start:end]
                if start > 0:
                    sub = "##" + sub
                if sub in self.vocab:
                    found = sub
                    break
                end -= 1
            if found is None:
                return ["[UNK]"]
            out.append(found)
            start = end
        return out

    def encode(self, text: str, max_len: int = 96) -> list[int]:
        """text -> [CLS] ids [SEP] (truncated). Errors: none."""
        ids = [self.vocab["[CLS]"]]
        for tok in _basic_split(text):
            for p in self.piece(tok):
                ids.append(self.vocab.get(p, self.unk))
        ids.append(self.vocab["[SEP]"])
        if len(ids) > max_len:
            ids = ids[: max_len - 1] + [self.vocab["[SEP]"]]
        return ids

    def decode_len(self, n: int) -> int:
        return n
