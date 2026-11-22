"""marbert_encoder - minimal BERT-family encoder for the extracted
representation layer of an OPEN-SOURCE checkpoint, implemented
directly in torch from the released state dict (no transformers
dependency at runtime). Config-driven: works for MARBERT (768d) and
for the paraphrase-multilingual MiniLM (384d) chosen after the
measured anisotropy analysis. mean-pool of the last layer over the
attention mask -> L2-normalized sentence vector. Weights load from
the raw pytorch_model.bin (build time) or the packed int8 npz
(runtime). Errors: none (missing files raise at load; callers
degrade to the symbolic engine)."""
from __future__ import annotations

import json
import math

import numpy as np
import torch
import torch.nn as nn

from marbert_tokenizer import WordPiece


def _gelu(x: torch.Tensor) -> torch.Tensor:
    """Exact BERT GELU (erf-based). Errors: none."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class _Layer(nn.Module):
    """One BERT transformer layer (GELU, post-LN), size-agnostic."""
    def __init__(self, d: int, heads: int, inter: int,
                 eps: float = 1e-12):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.att_out = nn.Linear(d, d)
        self.ln1 = nn.LayerNorm(d, eps=eps)
        self.ff = nn.Linear(d, inter)
        self.ff_out = nn.Linear(inter, d)
        self.ln2 = nn.LayerNorm(d, eps=eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.h, D // self.h).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, D // self.h).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, D // self.h).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(D // self.h)
        att = att.masked_fill(mask == 0, -1e9)
        att = torch.softmax(att, dim=-1)
        ctx = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = self.ln1(x + self.att_out(ctx))
        x = self.ln2(x + self.ff_out(_gelu(self.ff(x))))
        return x


def _build_layers(cfg: dict) -> nn.ModuleList:
    d = cfg.get("hidden_size", 768)
    heads = cfg.get("num_attention_heads", 12)
    inter = cfg.get("intermediate_size", 4 * d)
    eps = cfg.get("layer_norm_eps", 1e-12)
    nl = cfg.get("num_hidden_layers", 12)
    return nn.ModuleList([_Layer(d, heads, inter, eps)
                          for _ in range(nl)])


class BertExtractor:
    """Feature extractor over an open-source checkpoint: text ->
    L2-normalized mean-pooled vector. Tokenizer: the official
    `tokenizers` library on tokenizer.json (any model type -
    WordPiece or Unigram), falling back to our own WordPiece on
    vocab.txt. Loads weights from raw bin or int8 pack. Errors:
    none (construction may raise; callers degrade)."""

    def __init__(self, model_dir: str):
        import os
        tj = os.path.join(model_dir, "tokenizer.json")
        vp = os.path.join(model_dir, "vocab.txt")
        if os.path.isfile(tj):
            from tokenizers import Tokenizer
            self.tok = Tokenizer.from_file(tj)
            self.hf_tok = True
        else:
            self.wp = WordPiece(vp)
            self.hf_tok = False
        cfg = json.load(open(os.path.join(model_dir,
                                          "config.json"),
                             encoding="utf-8"))
        self.d = cfg.get("hidden_size", 768)
        self.eps = cfg.get("layer_norm_eps", 1e-12)
        self.layers = _build_layers(cfg)

    def _post(self) -> None:
        for m in ([self.emb, self.pos, self.ttype, self.ln0]
                  + list(self.layers)):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

    def _load_bin(self, path: str) -> None:
        sd = torch.load(path, map_location="cpu",
                        weights_only=True)
        # sentence-transformers repos may omit the "bert." prefix
        pre = ("bert." if any(k.startswith("bert.embeddings")
                              for k in sd) else "")
        self.emb = nn.Embedding(
            *sd[pre + "embeddings.word_embeddings.weight"].shape)
        self.emb.weight.data = sd[
            pre + "embeddings.word_embeddings.weight"].float()
        self.pos = nn.Embedding(
            *sd[pre + "embeddings.position_embeddings.weight"]
            .shape)
        self.pos.weight.data = sd[
            pre + "embeddings.position_embeddings.weight"].float()
        self.ttype = nn.Embedding(2, self.d)
        self.ttype.weight.data = sd[
            pre + "embeddings.token_type_embeddings.weight"] \
            .float()
        self.ln0 = nn.LayerNorm(self.d, eps=self.eps)
        self.ln0.weight.data = sd[
            pre + "embeddings.LayerNorm.weight"].float()
        self.ln0.bias.data = sd[
            pre + "embeddings.LayerNorm.bias"].float()
        for i, layer in enumerate(self.layers):
            b = pre + f"encoder.layer.{i}."
            for name, mod in (("q", layer.q), ("k", layer.k),
                              ("v", layer.v),
                              ("att_out", layer.att_out),
                              ("ff", layer.ff),
                              ("ff_out", layer.ff_out)):
                sub = {"q": "attention.self.query",
                       "k": "attention.self.key",
                       "v": "attention.self.value",
                       "att_out": "attention.output.dense",
                       "ff": "intermediate.dense",
                       "ff_out": "output.dense"}[name]
                mod.weight.data = sd[b + sub + ".weight"].float()
                mod.bias.data = sd[b + sub + ".bias"].float()
            layer.ln1.weight.data = sd[
                b + "attention.output.LayerNorm.weight"].float()
            layer.ln1.bias.data = sd[
                b + "attention.output.LayerNorm.bias"].float()
            layer.ln2.weight.data = sd[
                b + "output.LayerNorm.weight"].float()
            layer.ln2.bias.data = sd[
                b + "output.LayerNorm.bias"].float()
        self._post()

    def _load_int8(self, path: str) -> None:
        z = np.load(path)
        W = {k[3:]: torch.from_numpy(z[k]) for k in z.files
             if k.startswith("fp:")}
        Q = {k[2:]: (torch.from_numpy(z[k][:, 1:].astype(np.int8)),
                     torch.from_numpy(z[k][:, 0]).float())
             for k in z.files if k.startswith("q:")}
        # phase-3: the 96M-param word table ships int8
        # (q:emb.word) to cut the pack from 470MB to ~100MB;
        # fp:emb.word stays the fallback for legacy packs.
        if "q4a:emb.word" in z:
            # asymmetric 4-bit: unsigned levels 0..15 over
            # [min, max] per row - better resolution for skewed
            # embedding distributions than symmetric +/-7
            packed = z["q4a:emb.word"]
            step = torch.from_numpy(z["q4s:emb.word"]).float()
            wmin = torch.from_numpy(z["q4m:emb.word"]).float()
            lo = (packed & 0x0F).astype(np.int16)
            hi = ((packed >> 4) & 0x0F).astype(np.int16)
            inter = np.empty((packed.shape[0],
                              packed.shape[1] * 2), np.int16)
            inter[:, 0::2] = lo
            inter[:, 1::2] = hi
            qw = torch.from_numpy(inter)
            self.emb = nn.Embedding(*qw.shape)
            self.emb.weight.data = (qw.float() * step.unsqueeze(1)
                                    + wmin.unsqueeze(1))
        elif "q4:emb.word" in z:
            # 4-bit nibble-packed embeddings: unpack 2 values per
            # byte (low nibble first), sign-extend, scale per row
            packed = z["q4:emb.word"]
            sc = torch.from_numpy(z["q4s:emb.word"]).float()
            lo = (packed & 0x0F).astype(np.int8)
            hi = ((packed >> 4) & 0x0F).astype(np.int8)
            lo = np.where(lo > 7, lo - 16, lo)
            hi = np.where(hi > 7, hi - 16, hi)
            inter = np.empty((packed.shape[0],
                              packed.shape[1] * 2), np.int8)
            inter[:, 0::2] = lo
            inter[:, 1::2] = hi
            qw = torch.from_numpy(inter)
            self.emb = nn.Embedding(*qw.shape)
            self.emb.weight.data = qw.float() * sc.unsqueeze(1)
        elif "q:emb.word" in z:
            qz = z["q:emb.word"]
            sc = torch.from_numpy(qz[:, 0]).float()
            qw = torch.from_numpy(qz[:, 1:].astype(np.int8))
            self.emb = nn.Embedding(*qw.shape)
            self.emb.weight.data = qw.float() * sc.unsqueeze(1)
        else:
            self.emb = nn.Embedding(*W["emb.word"].shape)
            self.emb.weight.data = W["emb.word"].float()
        self.pos = nn.Embedding(*W["emb.pos"].shape)
        self.pos.weight.data = W["emb.pos"].float()
        self.ttype = nn.Embedding(*W["emb.tt"].shape)
        self.ttype.weight.data = W["emb.tt"].float()
        self.ln0 = nn.LayerNorm(self.d, eps=self.eps)
        self.ln0.weight.data = W["ln0.w"].float()
        self.ln0.bias.data = W["ln0.b"].float()
        nl = len(self.layers)
        for i, layer in enumerate(self.layers):
            b = f"L{i}."
            for name, mod in (("q", layer.q), ("k", layer.k),
                              ("v", layer.v),
                              ("att_out", layer.att_out),
                              ("ff", layer.ff),
                              ("ff_out", layer.ff_out)):
                qw, sc = Q[b + name]
                mod.weight.data = qw.float() * sc.unsqueeze(1)
                mod.bias.data = W[b + name + ".b"].float()
            layer.ln1.weight.data = W[b + "ln1.w"].float()
            layer.ln1.bias.data = W[b + "ln1.b"].float()
            layer.ln2.weight.data = W[b + "ln2.w"].float()
            layer.ln2.bias.data = W[b + "ln2.b"].float()
        self._post()

    def ids_batch(self, texts: list[str],
                  max_len: int = 64) -> tuple[torch.Tensor,
                                              torch.Tensor]:
        """Pad a batch to [B, T] ids + mask. Errors: none."""
        seqs = []
        for t in texts:
            if self.hf_tok:
                ids = self.tok.encode(t).ids
                if len(ids) > max_len:
                    ids = ids[: max_len - 1] + [ids[-1]]
                seqs.append(ids)
            else:
                seqs.append(self.wp.encode(t, max_len=max_len))
        T = max(len(s) for s in seqs)
        ids_t = torch.zeros(len(seqs), T, dtype=torch.long)
        mask = torch.zeros(len(seqs), T, dtype=torch.float)
        for i, s in enumerate(seqs):
            ids_t[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1.0
        return ids_t, mask

    @torch.no_grad()
    def embed(self, texts: list[str], max_len: int = 64,
              batch: int = 16) -> np.ndarray:
        """texts -> (N, hidden) L2-normalized mean-pooled vectors.
        Errors: none (empty list -> zero rows)."""
        if not texts:
            return np.zeros((0, self.d), dtype=np.float32)
        out = []
        for i in range(0, len(texts), batch):
            ids, mask = self.ids_batch(texts[i:i + batch],
                                       max_len=max_len)
            B, T = ids.shape
            pos = torch.arange(T).unsqueeze(0)
            x = (self.emb(ids) + self.pos(pos)
                 + self.ttype(torch.zeros_like(ids)))
            x = self.ln0(x)
            for layer in self.layers:
                x = layer(x, mask.unsqueeze(1).unsqueeze(1))
            m = mask.unsqueeze(-1)
            pooled = (x * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out.append(pooled)
        v = torch.cat(out)
        return torch.nn.functional.normalize(v, dim=1).numpy()
