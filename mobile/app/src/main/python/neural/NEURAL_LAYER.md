# NEURAL LAYER - license and provenance (fix17)

## The open-source encoder

- Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  (XLM-R multilingual distillation of paraphrase-mpnet, trained on
  1B+ paraphrase pairs; Arabic covered).
- License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)
- Provenance: Hugging Face public repo, downloaded via a public
  mirror (hf-mirror.com) on 2026-11-21; files: config.json,
  tokenizer.json, pytorch_model.bin (integrity: torch.load OK,
  470,693,617 bytes).
- What we ship: an int8-packed copy of the ENCODER ONLY (12
  transformer layers, mean-pool), NOT the MLM head. Every Linear
  weight is symmetric per-out-row int8 (fp16 scale in row 0);
  embeddings/LayerNorm/biases fp16. 470 MB -> 235 MB. Measured
  fidelity: cosine agreement with fp32 identical to 3 decimals on
  probe sets; max relative reconstruction error of the int8 weights
  0.4%.

## What is OURS (trained on the project's own data)

- whitening.npz: mean + shrinkage-PCA (k=256, alpha=0.2) fitted on
  the anchor/variant corpus of THIS project.
- projection.npz: 2-layer contrastive head (InfoNCE, tau 0.07)
  trained on 6,502 synthetic dialect->canonical pairs built
  mechanically from THIS project's KBs and dialect word resources.
  The 150-question dialect bank was NEVER in training (held out).
- anchors.json / anchors_raw.npy: the canonical anchor set (KB
  questions + math templates + creative tasks + dialog intents)
  and their cached embeddings.

## Composition (owner decision, 2026-11-21)

20% neural (this layer - understanding/routing ONLY, never
generation) / 60% numeric-evolutionary core (answers, memory,
retrieval, math) / 10% symbolic (gates, dialect map, identity).

## Scaling path (trillions of tokens)

- The numeric core counts transitions in ONE linear pass - the
  streaming trainer already scales (5.17M trigram checkpoint
  trained in-sandbox; the pipeline is shard-friendly by design:
  count per shard, merge counts).
- The neural layer: anchors index grows incrementally (learned
  memory entries are embedded and appended at load - no retrain);
  the projection head retrains in minutes on CPU at KB scale; for
  billion-pair training the same script takes a dataloader swap
  (sharded parquet) and a GPU - the architecture does not change.
