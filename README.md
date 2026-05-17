# Multimodal Hateful Meme Detection using CLIP and Deep Learning

**Author:** Jagadeesh Venkatakumar  
**Course:** CMPE 257 — Spring 2026

Binary classification of memes as **Hateful** vs **Non-Hateful** using image and text on the [Facebook Hateful Memes Dataset](https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset).

## Models

| Model | CLI `--model` | Description |
|-------|---------------|-------------|
| Zero-shot CLIP | — | `src.zeroshot` — no training |
| Frozen CLIP + MLP | `clip_mlp` | Concat CLIP embeddings → MLP |
| CLIP + BERT fusion | `clip_bert` | CLIP image + BERT text → concat → MLP |
| CLIP + BERT + cross-attention | `clip_bert_cross` | Image attends to BERT tokens → classifier |
| CLIP + BERT + **co-attention** | `clip_bert_coattn` | **Bidirectional** image↔text attention + contrastive loss (best) |

CLIP image encoders are **frozen** in all trained models. Only classification / fusion layers (and BERT for BERT-based models) are trained.

## Project structure

```
├── src/
│   ├── dataset.py
│   ├── model.py                  # CLIP + MLP
│   ├── model_bert.py             # CLIP + BERT fusion
│   ├── model_cross_attention.py  # CLIP + BERT + cross-attention
│   ├── model_coattention.py      # CLIP + BERT + bidirectional co-attention
│   ├── models_registry.py        # Factory & checkpoints
│   ├── train.py / evaluate.py / zeroshot.py / predict.py
│   └── verify_data.py
├── api/main.py
├── notebooks/training.ipynb      # Google Colab workflow
└── checkpoints/                  # .pt files (gitignored)
```

## Quick start (Colab)

1. Open `notebooks/training.ipynb` in Google Colab (GPU enabled).
2. Clone repo, install deps, set Kaggle token, download data.
3. Run each model section; download checkpoints when done.

## Local commands

```bash
pip install -r requirements.txt

# Zero-shot
python -m src.zeroshot --data-dir data --split dev

# Train
python -m src.train --model clip_mlp --data-dir data
python -m src.train --model clip_bert --data-dir data
python -m src.train --model clip_bert_cross --data-dir data
python -m src.train --model clip_bert_coattn --data-dir data

# Evaluate
python -m src.evaluate --model clip_mlp --data-dir data --split dev
```

## Dataset

Place data under `data/` (see `data/README.md`) or use Kaggle in Colab:

```bash
python -m src.kaggle_data
```

Official `test.jsonl` has no public labels — evaluate on **dev**.

## Checkpoints

Saved under `checkpoints/` (gitignored). **Do not push `.pt` files to GitHub.** Download from Colab after training.

## API (CLIP + MLP)

```bash
uvicorn api.main:app --reload --port 8000
```

Requires `checkpoints/clip_mlp.pt` locally.

## License

MIT — Copyright (c) 2026 Jagadeesh Venkatakumar. See [LICENSE](LICENSE).
