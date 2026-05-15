# Multimodal Hateful Meme Detection using CLIP and Deep Learning

**Author:** Jagadeesh Venkatakumar  
**Course:** CMPE 257 — Spring 2026

Binary classification of memes as **Hateful** vs **Non-Hateful** using image and text, built on the [Facebook Hateful Memes Dataset](https://github.com/facebookresearch/fairseq/tree/main/examples/MMFT/hateful_memes).

## Approaches

| Method | Description |
|--------|-------------|
| **Zero-shot CLIP** | Frozen ViT-B/32; classify via prompt similarity (`src/zeroshot.py`) |
| **CLIP + MLP** (main) | Frozen CLIP embeddings → concat → MLP → binary logits |

CLIP is **not** fine-tuned; only the MLP head is trained.

## Project structure

```
├── src/
│   ├── dataset.py      # JSONL + CLIP preprocessing
│   ├── model.py        # Frozen CLIP + MLP classifier
│   ├── train.py        # Training loop
│   ├── evaluate.py     # Metrics & confusion matrix
│   ├── predict.py      # Single-meme inference
│   ├── zeroshot.py     # Zero-shot baseline
│   └── utils.py        # Device, paths, checkpoints
├── api/
│   └── main.py         # FastAPI /predict endpoint
├── notebooks/
│   └── training.ipynb  # Google Colab workflow
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset setup

Download the Facebook Hateful Memes dataset and arrange files as:

```
data/
├── img/
├── train.jsonl
├── dev.jsonl
└── test.jsonl
```

Example JSONL line:

```json
{"id": 42953, "img": "img/42953.png", "label": 1, "text": "caption text here"}
```

`data/` is gitignored — do not commit images or labels.

## Installation

```bash
git clone <your-repo-url>
cd CMPE257_PROJECT
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS / Colab
source venv/bin/activate

pip install -r requirements.txt
```

## Training (local or Colab)

From the **project root**:

```bash
python -m src.train --data-dir data --epochs 5 --batch-size 32 --lr 1e-3
```

Checkpoint saved to `checkpoints/clip_mlp.pt` (~2MB, MLP weights only; CLIP is not stored). File is gitignored.

**Note:** Official `test.jsonl` has no public labels — evaluate on `dev` or `train`, not `test`.

### Google Colab

1. Clone repo: `!git clone <repo> && %cd CMPE257_PROJECT`
2. Install: `!pip install -r requirements.txt`
3. Upload or mount `data/` (Drive: `!ln -s /content/drive/MyDrive/hateful_memes/data data`)
4. Open `notebooks/training.ipynb` or run:

```python
!python -m src.train --data-dir data --epochs 5
```

5. Download `checkpoints/clip_mlp.pt` for local API use.

## Zero-shot baseline

```bash
python -m src.zeroshot --data-dir data --split dev
```

## Evaluation

```bash
python -m src.evaluate --data-dir data --split dev --checkpoint checkpoints/clip_mlp.pt
```

Reports accuracy, precision, recall, F1, and confusion matrix.

## Single prediction (CLI)

```bash
python -m src.predict --image data/img/42953.png --text "optional caption" --checkpoint checkpoints/clip_mlp.pt
```

## API

Place trained weights at `checkpoints/clip_mlp.pt`, then:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

- Docs: http://localhost:8000/docs  
- Health: `GET /health`  
- Predict: `POST /predict` (multipart form: `image` file, optional `text`)

Example with curl:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "image=@data/img/42953.png" \
  -F "text=meme caption here"
```

Response:

```json
{"label": "Hateful", "confidence": 0.91}
```

## Literature comparison

CNN + BERT results from prior papers can be cited in your report; this repo implements **CLIP zero-shot** and **CLIP + MLP** only.

## License

MIT License — Copyright (c) 2026 Jagadeesh Venkatakumar. See [LICENSE](LICENSE).
