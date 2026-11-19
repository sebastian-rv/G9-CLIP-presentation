# G9 · Image–text representations with CLIP

Group presentation for *Programming in Python for Business and Life Science Analytics* (TUM).

**Group G9:** Thi Thu Huyen Dinh, Daniela Birle Mateos, Sebastian Ruano Vila

**Research question:** Does learning individual attributes (every color, every shape) guarantee that a
model recognises a *new combination* of them?

## What the notebook does

Everything uses one pretrained model, [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32).

| Part | Content |
|---|---|
| **A** | The Hugging Face CLIP example: one photo, three candidate captions, with an audience question before and the answer after |
| **B** | Inside CLIP: image and text encoders, normalised vectors, cosine similarity and temperature, reproduced by hand |
| **C** | The same CLIP, zero-shot, on 12 synthetic colored-shape captions (4 colors × 3 shapes), with a similarity matrix and error analysis |
| **D** | Held-out experiment: encoders frozen, their final layers retrained contrastively on 9 seen combinations, tested on 3 held-out combinations (red circle, green square, blue triangle). One change: training seed 0 vs 1 |

**Data split** (same sizes as the course's Mini-CLIP example): 720 train · 135 validation · 270 test-seen ·
90 test-held-out, generated with a fixed seed (2026) so the split never changes.

Limitations are listed at the end of the notebook.

## Files

```
G9_CLIP_v2.ipynb     main notebook (saved with outputs)
g9_helpers.py        synthetic image generation + all plotting functions
figures/             diagrams used in the notebook + figures the notebook saves
requirements.txt     package versions we used
```

`g9_helpers.py` and `figures/` must stay in the same folder as the notebook.

## How to run

1. Install Python 3.14 (other recent versions should also work).
2. Install the packages:
   ```
   pip install -r requirements.txt
   ```
3. Open `G9_CLIP_v2.ipynb` in VS Code (Python + Jupyter extensions) or Jupyter, select that Python as the kernel, and **Run All**.

The first run downloads CLIP (~600 MB) and needs internet access. On a laptop CPU the full notebook takes about 10 minutes.

## Sources and borrowed code

- **Parts A and B:** example code from the Hugging Face CLIP documentation:
  https://huggingface.co/docs/transformers/model_doc/clip
- **Experiment design** (12 captions, 3 held-out combinations, split sizes, symmetric contrastive loss):
  based on the course's Day 5 slides (Mini-CLIP example).
- **CLIP:** Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision*,
  https://arxiv.org/abs/2103.00020
- The notebook and helper code were developed with the help of an AI assistant (Claude) and reviewed by the group.
