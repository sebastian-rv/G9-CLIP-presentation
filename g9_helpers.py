"""G9 helper functions: synthetic data generation and all plotting.
The CLIP logic itself (encoders, similarity, loss, training, evaluation) lives in the notebook."""
import os, hashlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

os.makedirs("figures", exist_ok=True)
BLUE, ORANGE = "#0065BD", "#E37222"

# ---------------------------------------------------------------- data
RGB = {"red": (215, 35, 40), "green": (35, 155, 65), "blue": (35, 75, 215), "yellow": (235, 200, 25)}
SIZE = 112

def render(color, shape, rng):
    """Draw one colored shape at a random size, position and shade on a noisy grey background."""
    bg = (np.full((SIZE, SIZE, 3), 238) + rng.normal(0, 6, (SIZE, SIZE, 3))).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(bg); d = ImageDraw.Draw(im)
    r = rng.uniform(18, 38); cx, cy = rng.uniform(r + 4, SIZE - r - 4, 2)
    col = tuple(int(np.clip(v + rng.uniform(-20, 20), 0, 255)) for v in RGB[color])
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
    elif shape == "square":
        s = 0.85 * r; d.rectangle([cx - s, cy - s, cx + s, cy + s], fill=col)
    else:
        d.polygon([(cx, cy - r), (cx - r, cy + 0.8 * r), (cx + r, cy + 0.8 * r)], fill=col)
    return im

def make_split(captions, seen, held_out, seed=2026):
    """720 train / 135 val / 270 test-seen (80/15/30 per seen caption) + 90 test-held-out (30 per held-out caption).
    Uses its own fixed seed, so the split never changes. Every image is unique (checked by hash)."""
    rng = np.random.default_rng(seed); hashes = set()
    parts = {p: ([], []) for p in ["train", "val", "test_seen", "test_held_out"]}
    plan = [(p, c, n) for p, n in [("train", 80), ("val", 15), ("test_seen", 30)] for c in seen] + \
           [("test_held_out", c, 30) for c in held_out]
    for part, cap, n in plan:
        k = 0
        while k < n:
            im = render(*cap.split(), rng); h = hashlib.sha256(im.tobytes()).hexdigest()
            if h in hashes: continue
            hashes.add(h); parts[part][0].append(im); parts[part][1].append(captions.index(cap)); k += 1
    return {p: (ims, torch.tensor(y)) for p, (ims, y) in parts.items()}

# ---------------------------------------------------------------- Part A
def audience_question(image, labels):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 1.1]})
    ax[0].imshow(image); ax[0].axis("off"); ax[1].axis("off")
    ax[1].text(0, 0.9, "Which caption matches best?", fontsize=20, weight="bold")
    for k, lab in enumerate(labels):
        ax[1].text(0.02, 0.65 - 0.17 * k, f"{'ABC'[k]})  “{lab}”", fontsize=17)
    ax[1].text(0, 0.05, "Guess the probability of each one!", fontsize=13, style="italic", color="gray")
    plt.tight_layout(); plt.savefig("figures/A_audience_question.png", dpi=200); plt.show()

def audience_reveal(image, labels, probs, logits):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 1.1]})
    ax[0].imshow(image); ax[0].axis("off")
    y = np.arange(len(labels))[::-1]
    bars = ax[1].barh(y, probs, color=[BLUE if p == max(probs) else "#98C6EA" for p in probs])
    ax[1].set_yticks(y); ax[1].set_yticklabels([f"“{l}”" for l in labels], fontsize=13)
    for b, p, l in zip(bars, probs, logits):
        ax[1].text(min(p + 0.02, 0.75), b.get_y() + b.get_height() / 2, f"{p:.1%}   (logit {l:.1f})", va="center", fontsize=12)
    ax[1].set_xlim(0, 1.05); ax[1].set_xlabel("softmax over these candidates")
    for s in ["top", "right"]: ax[1].spines[s].set_visible(False)
    ax[1].set_title("CLIP's answer", fontsize=16, weight="bold")
    plt.tight_layout(); plt.savefig("figures/A_audience_reveal.png", dpi=200); plt.show()

# ---------------------------------------------------------------- Part C / D
def show_dataset(data, captions, held_out):
    fig, axes = plt.subplots(3, 4, figsize=(8, 6.4))
    for ax, cap in zip(axes.T.flat, captions):
        ims, y = data["test_held_out" if cap in held_out else "train"]
        ax.imshow(ims[(y == captions.index(cap)).nonzero()[0, 0]]); ax.axis("off")
        ax.set_title(cap + ("\nHELD OUT" if cap in held_out else ""), fontsize=9,
                     color="crimson" if cap in held_out else "black")
    fig.suptitle("Encoder inputs: one example per caption"); plt.tight_layout()
    plt.savefig("figures/C_dataset.png", dpi=200); plt.show()

def similarity_matrix(img_emb, labels, txt_emb, captions, held_out, title, fname, vmin=None, vmax=None):
    """Rows: mean embedding of all test images of one caption. Columns: the 12 caption embeddings."""
    rows = [torch.nn.functional.normalize(img_emb[labels == i].mean(0), dim=-1) for i in range(len(captions))]
    S = (torch.stack(rows) @ txt_emb.T).numpy()
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(S, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(captions))); ax.set_xticklabels(captions, rotation=55, ha="right")
    ax.set_yticks(range(len(captions))); ax.set_yticklabels(captions)
    for t in ax.get_yticklabels():
        if t.get_text() in held_out: t.set_color("crimson"); t.set_fontweight("bold")
    for i in range(len(captions)):
        for j in range(len(captions)):
            ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    fontweight="bold" if S[i].argmax() == j else "normal")
    ax.set(xlabel="Text candidate", ylabel="Mean test-image embedding (red = held-out)", title=title)
    plt.colorbar(im, label="Cosine similarity"); plt.tight_layout(); plt.savefig(f"figures/{fname}", dpi=200); plt.show()

def show_errors(errors, data, fname, n_show=6):
    if not errors:
        print("no errors to show"); return
    pick = errors[:: max(1, len(errors) // n_show)][:n_show]
    fig, axes = plt.subplots(1, len(pick), figsize=(2.1 * len(pick), 2.7))
    for ax, e in zip(np.atleast_1d(axes), pick):
        ax.imshow(data[e["part"]][0][e["k"]]); ax.axis("off")
        ax.set_title(f"true: {e['true']}\npred: {e['pred']}\n({e['kind']} wrong)", fontsize=8)
    plt.tight_layout(); plt.savefig(f"figures/{fname}", dpi=200); plt.show()

def training_curves(hist, seed):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    ax[0].plot([h[0] for h in hist]); ax[0].set(title="Training loss", xlabel="epoch")
    ax[1].plot([h[1] for h in hist], color="tab:green")
    ax[1].set(title="Validation accuracy (9 seen captions)", xlabel="epoch", ylim=(0, 1.05))
    fig.suptitle(f"Seed {seed}"); plt.tight_layout(); plt.savefig(f"figures/D_training_seed{seed}.png", dpi=200); plt.show()

def seed_comparison(results, zero_shot):
    """results = {seed: (seen_acc, held_out_acc)}; zero_shot = (seen_acc, held_out_acc)."""
    names = ["zero-shot\n(no training)"] + [f"trained\nseed {s}" for s in results]
    vals = [zero_shot] + list(results.values())
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(vals))
    for j, (lab, col) in enumerate([("seen (270 images)", BLUE), ("held-out (90 images)", ORANGE)]):
        bars = ax.bar(x + (j - 0.5) * 0.36, [v[j] for v in vals], 0.34, color=col, label=lab)
        for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v[j] + 0.02, f"{v[j]:.0%}", ha="center", fontsize=10)
    ax.axhline(1 / 12, ls="--", color="gray", label="uniform guess (1/12)")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set(ylabel="Top-1 accuracy (12 candidates)", ylim=(0, 1.3), title="Same CLIP, same split, same 12 candidates")
    ax.legend(fontsize=8, loc="upper center", ncol=3); plt.tight_layout(); plt.savefig("figures/D_seed_comparison.png", dpi=200); plt.show()
