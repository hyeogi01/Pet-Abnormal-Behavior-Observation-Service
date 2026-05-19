"""
normal_omni_train_heads.py  (Step 2/2)
────────────────────────────────────────────────────────────────────
Pre-extracted SigLIP/Wav2Vec2 features 위에서 classification heads만 학습.

- backbone forward 없음 → 분 단위로 끝남
- SigLIP features: 1152-dim (behavior, emotion, patella)
- Wav2Vec2 features: 768-dim (sound)
- features/*.pt 파일에서 자동으로 dim 감지

실행 순서:
  1. python normal_omni_extract.py     (feature 추출, 10-30분)
  2. python normal_omni_train_heads.py  (head 학습, 수 분)

Usage:
  python normal_omni_train_heads.py
  python normal_omni_train_heads.py --epochs 200 --lr 1e-4
────────────────────────────────────────────────────────────────────
"""

import os, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"
FEATURE_DIR = "features"
SAVE_DIR    = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

PATELLA_FEAT_DIM = 59

# Class definitions
DOG_BEHAVIOR = ["DOG_BODYLOWER","DOG_BODYSCRATCH","DOG_BODYSHAKE","DOG_FEETUP",
                "DOG_FOOTUP","DOG_HEADING","DOG_LYING","DOG_MOUNTING",
                "DOG_SIT","DOG_TAILING","DOG_TAILLOW","DOG_TURN","DOG_WALKRUN"]
DOG_EMOTION  = ["dog_angry","dog_anxious","dog_confused","dog_happy","dog_relaxed","dog_sad"]
DOG_SOUND    = ["dog_bark","dog_howling","dog_respiratory_event","dog_whining"]
DOG_PATELLA  = ["1","2","3","4","normal"]

CAT_BEHAVIOR = ["CAT_ARCH","CAT_ARMSTRETCH","CAT_FOOTPUSH","CAT_GETDOWN",
                "CAT_GROOMING","CAT_HEADING","CAT_LAYDOWN","CAT_LYING",
                "CAT_ROLL","CAT_SITDOWN","CAT_TAILING","CAT_WALKRUN"]
CAT_EMOTION  = ["cat_relaxed","cat_happy","cat_attentive","cat_sad"]
CAT_SOUND    = ["cat_aggressive","cat_positive"]


# ═══════════════════════════════════════════════════════════════════
#  MODEL: Classification Heads (dual hidden_dim 지원)
# ═══════════════════════════════════════════════════════════════════
class ClassificationHeads(nn.Module):
    """
    Species-separated heads.
    이미지 태스크(1152-dim)와 오디오 태스크(768-dim)의 입력 차원이 다름.
    """
    def __init__(self, img_dim, aud_dim):
        super().__init__()
        self.img_dim = img_dim
        self.aud_dim = aud_dim

        def _head(in_dim, n_cls):
            return nn.Sequential(
                nn.Dropout(0.3), nn.Linear(in_dim, 256),
                nn.BatchNorm1d(256), nn.GELU(),
                nn.Dropout(0.2), nn.Linear(256, n_cls),
            )

        # Image-based heads (SigLIP 1152-dim)
        self.dog_behavior = _head(img_dim, len(DOG_BEHAVIOR))
        self.dog_emotion  = _head(img_dim, len(DOG_EMOTION))
        self.cat_behavior = _head(img_dim, len(CAT_BEHAVIOR))
        self.cat_emotion  = _head(img_dim, len(CAT_EMOTION))

        # Audio-based heads (Wav2Vec2 768-dim)
        self.dog_sound = _head(aud_dim, len(DOG_SOUND))
        self.cat_sound = _head(aud_dim, len(CAT_SOUND))

        # Patella: image + 59-dim MLP branch
        self.patella_branch = nn.Sequential(
            nn.Linear(PATELLA_FEAT_DIM, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(128, 128), nn.GELU(),
        )
        self.dog_patella = _head(img_dim + 128, len(DOG_PATELLA))

    def forward(self, h, species, task, patella_feat=None):
        if task == "patella" and species == "dog" and patella_feat is not None:
            pf = self.patella_branch(patella_feat)
            return self.dog_patella(torch.cat([h, pf], 1))
        heads = {
            ("dog","behavior"): self.dog_behavior, ("dog","emotion"): self.dog_emotion,
            ("dog","sound"): self.dog_sound,
            ("cat","behavior"): self.cat_behavior, ("cat","emotion"): self.cat_emotion,
            ("cat","sound"): self.cat_sound,
        }
        return heads[(species, task)](h)


# ═══════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════
class FeatureDataset(Dataset):
    def __init__(self, features, labels, patella_feats=None):
        self.features = features
        self.labels = labels
        self.patella_feats = patella_feats
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {"features": self.features[idx], "label": self.labels[idx]}
        if self.patella_feats is not None:
            item["patella_feat"] = self.patella_feats[idx]
        return item


def load_feature_file(species, task, split):
    path = os.path.join(FEATURE_DIR, f"{species}_{task}_{split}.pt")
    if not os.path.exists(path): return None
    return torch.load(path, map_location="cpu")


# ═══════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════
def train(args):
    print(f"🚀 Head-only training (SigLIP + Wav2Vec2 features)")
    print(f"   Device: {DEVICE} | Epochs: {args.epochs} | LR: {args.lr} | Batch: {args.batch_size}")

    task_configs = [
        ("dog", "behavior", DOG_BEHAVIOR),
        ("dog", "emotion",  DOG_EMOTION),
        ("dog", "sound",    DOG_SOUND),
        ("dog", "patella",  DOG_PATELLA),
        ("cat", "behavior", CAT_BEHAVIOR),
        ("cat", "emotion",  CAT_EMOTION),
        ("cat", "sound",    CAT_SOUND),
    ]

    tasks = {}
    img_dim, aud_dim = None, None

    print("\n📂 Loading pre-extracted features...")
    for species, task_name, cls_list in task_configs:
        train_data = load_feature_file(species, task_name, "train")
        val_data   = load_feature_file(species, task_name, "val")
        if train_data is None:
            print(f"  ⚠️  {species}/{task_name} not found, skipping")
            continue

        hdim = train_data["hidden_dim"]
        modality = train_data.get("modality", "image")
        if modality == "audio":
            aud_dim = hdim
        else:
            img_dim = hdim

        is_patella = task_name == "patella"
        train_ds = FeatureDataset(
            train_data["features"], train_data["labels"],
            train_data.get("patella_feats") if is_patella else None)
        val_ds = FeatureDataset(
            val_data["features"], val_data["labels"],
            val_data.get("patella_feats") if is_patella else None) if val_data else None

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, drop_last=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=0) if val_ds else None

        # Class weights
        y = train_data["labels"].numpy()
        try:
            w = compute_class_weight('balanced', classes=np.arange(len(cls_list)), y=y)
            cw = torch.tensor(w, dtype=torch.float).to(DEVICE)
        except: cw = None

        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
        n_train = len(train_ds)
        n_val = len(val_ds) if val_ds else 0
        print(f"  ✅ {species}/{task_name}: train={n_train}, val={n_val}, "
              f"classes={len(cls_list)}, dim={hdim} ({modality})")

        tasks[(species, task_name)] = {
            "train_loader": train_loader, "val_loader": val_loader,
            "classes": cls_list, "criterion": criterion,
        }

    if not tasks:
        print("❌ No feature files found. Run normal_omni_extract.py first.")
        return

    # Fallback dims
    if img_dim is None: img_dim = 1152
    if aud_dim is None: aud_dim = 768
    print(f"\n  Image dim: {img_dim}, Audio dim: {aud_dim}")

    # ── Model + Optimizer ────────────────────────────────────────
    heads = ClassificationHeads(img_dim, aud_dim).to(DEVICE)
    total_params = sum(p.numel() for p in heads.parameters())
    print(f"  Head params: {total_params:,}")

    optimizer = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = sum(len(t["train_loader"]) for t in tasks.values()) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6)

    # ── Training loop ────────────────────────────────────────────
    best_avg, history = 0.0, []
    print(f"\n{'='*60}")

    for epoch in range(args.epochs):
        heads.train()

        for (species, task_name), task_info in tasks.items():
            loader = task_info["train_loader"]
            criterion = task_info["criterion"]

            for batch in loader:
                h = batch["features"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                pf = batch.get("patella_feat")
                if pf is not None: pf = pf.to(DEVICE)

                optimizer.zero_grad()
                logits = heads(h, species, task_name, patella_feat=pf)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

        # ── Validation ───────────────────────────────────────────
        heads.eval()
        val_accs = {}
        with torch.no_grad():
            for (species, task_name), task_info in tasks.items():
                if task_info["val_loader"] is None: continue
                tag = f"{species}/{task_name}"
                correct, total = 0, 0
                for batch in task_info["val_loader"]:
                    h = batch["features"].to(DEVICE)
                    labels = batch["label"].to(DEVICE)
                    pf = batch.get("patella_feat")
                    if pf is not None: pf = pf.to(DEVICE)
                    logits = heads(h, species, task_name, patella_feat=pf)
                    correct += (logits.argmax(1) == labels).sum().item()
                    total += labels.size(0)
                val_accs[tag] = correct / max(total, 1)

        avg_acc = np.mean(list(val_accs.values())) if val_accs else 0.0

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
            lr_now = optimizer.param_groups[0]["lr"]
            parts = " | ".join(f"{k}:{v*100:.1f}%" for k, v in val_accs.items())
            print(f"  Ep {epoch+1:3d}/{args.epochs} | {parts} | avg:{avg_acc*100:.1f}% | lr:{lr_now:.1e}")

        history.append({"epoch": epoch+1, **val_accs, "avg": avg_acc})

        if avg_acc > best_avg:
            best_avg = avg_acc
            torch.save({
                "heads_state_dict": heads.state_dict(),
                "img_dim": img_dim,
                "aud_dim": aud_dim,
                "classes": {
                    "dog_behavior": DOG_BEHAVIOR, "dog_emotion": DOG_EMOTION,
                    "dog_sound": DOG_SOUND, "dog_patella": DOG_PATELLA,
                    "cat_behavior": CAT_BEHAVIOR, "cat_emotion": CAT_EMOTION,
                    "cat_sound": CAT_SOUND,
                },
                "best_epoch": epoch + 1, "best_acc": best_avg,
                "history": history,
            }, os.path.join(SAVE_DIR, "normal_omni_best.pth"))

    print(f"\n{'='*60}")
    print(f"🎉 Done! Best Avg: {best_avg*100:.1f}%")
    print(f"   Saved: {SAVE_DIR}/normal_omni_best.pth")

    _save_plot(history, best_avg)


def _save_plot(history, best_acc):
    if not history: return
    keys = [k for k in history[0].keys() if k not in ("epoch", "avg")]
    n = len(keys)
    if n == 0: return
    fig, axes = plt.subplots(1, min(n, 7), figsize=(4 * min(n, 7), 4))
    if n == 1: axes = [axes]
    for ax, key in zip(axes, keys):
        vals = [h.get(key, 0) for h in history]
        color = "steelblue" if "dog" in key else "coral"
        ax.plot(vals, color=color, linewidth=2)
        ax.set_title(key, fontsize=8); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
    plt.suptitle(f"Normal Omni Heads (SigLIP+Wav2Vec2) | Best Avg {best_acc*100:.1f}%", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "normal_omni_heads_history.png"), dpi=150)
    plt.close()
    print(f"  📈 {SAVE_DIR}/normal_omni_heads_history.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    train(args)
