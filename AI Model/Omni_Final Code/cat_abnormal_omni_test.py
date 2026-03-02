"""
cat_abnormal_omni_test.py
────────────────────────────────────────────────────────────────────
고양이 이상 증상 분류 모델 테스트

[추론 전략]
  feat = zeros (항상) — JSON 없음 상황 재현
  학습 시 JSON Modality Dropout(p=0.5)으로 zeros 상황을 충분히 경험
  → 이미지 특징만으로 robust하게 분류

사용법:
  python cat_abnormal_omni_test.py
  python cat_abnormal_omni_test.py --model_path cat_abnormal_omni_best.pth --device cuda:1
────────────────────────────────────────────────────────────────────
"""

import os, gc, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_v2_s
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ─────────────────────────── CONFIG ───────────────────────────────────────────
SEED = 42
random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED)

WORK_DIR     = "files/work/cat_abnormal_dataset"
DEFAULT_CKPT = "cat_abnormal_omni_best.pth"
IMG_SIZE     = 384
BATCH_SIZE   = 64
NUM_WORKERS  = 24

def parse_args():
    p = argparse.ArgumentParser(description="Cat Abnormal Omni — Test")
    p.add_argument("--model_path", type=str, default=DEFAULT_CKPT)
    p.add_argument("--device",     type=str,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--work_dir",   type=str, default=WORK_DIR)
    return p.parse_args()

TRANSFORM_TEST = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─────────────────────────── DATASET ──────────────────────────────────────────
class InferenceDataset(Dataset):
    """
    추론 전용. feat = zeros (항상).
    학습 시 JSON Modality Dropout으로 zeros 상황을 경험했으므로 robust.
    Skin·Eyes 공용.
    """
    def __init__(self, task_dir, class_list, json_feat_dim=5):
        self.json_feat_dim = json_feat_dim
        self.name_to_idx   = {n: i for i, n in enumerate(class_list)}
        self.samples       = []
        for cls in sorted(os.listdir(task_dir)):
            d = os.path.join(task_dir, cls)
            if not os.path.isdir(d) or cls not in self.name_to_idx: continue
            for f in os.listdir(d):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(d, f), self.name_to_idx[cls]))
        print(f"  📂 {os.path.basename(task_dir)}/test: {len(self.samples)} samples")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        p, l = self.samples[idx]
        img  = TRANSFORM_TEST(Image.open(p).convert("RGB"))
        feat = np.zeros(self.json_feat_dim, dtype=np.float32)   # 항상 zeros
        return img, torch.tensor(feat, dtype=torch.float32), l

# ─────────────────────────── MODELS ───────────────────────────────────────────
def _efficientnet_backbone():
    b = efficientnet_v2_s(weights=None)
    feat = b.classifier[1].in_features
    b.classifier = nn.Identity()
    return b, feat

class _FusionBase(nn.Module):
    def __init__(self, num_classes, json_feat_dim, head_hidden):
        super().__init__()
        self.backbone, img_feat = _efficientnet_backbone()
        self.feat_branch = nn.Sequential(
            nn.Linear(json_feat_dim, 32), nn.LayerNorm(32), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(32, 32), nn.GELU(),
        )
        fused  = img_feat + 32
        layers = [nn.Dropout(0.5), nn.Linear(fused, head_hidden[0]),
                  nn.BatchNorm1d(head_hidden[0]), nn.GELU()]
        for i in range(1, len(head_hidden)):
            layers += [nn.Dropout(0.4), nn.Linear(head_hidden[i-1], head_hidden[i]),
                       nn.BatchNorm1d(head_hidden[i]), nn.GELU()]
        layers += [nn.Dropout(0.4), nn.Linear(head_hidden[-1], num_classes)]
        self.head = nn.Sequential(*layers)

    def forward(self, img, feat):
        # 추론 시 self.training=False → Modality Dropout 미적용
        # feat = zeros이므로 feat_branch는 "no info" 임베딩 출력
        return self.head(torch.cat([self.backbone(img), self.feat_branch(feat)], dim=1))

class SkinModel(_FusionBase):
    def __init__(self, num_classes, json_feat_dim=5):
        super().__init__(num_classes, json_feat_dim, head_hidden=[256])

class EyesModel(_FusionBase):
    def __init__(self, num_classes, json_feat_dim=5):
        super().__init__(num_classes, json_feat_dim, head_hidden=[512])

# ─────────────────────────── HELPERS ──────────────────────────────────────────
def clear(): gc.collect(); torch.cuda.empty_cache()

def make_loader(ds):
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True,
                      persistent_workers=True, prefetch_factor=2,
                      multiprocessing_context="fork")

# ─────────────────────────── EVALUATION ───────────────────────────────────────
def evaluate(model, loader, class_list, device):
    n = len(class_list)
    correct = np.zeros(n, dtype=int); total = np.zeros(n, dtype=int)
    all_preds, all_labels = [], []
    model.to(device).eval()
    with torch.no_grad():
        for imgs, feats, labels in tqdm(loader, desc="  Eval", ncols=80, leave=False):
            imgs, feats, labels = imgs.to(device), feats.to(device), labels.to(device)
            with torch.amp.autocast("cuda"):
                preds = model(imgs, feats).argmax(1)
            for p, l in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                correct[l] += int(p == l); total[l] += 1
                all_preds.append(p); all_labels.append(l)
    model.cpu(); clear()
    per_class = np.where(total > 0, correct / total, 0.0)
    return correct.sum() / total.sum(), per_class, np.array(all_preds), np.array(all_labels)

# ─────────────────────────── PLOTS ────────────────────────────────────────────
def plot_confusion_matrix(preds, labels, class_list, title, save_path):
    n = len(class_list)
    cm = np.zeros((n, n), dtype=int)
    for p, l in zip(preds, labels): cm[l][p] += 1
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sum > 0, cm / row_sum.astype(float), 0.0)
    fig, axes = plt.subplots(1, 2, figsize=(max(10, n + 2), max(7, n + 1)))
    for ax, data, fmt, ttl in zip(axes, [cm, cm_norm], [".0f", ".2f"],
                                   ["Count", "Recall (row-normalized)"]):
        im = ax.imshow(data, interpolation="nearest",
                       cmap="Blues" if "Count" in ttl else "RdYlGn",
                       vmin=0, vmax=(None if "Count" in ttl else 1.0))
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(n)); ax.set_xticklabels(class_list, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(class_list, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"{title} — {ttl}")
        thresh = data.max() / 2.0
        for i in range(n):
            for j in range(n):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=7,
                        color="white" if data[i, j] > thresh else "black")
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  💾 {save_path}")

def save_summary_plot(results, class_map, avg_acc, best_val_acc, best_epoch):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (task, res), color in zip(axes, results.items(), ["steelblue", "tomato"]):
        classes = class_map[task]; per_cls = res["per_class"]; x = np.arange(len(classes))
        bars = ax.bar(x, per_cls * 100, color=color, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.axhline(y=res["acc"] * 100, color="red", linestyle="--", linewidth=1.5,
                   label=f"Overall {res['acc']*100:.1f}%")
        ax.set_xticks(x); ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 115); ax.set_ylabel("Accuracy (%)"); ax.set_title(f"Cat {task.capitalize()} — Test")
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, per_cls):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{val*100:.0f}", ha="center", va="bottom", fontsize=8)
    plt.suptitle(f"Cat Abnormal Omni — Test Summary\n"
                 f"Test Avg: {avg_acc*100:.2f}%  |  Best Val: {best_val_acc*100:.1f}% (Ep {best_epoch})\n"
                 f"[Inference mode: feat=zeros, no JSON input]",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig("cat_abnormal_omni_test_summary.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  💾 cat_abnormal_omni_test_summary.png")

# ─────────────────────────── MAIN ─────────────────────────────────────────────
def test():
    args = parse_args()
    device, ckpt_path, work_dir = args.device, args.model_path, args.work_dir

    print(f"\n🐱 Cat Abnormal Omni — Test")
    print(f"  Device: {device}  |  Checkpoint: {ckpt_path}")
    print(f"  [추론 모드] feat=zeros — JSON 없음 상황 (학습 시 Modality Dropout으로 대비됨)")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"체크포인트 없음: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    CAT_SKIN_CLASSES   = ckpt["cat_skin_classes"]
    CAT_EYES_CLASSES   = ckpt["cat_eyes_classes"]
    json_feat_dim      = ckpt.get("json_feat_dim",      5)
    json_train_dropout = ckpt.get("json_train_dropout", 0.5)
    best_epoch         = ckpt.get("best_epoch", "?")
    best_acc           = ckpt.get("best_acc",   0.0)

    print(f"  Best Epoch: {best_epoch}  |  Best Val Avg: {best_acc*100:.1f}%")
    print(f"  JSON feat dim: {json_feat_dim}  |  Train dropout: {json_train_dropout}")
    print(f"  Skin classes ({len(CAT_SKIN_CLASSES)}): {CAT_SKIN_CLASSES}")
    print(f"  Eyes classes ({len(CAT_EYES_CLASSES)}): {CAT_EYES_CLASSES}")

    skin_model = SkinModel(len(CAT_SKIN_CLASSES), json_feat_dim=json_feat_dim)
    eyes_model = EyesModel(len(CAT_EYES_CLASSES), json_feat_dim=json_feat_dim)
    skin_model.load_state_dict(ckpt["skin_model"])
    eyes_model.load_state_dict(ckpt["eyes_model"])
    print("  ✅ 가중치 로드 완료")

    test_root = os.path.join(work_dir, "test")
    skin_loader = make_loader(
        InferenceDataset(os.path.join(test_root, "skin"), CAT_SKIN_CLASSES, json_feat_dim))
    eyes_loader = make_loader(
        InferenceDataset(os.path.join(test_root, "eyes"), CAT_EYES_CLASSES, json_feat_dim))

    print("\n" + "="*55)
    print("  📊 TEST EVALUATION  (feat=zeros, inference mode)")
    print("="*55)

    print("\n🩺 Skin")
    s_acc, s_per, s_preds, s_labels = evaluate(skin_model, skin_loader, CAT_SKIN_CLASSES, device)
    print(f"  Overall: {s_acc*100:.2f}%")
    for cls, acc in zip(CAT_SKIN_CLASSES, s_per):
        print(f"    {cls:<40s}: {acc*100:.1f}%")

    print("\n👁️  Eyes")
    e_acc, e_per, e_preds, e_labels = evaluate(eyes_model, eyes_loader, CAT_EYES_CLASSES, device)
    print(f"  Overall: {e_acc*100:.2f}%")
    for cls, acc in zip(CAT_EYES_CLASSES, e_per):
        print(f"    {cls:<40s}: {acc*100:.1f}%")

    avg = (s_acc + e_acc) / 2
    print("\n" + "="*55)
    print(f"  ✅ Skin : {s_acc*100:.2f}%  |  ✅ Eyes : {e_acc*100:.2f}%")
    print(f"  📌 Test Avg : {avg*100:.2f}%  |  Best Val: {best_acc*100:.1f}% (Ep {best_epoch})")
    print("="*55)

    print("\n  📈 Confusion Matrix 저장...")
    plot_confusion_matrix(s_preds, s_labels, CAT_SKIN_CLASSES,
                          "Cat Skin — Test (no JSON)", "cat_abnormal_skin_cm.png")
    plot_confusion_matrix(e_preds, e_labels, CAT_EYES_CLASSES,
                          "Cat Eyes — Test (no JSON)", "cat_abnormal_eyes_cm.png")
    save_summary_plot(
        {"skin": {"acc": s_acc, "per_class": s_per},
         "eyes": {"acc": e_acc, "per_class": e_per}},
        {"skin": CAT_SKIN_CLASSES, "eyes": CAT_EYES_CLASSES},
        avg, best_acc, best_epoch,
    )
    print("\n🎉 테스트 완료!")


if __name__ == "__main__":
    test()
