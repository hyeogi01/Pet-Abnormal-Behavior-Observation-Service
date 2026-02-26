"""
pet_abnormal_omni_test.py

학습된 pet_abnormal_omni_best.pth 를 로드하여
WORK_DIR/test/ 하위에 고정된 test split에 대한 최종 성능을 평가한다.

[변경] seed 재현 방식 → WORK_DIR/test/ 직접 읽기
  - OS·파일 추가/삭제·seed 변경에 영향받지 않는 안전한 test set 보장
  - work_dir 은 checkpoint 에서 자동으로 읽어옴

출력 항목:
  - Skin / Eyes 전체 Accuracy
  - 클래스별 Precision / Recall / F1 (classification_report)
  - 클래스별 혼동행렬 히트맵 PNG
  - 클래스별 F1 바차트 PNG
  - 요약 결과 JSON (test_results.json)

사용법:
  python pet_abnormal_omni_test.py \
      --ckpt        pet_abnormal_omni_best.pth \
      --output_dir  test_results

  # work_dir 을 직접 지정하고 싶은 경우:
  python pet_abnormal_omni_test.py \
      --ckpt        pet_abnormal_omni_best.pth \
      --work_dir    files/work/abnormal_dataset \
      --output_dir  test_results
"""

import os
import gc
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ===============================
# CONFIG
# ===============================

BATCH_SIZE  = 32
NUM_WORKERS = 8

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── 클래스 정의 (train과 동일, checkpoint에서도 읽어옴) ──────────────────────
SKIN_CLASSES = [
    "cat_normal", "cat_결절,종괴", "cat_농포,여드름",
    "cat_비듬,각질,상피성잔고리", "dog_normal",
    "dog_결절,종괴", "dog_농포,여드름", "dog_미란,궤양",
    "dog_비듬,각질,상피성잔고리",
]

EYES_CLASSES = [
    "cat_normal", "cat_각막궤양", "cat_각막부골편",
    "cat_결막염", "cat_비궤양성각막염", "cat_안검염",
    "dog_normal", "dog_결막염", "dog_궤양성각막질환_상",
    "dog_궤양성각막질환_하", "dog_백내장_비성숙", "dog_백내장_성숙",
    "dog_백내장_초기", "dog_비궤양성각막질환_상", "dog_비궤양성각막질환_하",
    "dog_색소침착성각막염", "dog_안검내반증", "dog_안검염",
    "dog_안검종양", "dog_유루증", "dog_핵경화",
]


# ===============================
# MODEL  (train 파일과 동일 — EfficientNet-B3 + SE Attention)
# ===============================

def _efficientnet_b3_backbone():
    backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    feat_dim = backbone.classifier[1].in_features  # 1536
    backbone.classifier = nn.Identity()
    return backbone, feat_dim


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class AnomalyMultiBackbone(nn.Module):
    def __init__(self, num_skin_classes: int, num_eyes_classes: int):
        super().__init__()

        self.skin_backbone, skin_feat_dim = _efficientnet_b3_backbone()
        self.skin_head = nn.Sequential(
            nn.Linear(skin_feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_skin_classes),
        )

        self.eyes_backbone, eyes_feat_dim = _efficientnet_b3_backbone()
        self.eyes_se   = SqueezeExcitation(eyes_feat_dim, reduction=16)
        self.eyes_head = nn.Sequential(
            nn.Linear(eyes_feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_eyes_classes),
        )

    def forward(self, x: torch.Tensor, task: str = "skin") -> torch.Tensor:
        if task == "skin":
            return self.skin_head(self.skin_backbone(x))
        elif task == "eyes":
            feat = self.eyes_backbone(x)
            feat = self.eyes_se(feat)
            return self.eyes_head(feat)
        else:
            raise ValueError(f"Unknown task: {task!r}")


# ===============================
# DATASET  (WORK_DIR/test/ 폴더를 직접 읽음)
# ===============================

class TestDataset(Dataset):
    """
    WORK_DIR/test/{task}/{class_name}/ 구조에서 이미지를 읽는 Dataset.
    Augmentation 없이 resize만 적용.
    """

    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(self, task_test_dir: str, class_names: list):
        """
        Args:
            task_test_dir : WORK_DIR/test/{task}/ 경로
            class_names   : 클래스 이름 리스트 (순서가 label_idx에 대응)
        """
        name_to_idx   = {n: i for i, n in enumerate(class_names)}
        self.samples  = []   # [(img_path, label_idx), ...]

        for class_name in sorted(os.listdir(task_test_dir)):
            class_dir = os.path.join(task_test_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            if class_name not in name_to_idx:
                print(f"  ⚠️  '{class_name}' 이 class_names 에 없습니다. 건너뜁니다.")
                continue
            label_idx = name_to_idx[class_name]
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_dir, fname), label_idx))

        print(f"  📊 {os.path.basename(task_test_dir)}: {len(self.samples)} test samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.TRANSFORM(img), label


# ===============================
# EVALUATION UTILITIES
# ===============================

def run_inference(model, loader, task: str) -> tuple:
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"  [{task.upper()}] Inference", ncols=110, leave=False):
            images = images.to(DEVICE)
            with autocast():
                logits = model(images, task=task)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())
    return all_preds, all_labels


def plot_confusion_matrix(cm, class_names, title, save_path):
    n        = len(class_names)
    figsize  = (max(10, n * 0.8), max(8, n * 0.7))
    fontsize = max(6, 12 - n // 5)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=fontsize)
    ax.set_yticklabels(class_names, fontsize=fontsize)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=fontsize,
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {save_path}")


def evaluate_task(model, task_test_dir, class_names, task, output_dir) -> dict:
    """단일 태스크 전체 평가 수행."""
    print(f"\n{'─'*55}")
    print(f"[Test] {task.upper()}  —  {task_test_dir}")

    ds = TestDataset(task_test_dir, class_names)
    if len(ds) == 0:
        print("  ⚠️  test samples 가 없습니다. 건너뜁니다.")
        return {}

    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0), prefetch_factor=4,
    )

    all_preds, all_labels = run_inference(model, loader, task)
    del ds, loader
    gc.collect(); torch.cuda.empty_cache()

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n  Overall Accuracy: {acc*100:.2f}%")

    present_ids   = sorted(set(all_labels))
    present_names = [class_names[i] for i in present_ids]

    report_str  = classification_report(
        all_labels, all_preds, labels=present_ids, target_names=present_names,
        digits=4, zero_division=0,
    )
    report_dict = classification_report(
        all_labels, all_preds, labels=present_ids, target_names=present_names,
        digits=4, zero_division=0, output_dict=True,
    )
    print(f"\n{report_str}")

    cm = confusion_matrix(all_labels, all_preds, labels=present_ids)
    plot_confusion_matrix(
        cm, present_names,
        title     = f"{task.upper()} Confusion Matrix  (Acc: {acc*100:.2f}%)",
        save_path = os.path.join(output_dir, f"confusion_matrix_{task}.png"),
    )

    # F1 바차트
    _save_f1_bar(report_dict, present_names, acc, task, output_dir)

    return {
        "task"            : task,
        "num_test_samples": len(all_labels),
        "accuracy"        : round(acc, 6),
        "report"          : report_dict,
        "confusion_matrix": cm.tolist(),
        "class_names"     : present_names,
    }


def _save_f1_bar(report_dict, class_names, acc, task, output_dir):
    f1_scores = [report_dict.get(cn, {}).get("f1-score", 0.0) for cn in class_names]
    colors    = ["steelblue" if s >= 0.7 else "tomato" if s < 0.4 else "orange" for s in f1_scores]

    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.8), 5))
    bars = ax.bar(range(len(class_names)), f1_scores, color=colors, edgecolor="white", alpha=0.88)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1); ax.set_ylabel("F1 Score")
    ax.set_title(f"{task.upper()} — Per-class F1  (Acc: {acc*100:.2f}%)",
                 fontsize=12, fontweight="bold")
    ax.axhline(acc, color="black", linestyle="--", alpha=0.6,
               label=f"Overall Acc {acc*100:.2f}%")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    for bar, score in zip(bars, f1_scores):
        if score > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{score:.2f}", ha="center", va="bottom", fontsize=7)

    save_path = os.path.join(output_dir, f"f1_bar_{task}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {save_path}")


# ===============================
# MAIN TEST FUNCTION
# ===============================

def test(
    ckpt_path : str = "pet_abnormal_omni_best.pth",
    work_dir  : str = None,   # None이면 checkpoint에서 읽어옴
    output_dir: str = "test_results",
):
    """
    Best checkpoint 를 로드하고 WORK_DIR/test/ 하위의 test set을 평가한다.

    train 시 collect_copy_split() 이 파일을 WORK_DIR에 물리적으로 복사해두었으므로
    seed / ratio 재현 없이 안전하게 test set을 사용할 수 있다.

    Args:
        ckpt_path  : 학습 중 저장된 .pth 파일 경로
        work_dir   : WORK_DIR 경로. None 이면 checkpoint 의 'work_dir' 키에서 자동 로드
        output_dir : 결과 파일 저장 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n🎯 Device : {DEVICE}")
    print(f"📂 Checkpoint : {ckpt_path}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    # work_dir 결정: 인자 > checkpoint > 기본값
    if work_dir is None:
        work_dir = ckpt.get("work_dir", "files/work/abnormal_dataset")
    print(f"📂 Work Dir   : {work_dir}")

    skin_classes = ckpt.get("skin_classes", SKIN_CLASSES)
    eyes_classes = ckpt.get("eyes_classes", EYES_CLASSES)
    best_epoch   = ckpt.get("epoch",        "?")
    best_val_acc = ckpt.get("best_avg_acc", None)

    print(f"📌 Checkpoint info  →  epoch: {best_epoch}"
          + (f"  |  val avg acc: {best_val_acc*100:.2f}%" if best_val_acc else ""))

    # ── 모델 복원 ──────────────────────────────────────────────────────────────
    model = AnomalyMultiBackbone(len(skin_classes), len(eyes_classes))
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded successfully.")

    test_dir = os.path.join(work_dir, "test")

    # ── Skin 평가 ─────────────────────────────────────────────────────────────
    skin_results = evaluate_task(
        model,
        task_test_dir = os.path.join(test_dir, "skin"),
        class_names   = skin_classes,
        task          = "skin",
        output_dir    = output_dir,
    )

    # ── Eyes 평가 ─────────────────────────────────────────────────────────────
    eyes_results = evaluate_task(
        model,
        task_test_dir = os.path.join(test_dir, "eyes"),
        class_names   = eyes_classes,
        task          = "eyes",
        output_dir    = output_dir,
    )

    # ── 최종 요약 ──────────────────────────────────────────────────────────────
    skin_acc = skin_results.get("accuracy", 0.0)
    eyes_acc = eyes_results.get("accuracy", 0.0)
    avg_acc  = (skin_acc + eyes_acc) / 2

    print(f"\n{'='*55}")
    print(f"🏆  Final Test Results")
    print(f"{'='*55}")
    print(f"  Skin Acc : {skin_acc*100:.2f}%")
    print(f"  Eyes Acc : {eyes_acc*100:.2f}%")
    print(f"  Avg  Acc : {avg_acc*100:.2f}%")
    print(f"{'='*55}")

    # ── 요약 바차트 ────────────────────────────────────────────────────────────
    _save_summary_bar(skin_results, eyes_results, avg_acc, output_dir)

    # ── JSON 저장 ──────────────────────────────────────────────────────────────
    summary = {
        "checkpoint"      : ckpt_path,
        "work_dir"        : work_dir,
        "best_val_epoch"  : best_epoch,
        "best_val_avg_acc": best_val_acc,
        "test_skin_acc"   : round(skin_acc, 6),
        "test_eyes_acc"   : round(eyes_acc, 6),
        "test_avg_acc"    : round(avg_acc,  6),
        "skin_detail"     : skin_results,
        "eyes_detail"     : eyes_results,
    }

    json_path = os.path.join(output_dir, "test_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 Saved: {json_path}")

    return summary


def _save_summary_bar(skin_results, eyes_results, avg_acc, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(22, 6))

    for ax, results, color in zip(axes, [skin_results, eyes_results], ["steelblue", "tomato"]):
        if not results:
            ax.set_visible(False)
            continue
        task        = results["task"]
        report      = results["report"]
        class_names = results["class_names"]
        f1_scores   = [report.get(cn, {}).get("f1-score", 0.0) for cn in class_names]

        bars = ax.bar(range(len(class_names)), f1_scores, color=color, alpha=0.85, edgecolor="white")
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_ylabel("F1 Score")
        ax.set_title(f"{task.upper()} — Per-class F1  (Acc: {results['accuracy']*100:.2f}%)",
                     fontsize=12, fontweight="bold")
        ax.axhline(results["accuracy"], color="black", linestyle="--", alpha=0.6,
                   label=f"Overall Acc {results['accuracy']*100:.2f}%")
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
        for bar, score in zip(bars, f1_scores):
            if score > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{score:.2f}", ha="center", va="bottom", fontsize=7)

    plt.suptitle(f"Test Summary — Avg Acc: {avg_acc*100:.2f}%", fontsize=14, fontweight="bold")
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "test_f1_summary.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {bar_path}")


# ===============================
# ENTRY POINT
# ===============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly model test evaluation")
    parser.add_argument("--ckpt",       type=str, default="pet_abnormal_omni_best.pth")
    parser.add_argument("--work_dir",   type=str, default=None,
                        help="WORK_DIR used during training. "
                             "If not specified, read from checkpoint's 'work_dir' key.")
    parser.add_argument("--output_dir", type=str, default="test_results")
    args = parser.parse_args()

    test(
        ckpt_path  = args.ckpt,
        work_dir   = args.work_dir,
        output_dir = args.output_dir,
    )
