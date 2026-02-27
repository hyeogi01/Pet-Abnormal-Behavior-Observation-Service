import os
import gc
import random
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights  # [v2] ResNet50 → EfficientNet-V2-S
from collections import defaultdict
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ===============================
# CONFIG
# ===============================

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCHS        = 50
BATCH_SIZE    = 32          # [v2] 384×384 이미지 → VRAM 증가 → 64→32
NUM_WORKERS   = 24
LR_BACKBONE   = 2e-5       # [v2] EfficientNet backbone (ResNet 5e-6 → 2e-5)
LR_HEAD       = 2e-4       # [v2] Head (5e-4 → 2e-4, EfficientNet 안정성 고려)
FREEZE_EPOCHS = 5
LABEL_SMOOTHING = 0.1

# [v2] EfficientNet-V2-S 권장 해상도 384×384
IMG_SIZE   = 384
IMG_RESIZE = 416

VAL_RATIO  = 0.1
TEST_RATIO = 0.1

WORK_DIR = "files/work/abnormal_dataset"

# ─────────────────────────────────────────────────────────────────────────────
# CLASS DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

# ── 4_Animal_Skin ──
# [v2] cat/dog 분리: 공유 backbone + species-specific head 방식.
# - 2단계 cascade (종 분류 → 피부 분류) 대신 species-conditioned head 채택.
#   이유: (1) 실서비스에서 사용자는 종을 이미 앎
#         (2) cascade는 종 오류 → 질환 오류 전파 위험
#         (3) head 분리만으로 cat/dog cross-species 간섭 제거 효과 동일
CAT_SKIN_CLASSES = [
    "cat_normal", "cat_결절,종괴", "cat_농포,여드름",
    "cat_비듬,각질,상피성잔고리",
]
DOG_SKIN_CLASSES = [
    "dog_normal", "dog_결절,종괴", "dog_농포,여드름",
    "dog_미란,궤양", "dog_비듬,각질,상피성잔고리",
]
# 데이터 수집 및 split 시 전체 목록 사용 (순서 고정: cat → dog)
SKIN_CLASSES = CAT_SKIN_CLASSES + DOG_SKIN_CLASSES

# ── 5_Animal_Eyes ──
# [v2] 통합 후 17클래스 (Eyes_class.txt 기준)
# ⚠️  기존 21클래스(상/하, 백내장 3단계)로 준비된 WORK_DIR/eyes 폴더가 있다면
#     해당 폴더 삭제 후 재실행 필요 (새 클래스명 폴더로 재복사)
EYES_CLASSES = [
    "cat_normal", "cat_각막궤양", "cat_각막부골편",
    "cat_결막염", "cat_비궤양성각막염", "cat_안검염",
    "dog_normal", "dog_결막염", "dog_궤양성각막질환",
    "dog_비궤양성각막질환", "dog_백내장", "dog_색소침착성각막염",
    "dog_안검내반증", "dog_안검염", "dog_안검종양",
    "dog_유루증", "dog_핵경화",
]

EYES_SIMILAR_GROUPS = [
    ["dog_궤양성각막질환", "dog_비궤양성각막질환"],   # 각막 유형 혼동 방지
    ["dog_안검염", "dog_안검내반증"],                  # 안검 관련 혼동 방지
]


# ===============================
# LOSS: Hierarchical-Aware CE
# ===============================

class HierarchicalWeightedLoss(nn.Module):
    """
    CrossEntropyLoss + Label Smoothing + 유사 클래스 혼동 페널티

    Args:
        class_names    : 학습 task에 해당하는 클래스 이름 리스트
        similar_groups : 유사 클래스 묶음 [[cls_a, cls_b], ...]
        class_weights  : 클래스 불균형 보정 weight 텐서
        smoothing      : label smoothing ε
        extra_penalty  : 같은 그룹 내 오분류 시 loss 배율
    """

    def __init__(
        self,
        class_names,
        similar_groups=None,
        class_weights=None,
        smoothing=LABEL_SMOOTHING,
        extra_penalty=1.5,
    ):
        super().__init__()
        self.smoothing     = smoothing
        self.extra_penalty = extra_penalty
        self.num_classes   = len(class_names)
        self.class_names   = class_names
        self.name_to_idx   = {n: i for i, n in enumerate(class_names)}

        self.penalty_pairs = set()
        if similar_groups:
            for group in similar_groups:
                idxs = [self.name_to_idx[n] for n in group if n in self.name_to_idx]
                for i in range(len(idxs)):
                    for j in range(i + 1, len(idxs)):
                        self.penalty_pairs.add((idxs[i], idxs[j]))
                        self.penalty_pairs.add((idxs[j], idxs[i]))

        self.register_buffer("weight", class_weights)

    def forward(self, logits, targets):
        B, C   = logits.shape
        device = logits.device

        log_prob    = F.log_softmax(logits, dim=-1)
        smooth_loss = -log_prob.mean(dim=-1)
        nll_loss    = F.nll_loss(log_prob, targets, weight=self.weight, reduction="none")
        base_loss   = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss

        if self.penalty_pairs:
            pred_classes = logits.argmax(dim=-1)
            penalty_mask = torch.ones(B, device=device)
            for b in range(B):
                t = targets[b].item()
                p = pred_classes[b].item()
                if (t, p) in self.penalty_pairs:
                    penalty_mask[b] = self.extra_penalty
            base_loss = base_loss * penalty_mask

        return base_loss.mean()


# ===============================
# CLASS WEIGHT COMPUTATION
# ===============================

def compute_class_weights(sample_counts: dict, class_names: list) -> torch.Tensor:
    """Inverse-frequency 방식으로 클래스 가중치를 계산한다."""
    counts  = torch.tensor([sample_counts.get(n, 1) for n in class_names], dtype=torch.float)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(class_names)
    return weights


# ===============================
# BACKBONE & MODEL
# [v2] ResNet50 → EfficientNet-V2-S
# ===============================

def _efficientnet_backbone():
    """
    [v2] EfficientNet-V2-S backbone.
    classifier를 Identity로 교체하고 feat_dim(1280) 반환.

    ResNet50 대비 장점:
    - 피부/안구 미세 질감 패턴에서 우수한 특징 추출
    - 더 적은 파라미터로 더 높은 성능 (28M vs 25M)
    - MBConv + Fused-MBConv의 복합 구조 → 세밀한 패턴 학습
    - 권장 입력 해상도 384×384 (IMG_SIZE와 일치)
    """
    backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    feat_dim = backbone.classifier[1].in_features  # 1280
    backbone.classifier = nn.Identity()
    return backbone, feat_dim


class SqueezeExcitation(nn.Module):
    """1-D Squeeze-Excitation for feature vectors (after global avg pool)."""

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


def _classification_head(feat_dim: int, hidden: int, num_classes: int,
                          dropout1: float = 0.4, dropout2: float = 0.3) -> nn.Module:
    """공통 분류 헤드 팩토리 (BN 포함으로 학습 안정성 강화)."""
    return nn.Sequential(
        nn.Dropout(dropout1),
        nn.Linear(feat_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.GELU(),
        nn.Dropout(dropout2),
        nn.Linear(hidden, num_classes),
    )


class AnomalyMultiBackbone(nn.Module):
    """
    [v2] EfficientNet-V2-S 기반 이상 증상 Omni 모델

    ├── skin_backbone (공유, EfficientNet-V2-S, feat=1280)
    │   ├── skin_cat_head → cat 피부질환 4클래스
    │   └── skin_dog_head → dog 피부질환 5클래스
    │
    └── eyes_backbone (EfficientNet-V2-S + SE Attention, feat=1280)
        └── eyes_head → 안구질환 17클래스

    [v2 설계 변경]
    - Skin: 단일 9-class → cat(4) / dog(5) species-conditioned head 분리
      → cat/dog cross-species 간섭 제거, 각 종에 특화된 특징 학습
    - 공유 backbone이 cat/dog 피부 공통 특징 학습 후 head에서 분기
    - Inference 시 pet_type으로 적절한 head 선택 (cascade 불필요)
    """

    def __init__(
        self,
        num_cat_skin: int,
        num_dog_skin: int,
        num_eyes: int,
    ):
        super().__init__()

        # ── Skin: 공유 backbone + species-specific heads ─────────────────────
        self.skin_backbone, skin_feat = _efficientnet_backbone()
        self.skin_cat_head = _classification_head(skin_feat, 256, num_cat_skin)
        self.skin_dog_head = _classification_head(skin_feat, 256, num_dog_skin)

        # ── Eyes: backbone + SE + head ───────────────────────────────────────
        self.eyes_backbone, eyes_feat = _efficientnet_backbone()
        self.eyes_se   = SqueezeExcitation(eyes_feat, reduction=16)
        self.eyes_head = _classification_head(eyes_feat, 512, num_eyes,
                                              dropout1=0.4, dropout2=0.3)

    def freeze_backbones(self):
        for p in self.skin_backbone.parameters():
            p.requires_grad = False
        for p in self.eyes_backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbones(self):
        for p in self.skin_backbone.parameters():
            p.requires_grad = True
        for p in self.eyes_backbone.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor, task: str) -> torch.Tensor:
        if task == "skin_cat":
            return self.skin_cat_head(self.skin_backbone(x))
        elif task == "skin_dog":
            return self.skin_dog_head(self.skin_backbone(x))
        elif task == "eyes":
            feat = self.eyes_se(self.eyes_backbone(x))
            return self.eyes_head(feat)
        raise ValueError(f"Unknown task: {task!r}. Use 'skin_cat', 'skin_dog', or 'eyes'.")


# ===============================
# INFERENCE: Multi-Image Ensemble
# ===============================

def predict_anomaly(
    model: AnomalyMultiBackbone,
    images: list,
    task: str,
    pet_type: str,
    device=DEVICE,
) -> dict:
    """
    [v2] species-conditioned 구조에 맞게 수정.
    - task="skin": pet_type → skin_cat / skin_dog head 직접 선택
      (기존 21→17 마스킹 방식 불필요, 단순해짐)
    - task="eyes": 17클래스 중 pet_type에 해당하는 클래스만 필터링

    Args:
        images   : PIL.Image 리스트 (최대 5장 권장, 앙상블)
        task     : "skin" | "eyes"
        pet_type : "cat" | "dog"
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model.eval()
    model.to(device)

    if task == "skin":
        actual_task = f"skin_{pet_type}"                                        # "skin_cat" or "skin_dog"
        class_names = CAT_SKIN_CLASSES if pet_type == "cat" else DOG_SKIN_CLASSES
    elif task == "eyes":
        actual_task = "eyes"
        valid_idxs  = [i for i, n in enumerate(EYES_CLASSES) if n.startswith(pet_type + "_")]
        class_names = [EYES_CLASSES[i] for i in valid_idxs]
    else:
        raise ValueError(f"Unknown task: {task!r}")

    with torch.no_grad():
        probs_accum = torch.zeros(len(class_names), device=device)
        for img in images:
            tensor = transform(img).unsqueeze(0).to(device)
            logits = model(tensor, task=actual_task)

            if task == "eyes":
                # 17클래스 중 해당 종 클래스만 softmax
                masked = torch.full((len(EYES_CLASSES),), float("-inf"), device=device)
                masked[valid_idxs] = logits[0][valid_idxs]
                probs_accum += F.softmax(masked, dim=-1)[valid_idxs]
            else:
                # skin: head 출력이 이미 해당 종만 포함
                probs_accum += F.softmax(logits[0], dim=-1)

        probs_accum /= len(images)

    result = sorted(
        [(class_names[i], probs_accum[i].item()) for i in range(len(class_names))],
        key=lambda x: x[1], reverse=True,
    )
    return {
        "predicted_class": result[0][0],
        "confidence":      result[0][1],
        "top3":            result[:3],
    }


# ===============================
# DATA SPLIT & COPY UTILITY
# ===============================

def _task_ready(task_name: str) -> bool:
    """해당 task의 train 폴더가 존재하고 비어있지 않으면 True."""
    task_train = os.path.join(WORK_DIR, "train", task_name)
    return os.path.isdir(task_train) and len(os.listdir(task_train)) > 0


def collect_copy_split(
    root_dir: str,
    task_name: str,
    class_names: list,
    val_ratio: float  = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int         = SEED,
) -> tuple:
    """
    root_dir 하위 class 디렉토리에서 이미지를 수집하고
    클래스별 stratified split 후 WORK_DIR에 파일을 물리적으로 복사한다.

    Returns:
        train_samples, val_samples
        각 원소: (img_path: str, label_idx: int)  ← WORK_DIR 내 복사된 경로
    """
    rng         = random.Random(seed)
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    class_files = defaultdict(list)
    seen_paths  = set()

    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        label_idx = name_to_idx[class_name]
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(class_dir, fname)
            if fpath in seen_paths:
                continue
            seen_paths.add(fpath)
            class_files[label_idx].append(fpath)

    for split in ["train", "val", "test"]:
        for class_name in class_names:
            os.makedirs(os.path.join(WORK_DIR, split, task_name, class_name), exist_ok=True)

    train_samples, val_samples = [], []

    for label_idx, paths in class_files.items():
        class_name = class_names[label_idx]
        rng.shuffle(paths)
        n       = len(paths)
        n_val   = max(1, int(n * val_ratio))
        n_test  = max(1, int(n * test_ratio))
        n_train = n - n_val - n_test

        if n_train <= 0:
            print(f"  ⚠️  '{class_name}': 샘플 수({n})가 너무 적어 train이 0개입니다.")
            n_train, n_val, n_test = n, 0, 0

        split_map = {
            "train": paths[:n_train],
            "val"  : paths[n_train:n_train + n_val],
            "test" : paths[n_train + n_val:],
        }

        for split_name, split_paths in split_map.items():
            dst_dir = os.path.join(WORK_DIR, split_name, task_name, class_name)
            for src in tqdm(split_paths, desc=f"  copy {task_name}/{split_name}/{class_name}", leave=False):
                dst = os.path.join(dst_dir, os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                if split_name == "train":
                    train_samples.append((dst, label_idx))
                elif split_name == "val":
                    val_samples.append((dst, label_idx))

    print(f"  → {task_name}: train {len(train_samples)} | val {len(val_samples)}"
          f" | test → {os.path.join(WORK_DIR, 'test', task_name)}/")
    return train_samples, val_samples


def count_samples_from_split(samples: list, class_names: list) -> dict:
    """split된 samples에서 class_name별 개수를 반환 (class_weight 계산용)."""
    idx_to_name = {i: n for i, n in enumerate(class_names)}
    counts      = defaultdict(int)
    for _, label_idx in samples:
        counts[idx_to_name[label_idx]] += 1
    return dict(counts)


# ===============================
# DATASETS
# ===============================

class AnomalyDataset(Dataset):
    """
    [v2] EfficientNet-V2-S 권장 해상도 384×384 적용.
    Skin augmentation 완화: 피부색·질감이 진단 핵심 단서이므로 ColorJitter 억제.
    Eyes augmentation 유지: 안구 색 변화(충혈, 혼탁)가 진단 단서이므로 유지.
    """

    # [v2] Skin: ColorJitter 완화, RandomAdjustSharpness 추가 (질감 강조)
    TRANSFORM_SKIN_TRAIN = transforms.Compose([
        transforms.Resize((IMG_RESIZE, IMG_RESIZE)),            # 416
        transforms.RandomCrop(IMG_SIZE),                        # 384
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05),  # 완화
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),              # 질감 강조
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Eyes: 색 변화가 진단 단서 → 적당한 augmentation 유지 + GaussianBlur 추가
    TRANSFORM_EYES_TRAIN = transforms.Compose([
        transforms.Resize((IMG_RESIZE, IMG_RESIZE)),            # 416
        transforms.RandomCrop(IMG_SIZE),                        # 384
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5), p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    TRANSFORM_VAL = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),                # 384
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(self, samples: list, is_train: bool = True, task: str = "skin"):
        self.samples = samples
        if is_train and task == "eyes":
            self.transform = self.TRANSFORM_EYES_TRAIN
        elif is_train:
            self.transform = self.TRANSFORM_SKIN_TRAIN
        else:
            self.transform = self.TRANSFORM_VAL

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), label


# ===============================
# TRAIN FUNCTION
# ===============================

def train(
    skin_root: str = "files/4_Animal_Skin",
    eyes_root: str = "files/5_Animal_Eyes",
):
    print(f"🎯 Device: {DEVICE}")
    print(f"   Backbone : EfficientNet-V2-S (feat=1280)")
    print(f"   IMG_SIZE : {IMG_SIZE}×{IMG_SIZE}")
    print(f"   BATCH    : {BATCH_SIZE}  LR_BB={LR_BACKBONE}  LR_HEAD={LR_HEAD}")

    # ── Dataset Preparation ────────────────────────────────────────────────────
    if _task_ready("skin"):
        print("\n✅ skin already prepared, loading from WORK_DIR...")
        skin_train_samples = _load_samples_from_dir(
            os.path.join(WORK_DIR, "train", "skin"), SKIN_CLASSES
        )
        skin_val_samples = _load_samples_from_dir(
            os.path.join(WORK_DIR, "val", "skin"), SKIN_CLASSES
        )
    else:
        print("\n📦 Splitting & Copying Skin dataset...")
        skin_train_samples, skin_val_samples = collect_copy_split(
            skin_root, "skin", SKIN_CLASSES
        )

    if _task_ready("eyes"):
        print("✅ eyes already prepared, loading from WORK_DIR...")
        eyes_train_samples = _load_samples_from_dir(
            os.path.join(WORK_DIR, "train", "eyes"), EYES_CLASSES
        )
        eyes_val_samples = _load_samples_from_dir(
            os.path.join(WORK_DIR, "val", "eyes"), EYES_CLASSES
        )
    else:
        print("\n📦 Splitting & Copying Eyes dataset...")
        eyes_train_samples, eyes_val_samples = collect_copy_split(
            eyes_root, "eyes", EYES_CLASSES
        )

    # ── [v2] species별 skin 샘플 분리 ─────────────────────────────────────────
    # SKIN_CLASSES = CAT(0~3) + DOG(4~8) 순서 고정
    # cat: global label_idx 0~3 → local 0~3 (그대로)
    # dog: global label_idx 4~8 → local 0~4 (N_CAT 빼줌)
    N_CAT = len(CAT_SKIN_CLASSES)

    skin_cat_train = [(p, l)         for p, l in skin_train_samples if l < N_CAT]
    skin_dog_train = [(p, l - N_CAT) for p, l in skin_train_samples if l >= N_CAT]
    skin_cat_val   = [(p, l)         for p, l in skin_val_samples   if l < N_CAT]
    skin_dog_val   = [(p, l - N_CAT) for p, l in skin_val_samples   if l >= N_CAT]

    print(f"\n  📊 Skin split by species:")
    print(f"     cat → train: {len(skin_cat_train)}  val: {len(skin_cat_val)}")
    print(f"     dog → train: {len(skin_dog_train)}  val: {len(skin_dog_val)}")
    print(f"  📊 Eyes → train: {len(eyes_train_samples)}  val: {len(eyes_val_samples)}")

    # ── 클래스 가중치 ────────────────────────────────────────────────────────────
    cat_skin_counts   = count_samples_from_split(skin_cat_train,   CAT_SKIN_CLASSES)
    dog_skin_counts   = count_samples_from_split(skin_dog_train,   DOG_SKIN_CLASSES)
    eyes_train_counts = count_samples_from_split(eyes_train_samples, EYES_CLASSES)

    cat_skin_weights = compute_class_weights(cat_skin_counts,   CAT_SKIN_CLASSES).to(DEVICE)
    dog_skin_weights = compute_class_weights(dog_skin_counts,   DOG_SKIN_CLASSES).to(DEVICE)
    eyes_weights     = compute_class_weights(eyes_train_counts, EYES_CLASSES).to(DEVICE)

    # ── Loss ─────────────────────────────────────────────────────────────────────
    skin_cat_criterion = HierarchicalWeightedLoss(
        class_names=CAT_SKIN_CLASSES,
        class_weights=cat_skin_weights,
        smoothing=LABEL_SMOOTHING,
    )
    skin_dog_criterion = HierarchicalWeightedLoss(
        class_names=DOG_SKIN_CLASSES,
        class_weights=dog_skin_weights,
        smoothing=LABEL_SMOOTHING,
    )
    eyes_criterion = HierarchicalWeightedLoss(
        class_names=EYES_CLASSES,
        similar_groups=EYES_SIMILAR_GROUPS,
        class_weights=eyes_weights,
        smoothing=LABEL_SMOOTHING,
        extra_penalty=1.5,
    )

    # ── 모델 / Optimizer / Scheduler ─────────────────────────────────────────────
    model = AnomalyMultiBackbone(
        num_cat_skin=len(CAT_SKIN_CLASSES),
        num_dog_skin=len(DOG_SKIN_CLASSES),
        num_eyes=len(EYES_CLASSES),
    ).to(DEVICE)

    model.freeze_backbones()
    print(f"\n  🔒 Backbone frozen for first {FREEZE_EPOCHS} epochs (head-only training)")

    # [v2] skin_cat_head + skin_dog_head → 단일 optimizer로 관리
    #      backbone은 cat/dog 순서로 각각 업데이트 → 양쪽 특징 모두 학습
    skin_optimizer = torch.optim.AdamW([
        {"params": model.skin_backbone.parameters(),  "lr": LR_BACKBONE, "weight_decay": 1e-4},
        {"params": model.skin_cat_head.parameters(),  "lr": LR_HEAD,     "weight_decay": 1e-4},
        {"params": model.skin_dog_head.parameters(),  "lr": LR_HEAD,     "weight_decay": 1e-4},
    ])
    eyes_optimizer = torch.optim.AdamW([
        {"params": model.eyes_backbone.parameters(),  "lr": LR_BACKBONE, "weight_decay": 1e-4},
        {"params": model.eyes_se.parameters(),        "lr": LR_HEAD,     "weight_decay": 1e-4},
        {"params": model.eyes_head.parameters(),      "lr": LR_HEAD,     "weight_decay": 1e-4},
    ])

    # scheduler step 수: cat 배치 + dog 배치 합산
    _skin_cat_batches     = (len(skin_cat_train) + BATCH_SIZE - 1) // BATCH_SIZE
    _skin_dog_batches     = (len(skin_dog_train) + BATCH_SIZE - 1) // BATCH_SIZE
    _skin_steps_per_epoch = _skin_cat_batches + _skin_dog_batches
    _eyes_steps_per_epoch = (len(eyes_train_samples) + BATCH_SIZE - 1) // BATCH_SIZE

    skin_scheduler = get_cosine_schedule_with_warmup(
        skin_optimizer,
        num_warmup_steps  = _skin_steps_per_epoch * 2,          # 2 epoch warmup
        num_training_steps= _skin_steps_per_epoch * EPOCHS,
    )
    eyes_scheduler = get_cosine_schedule_with_warmup(
        eyes_optimizer,
        num_warmup_steps  = _eyes_steps_per_epoch * 3,          # 3 epoch warmup
        num_training_steps= _eyes_steps_per_epoch * EPOCHS,
    )

    skin_scaler = torch.amp.GradScaler("cuda")
    eyes_scaler = torch.amp.GradScaler("cuda")

    history      = []
    best_avg_acc = 0.0
    best_epoch   = 0

    # ── Training Loop ─────────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'='*60}")

        if epoch == FREEZE_EPOCHS:
            model.unfreeze_backbones()
            print(f"  🔓 Backbone unfrozen at epoch {epoch+1} "
                  f"(backbone_lr={LR_BACKBONE}, head_lr={LR_HEAD})")

        # ── 1. Skin (Cat) Training ───────────────────────────────────────────────
        print("\n[Train 1/3] Skin — Cat")
        model.train()

        cat_skin_ds     = AnomalyDataset(skin_cat_train, is_train=True, task="skin")
        cat_skin_loader = DataLoader(
            cat_skin_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=(NUM_WORKERS > 0), prefetch_factor=4,
        )

        cat_loss_sum, cat_correct, cat_total = 0.0, 0, 0
        for images, labels in tqdm(cat_skin_loader, desc=f"  CatSkin Ep{epoch+1:02d}", ncols=110, leave=True):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            skin_optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs = model(images, task="skin_cat")
                loss    = skin_cat_criterion(outputs, labels)
            skin_scaler.scale(loss).backward()
            skin_scaler.unscale_(skin_optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.skin_backbone.parameters()) +
                list(model.skin_cat_head.parameters()), 1.0
            )
            skin_scaler.step(skin_optimizer)
            skin_scaler.update()
            skin_scheduler.step()

            cat_loss_sum += loss.item() * images.size(0)
            cat_correct  += (outputs.argmax(1) == labels).sum().item()
            cat_total    += images.size(0)

        cat_skin_train_loss = cat_loss_sum / cat_total
        cat_skin_train_acc  = cat_correct  / cat_total

        del cat_skin_ds, cat_skin_loader
        gc.collect(); torch.cuda.empty_cache()

        # ── 2. Skin (Dog) Training ───────────────────────────────────────────────
        print("\n[Train 2/3] Skin — Dog")
        model.train()

        dog_skin_ds     = AnomalyDataset(skin_dog_train, is_train=True, task="skin")
        dog_skin_loader = DataLoader(
            dog_skin_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=(NUM_WORKERS > 0), prefetch_factor=4,
        )

        dog_loss_sum, dog_correct, dog_total = 0.0, 0, 0
        for images, labels in tqdm(dog_skin_loader, desc=f"  DogSkin Ep{epoch+1:02d}", ncols=110, leave=True):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            skin_optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs = model(images, task="skin_dog")
                loss    = skin_dog_criterion(outputs, labels)
            skin_scaler.scale(loss).backward()
            skin_scaler.unscale_(skin_optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.skin_backbone.parameters()) +
                list(model.skin_dog_head.parameters()), 1.0
            )
            skin_scaler.step(skin_optimizer)
            skin_scaler.update()
            skin_scheduler.step()

            dog_loss_sum += loss.item() * images.size(0)
            dog_correct  += (outputs.argmax(1) == labels).sum().item()
            dog_total    += images.size(0)

        dog_skin_train_loss = dog_loss_sum / dog_total
        dog_skin_train_acc  = dog_correct  / dog_total

        del dog_skin_ds, dog_skin_loader
        gc.collect(); torch.cuda.empty_cache()

        # ── 3. Eyes Training ─────────────────────────────────────────────────────
        print("\n[Train 3/3] Eyes")
        model.train()

        eyes_train_ds     = AnomalyDataset(eyes_train_samples, is_train=True, task="eyes")
        eyes_train_loader = DataLoader(
            eyes_train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=(NUM_WORKERS > 0), prefetch_factor=4,
        )

        eyes_loss_sum, eyes_correct, eyes_total = 0.0, 0, 0
        for images, labels in tqdm(eyes_train_loader, desc=f"  Eyes  Ep{epoch+1:02d}", ncols=110, leave=True):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            eyes_optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs = model(images, task="eyes")
                loss    = eyes_criterion(outputs, labels)
            eyes_scaler.scale(loss).backward()
            eyes_scaler.unscale_(eyes_optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.eyes_backbone.parameters()) +
                list(model.eyes_se.parameters()) +
                list(model.eyes_head.parameters()), 1.0
            )
            eyes_scaler.step(eyes_optimizer)
            eyes_scaler.update()
            eyes_scheduler.step()

            eyes_loss_sum += loss.item() * images.size(0)
            eyes_correct  += (outputs.argmax(1) == labels).sum().item()
            eyes_total    += images.size(0)

        eyes_train_loss = eyes_loss_sum / eyes_total
        eyes_train_acc  = eyes_correct  / eyes_total

        del eyes_train_ds, eyes_train_loader
        gc.collect(); torch.cuda.empty_cache()

        # ── 4. Validation ────────────────────────────────────────────────────────
        print("\n[Val] Skin (Cat / Dog) & Eyes")
        model.eval()

        def _run_val(ds_samples, task_str, criterion, desc):
            """공통 val 루프 헬퍼."""
            ds     = AnomalyDataset(ds_samples, is_train=False)
            loader = DataLoader(
                ds, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS // 2, pin_memory=True,
                persistent_workers=(NUM_WORKERS // 2 > 0), prefetch_factor=4,
            )
            loss_sum, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in tqdm(loader, desc=desc, ncols=110, leave=False):
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    with torch.amp.autocast("cuda"):
                        outputs = model(images, task=task_str)
                        loss    = criterion(outputs, labels)
                    loss_sum += loss.item() * images.size(0)
                    correct  += (outputs.argmax(1) == labels).sum().item()
                    total    += images.size(0)
            del ds, loader
            gc.collect(); torch.cuda.empty_cache()
            return loss_sum / total, correct / total

        cat_skin_val_loss, cat_skin_val_acc = _run_val(
            skin_cat_val, "skin_cat", skin_cat_criterion, "  CatSkin Val"
        )
        dog_skin_val_loss, dog_skin_val_acc = _run_val(
            skin_dog_val, "skin_dog", skin_dog_criterion, "  DogSkin Val"
        )
        eyes_val_loss, eyes_val_acc = _run_val(
            eyes_val_samples, "eyes", eyes_criterion, "  Eyes    Val"
        )

        skin_val_acc = (cat_skin_val_acc + dog_skin_val_acc) / 2
        avg_val_acc  = (cat_skin_val_acc + dog_skin_val_acc + eyes_val_acc) / 3

        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"  Skin Cat │ Train Loss: {cat_skin_train_loss:.4f}  Acc: {cat_skin_train_acc*100:.2f}%"
              f"  │  Val Loss: {cat_skin_val_loss:.4f}  Acc: {cat_skin_val_acc*100:.2f}%")
        print(f"  Skin Dog │ Train Loss: {dog_skin_train_loss:.4f}  Acc: {dog_skin_train_acc*100:.2f}%"
              f"  │  Val Loss: {dog_skin_val_loss:.4f}  Acc: {dog_skin_val_acc*100:.2f}%")
        print(f"  Eyes     │ Train Loss: {eyes_train_loss:.4f}  Acc: {eyes_train_acc*100:.2f}%"
              f"  │  Val Loss: {eyes_val_loss:.4f}  Acc: {eyes_val_acc*100:.2f}%")
        print(f"  Skin Avg Val Acc: {skin_val_acc*100:.2f}%  │  "
              f"Overall Avg (cat+dog+eyes)/3: {avg_val_acc*100:.2f}%")

        history.append({
            'epoch'               : epoch + 1,
            'cat_skin_train_loss' : cat_skin_train_loss,
            'cat_skin_train_acc'  : cat_skin_train_acc,
            'cat_skin_val_loss'   : cat_skin_val_loss,
            'cat_skin_val_acc'    : cat_skin_val_acc,
            'dog_skin_train_loss' : dog_skin_train_loss,
            'dog_skin_train_acc'  : dog_skin_train_acc,
            'dog_skin_val_loss'   : dog_skin_val_loss,
            'dog_skin_val_acc'    : dog_skin_val_acc,
            'skin_val_acc'        : skin_val_acc,
            'eyes_train_loss'     : eyes_train_loss,
            'eyes_train_acc'      : eyes_train_acc,
            'eyes_val_loss'       : eyes_val_loss,
            'eyes_val_acc'        : eyes_val_acc,
            'avg_val_acc'         : avg_val_acc,
        })

        if avg_val_acc > best_avg_acc:
            best_avg_acc = avg_val_acc
            best_epoch   = epoch + 1
            torch.save(
                {
                    "model"            : model.state_dict(),
                    "skin_optimizer"   : skin_optimizer.state_dict(),
                    "eyes_optimizer"   : eyes_optimizer.state_dict(),
                    "skin_scheduler"   : skin_scheduler.state_dict(),
                    "eyes_scheduler"   : eyes_scheduler.state_dict(),
                    "epoch"            : epoch + 1,
                    "best_avg_acc"     : best_avg_acc,
                    "skin_classes"     : SKIN_CLASSES,
                    "cat_skin_classes" : CAT_SKIN_CLASSES,
                    "dog_skin_classes" : DOG_SKIN_CLASSES,
                    "eyes_classes"     : EYES_CLASSES,
                    "work_dir"         : WORK_DIR,
                    "history"          : history,
                },
                "pet_abnormal_omni_best.pth",
            )
            print(f"  💾 Saved best model! (Epoch {best_epoch} | Val Avg Acc: {best_avg_acc*100:.2f}%)")

    print(f"\n🏆 Training Finished.")
    print(f"   Best Epoch: {best_epoch} | Best Val Avg Acc: {best_avg_acc*100:.2f}%")
    print(f"   Test set: {os.path.join(WORK_DIR, 'test')}/")

    # ── 학습 곡선 시각화 ──────────────────────────────────────────────────────────
    print("\n📈 Generating training history plot...")

    epochs_x = [h['epoch'] for h in history]
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # Row 0: Loss
    axes[0, 0].plot(epochs_x, [h['cat_skin_train_loss'] for h in history], 'b-',  label='Train')
    axes[0, 0].plot(epochs_x, [h['cat_skin_val_loss']   for h in history], 'b--', label='Val')
    axes[0, 0].axvline(best_epoch, color='gray', linestyle=':', alpha=0.7)
    axes[0, 0].set_title('Skin Cat Loss'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs_x, [h['dog_skin_train_loss'] for h in history], 'g-',  label='Train')
    axes[0, 1].plot(epochs_x, [h['dog_skin_val_loss']   for h in history], 'g--', label='Val')
    axes[0, 1].axvline(best_epoch, color='gray', linestyle=':', alpha=0.7)
    axes[0, 1].set_title('Skin Dog Loss'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(epochs_x, [h['eyes_train_loss'] for h in history], 'r-',  label='Train')
    axes[0, 2].plot(epochs_x, [h['eyes_val_loss']   for h in history], 'r--', label='Val')
    axes[0, 2].axvline(best_epoch, color='gray', linestyle=':', alpha=0.7)
    axes[0, 2].set_title('Eyes Loss'); axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)

    # Row 1: Accuracy
    axes[1, 0].plot(epochs_x, [h['cat_skin_train_acc'] for h in history], 'b-',  label='Train')
    axes[1, 0].plot(epochs_x, [h['cat_skin_val_acc']   for h in history], 'b--', label='Val')
    axes[1, 0].axvline(best_epoch, color='gray', linestyle=':', alpha=0.7)
    axes[1, 0].set_title('Skin Cat Accuracy'); axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs_x, [h['dog_skin_train_acc'] for h in history], 'g-',  label='Train')
    axes[1, 1].plot(epochs_x, [h['dog_skin_val_acc']   for h in history], 'g--', label='Val')
    axes[1, 1].axvline(best_epoch, color='gray', linestyle=':', alpha=0.7)
    axes[1, 1].set_title('Skin Dog Accuracy'); axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(epochs_x, [h['eyes_train_acc'] for h in history], 'r-',  label='Train')
    axes[1, 2].plot(epochs_x, [h['eyes_val_acc']   for h in history], 'r--', label='Val')
    axes[1, 2].axvline(best_epoch, color='gray', linestyle=':', alpha=0.7)
    axes[1, 2].set_title('Eyes Accuracy'); axes[1, 2].set_ylim(0, 1)
    axes[1, 2].legend(); axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(
        f'Anomaly Model v2 (EfficientNet-V2-S) | Best Ep {best_epoch} | Avg {best_avg_acc*100:.1f}%',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('pet_abnormal_omni_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: pet_abnormal_omni_history.png")


def _load_samples_from_dir(task_dir: str, class_names: list) -> list:
    """
    WORK_DIR 하위 task 폴더에서 samples 리스트를 복원한다.
    (재학습 시 파일 복사를 skip하고 기존 WORK_DIR에서 바로 로드)
    """
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    samples     = []
    for class_name in sorted(os.listdir(task_dir)):
        class_dir = os.path.join(task_dir, class_name)
        if not os.path.isdir(class_dir) or class_name not in name_to_idx:
            continue
        label_idx = name_to_idx[class_name]
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(class_dir, fname), label_idx))
    return samples


# ===============================
# ENTRY POINT
# ===============================

if __name__ == "__main__":
    train()
