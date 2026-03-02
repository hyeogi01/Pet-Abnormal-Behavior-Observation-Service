"""
cat_abnormal_omni_train.py
────────────────────────────────────────────────────────────────────
고양이 이상 증상 분류 모델 (Skin / Eyes)

[JSON 전략]
  학습 시:
    normal class → JSON 없음 → feat = zeros (항상)
    disease class → JSON 있음 → feat = 실제값
                              + JSON Modality Dropout(p=0.5):
                                50% 확률로 feat를 zeros로 대체
  추론 시:
    모든 class → feat = zeros (JSON 입력 없음)
    → 모델이 학습에서 zeros 상황을 충분히 경험했으므로 robust

[Modality Dropout 효과]
  - 이미지 특징이 주된 분류 근거로 강제됨
  - JSON은 학습 시 refinement 신호로 기여 (추론 시 없어도 됨)
  - normal(zeros) + disease(50% zeros) 혼합 → "zeros = no info" 학습
────────────────────────────────────────────────────────────────────
"""

import os, gc, json, random, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import torch.optim.lr_scheduler as lr_scheduler
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ───────────────────────────────── CONFIG ─────────────────────────────────────
SEED = 42
random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED)

SKIN_ROOT = "files/4_Animal_Skin"
EYES_ROOT = "files/5_Animal_Eyes"
WORK_DIR  = "files/work/cat_abnormal_dataset"

DEVICE               = "cuda:0" if torch.cuda.is_available() else "cpu"
EPOCHS               = 100
BATCH_SIZE           = 64
NUM_WORKERS          = 24
LABEL_SMOOTHING      = 0.1
LABEL_SMOOTHING_EYES = 0.05
FREEZE_EPOCHS        = 8
WEIGHT_DECAY         = 5e-4
LR_BACKBONE          = 2e-5
LR_HEAD              = 2e-4

JSON_FEAT_DIM        = 5
# 학습 시 disease class JSON을 50% 확률로 zeros로 대체
# → 추론 시 zeros 입력에 robust한 모델 학습
JSON_TRAIN_DROPOUT   = 0.5

# normal class 이름 (JSON 제공 안 됨 → 항상 zeros)
NORMAL_CLASS_NAMES   = {"cat_normal"}

IMG_SIZE   = 384
IMG_RESIZE = 416

print(f"🐱 Cat Abnormal Omni | Device: {DEVICE}")
print(f"   Skin+Eyes: EfficientNet+JSON Fusion")
print(f"   JSON strategy: disease=real feat, normal=zeros, inference=zeros")
print(f"   JSON Modality Dropout: p={JSON_TRAIN_DROPOUT}")

# ──────────────────────────── BREED SETS ─────────────────────────────────────
BRACHYCEPHALIC_CATS = {
    '페르시안', '히말라얀', '엑조틱숏헤어', '브리티시숏헤어', '스코티시폴드',
    '버미즈', '봄베이'
}
LONGHAIRED_CATS = {
    '페르시안', '히말라얀', '메인쿤', '노르웨이숲', '시베리안', '랙돌',
    '터키시앙고라', '버만'
}

# ──────────────────────────────── CLASSES ─────────────────────────────────────
CAT_SKIN_CLASSES = [
    "cat_normal",
    "cat_결절,종괴",
    "cat_농포,여드름",
    "cat_비듬,각질,상피성잔고리",
]  # 4클래스

CAT_EYES_CLASSES = [
    "cat_normal",
    "cat_각막궤양",
    "cat_각막부골편",
    "cat_결막염",
    "cat_비궤양성각막염",
    "cat_안검염",
]  # 6클래스

EYES_SIMILAR_GROUPS = [
    ["cat_각막궤양", "cat_각막부골편", "cat_비궤양성각막염"],
]

VAL_RATIO  = 0.1
TEST_RATIO = 0.1

# ─────────────────────────────── LOSS ────────────────────────────────────────
class WeightedSmoothLoss(nn.Module):
    def __init__(self, class_names, similar_groups=None, class_weights=None,
                 smoothing=LABEL_SMOOTHING, penalty=1.5):
        super().__init__()
        self.smoothing = smoothing; self.penalty = penalty
        self.name_to_idx = {n: i for i, n in enumerate(class_names)}
        self.penalty_pairs = set()
        if similar_groups:
            for g in similar_groups:
                idxs = [self.name_to_idx[n] for n in g if n in self.name_to_idx]
                for i in range(len(idxs)):
                    for j in range(i+1, len(idxs)):
                        self.penalty_pairs.add((idxs[i], idxs[j]))
                        self.penalty_pairs.add((idxs[j], idxs[i]))
        self.register_buffer("weight", class_weights)

    def forward(self, logits, targets):
        log_p  = F.log_softmax(logits, dim=-1)
        nll    = F.nll_loss(log_p, targets, weight=self.weight, reduction="none")
        smooth = -log_p.mean(dim=-1)
        loss   = (1 - self.smoothing) * nll + self.smoothing * smooth
        if self.penalty_pairs:
            preds = logits.argmax(1)
            mask  = torch.ones(len(targets), device=logits.device)
            for b in range(len(targets)):
                if (targets[b].item(), preds[b].item()) in self.penalty_pairs:
                    mask[b] = self.penalty
            loss = loss * mask
        return loss.mean()

# ─────────────────────────────── BACKBONE ────────────────────────────────────
def _efficientnet_backbone():
    b = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    feat = b.classifier[1].in_features  # 1280
    b.classifier = nn.Identity()
    return b, feat

# ─────────────────────────── JSON PARSING ─────────────────────────────────────
def _parse_json_feat(json_path: str) -> np.ndarray:
    """
    5-dim: [age_norm, gender, eye_pos, is_brachycephalic, is_longhaired]

    disease class: JSON 파일 존재 → 실제값 반환
    normal class / JSON 없음: zeros 반환 (no-info 표현)
    → zeros = "JSON 정보 없음" 으로 모델이 학습
    """
    zeros = np.zeros(5, dtype=np.float32)
    if not os.path.exists(json_path):
        return zeros   # JSON 없음 → zeros (normal class 또는 누락)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        meta    = data.get('images', {}).get('meta', {})
        age     = float(meta.get('age', 7)) / 20.0
        gender  = float(meta.get('gender', 0))
        breed   = meta.get('breed', '')
        eye_str = meta.get('eye_position', '')
        eye_pos   = 0.0 if '왼' in eye_str else (1.0 if '오' in eye_str else 0.5)
        is_brachy = 1.0 if any(b in breed for b in BRACHYCEPHALIC_CATS) else 0.0
        is_lh     = 1.0 if any(b in breed for b in LONGHAIRED_CATS)     else 0.0
        return np.array([age, gender, eye_pos, is_brachy, is_lh], dtype=np.float32)
    except Exception:
        return zeros

# ─────────────────────────────── MODELS ───────────────────────────────────────
class _FusionBase(nn.Module):
    """
    EfficientNet(1280) + JSON branch(5→32) → concat(1312) → head

    JSON Modality Dropout (학습 시만):
      - 배치 내 각 샘플에 대해 독립적으로 적용
      - JSON 있는 disease 샘플도 50% 확률로 zeros 처리
      - 추론 시 feat=zeros 입력에 robust한 모델 학습

    추론 시:
      - feat = zeros (JSON 없음) → self.training=False → dropout 미적용
      - 모델은 이미지 특징만으로 분류
    """
    def __init__(self, num_classes, json_feat_dim, head_hidden,
                 json_dropout_prob=JSON_TRAIN_DROPOUT):
        super().__init__()
        self.json_dropout_prob = json_dropout_prob
        self.backbone, img_feat = _efficientnet_backbone()
        self.feat_branch = nn.Sequential(
            nn.Linear(json_feat_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.GELU(),
        )
        fused = img_feat + 32   # 1312
        layers = [nn.Dropout(0.5), nn.Linear(fused, head_hidden[0]),
                  nn.BatchNorm1d(head_hidden[0]), nn.GELU()]
        for i in range(1, len(head_hidden)):
            layers += [nn.Dropout(0.4), nn.Linear(head_hidden[i-1], head_hidden[i]),
                       nn.BatchNorm1d(head_hidden[i]), nn.GELU()]
        layers += [nn.Dropout(0.4), nn.Linear(head_hidden[-1], num_classes)]
        self.head = nn.Sequential(*layers)

    def forward(self, img, feat):
        img_f = self.backbone(img)

        # JSON Modality Dropout: 학습 시에만 적용
        # 배치 내 각 샘플별로 독립적으로 feat를 zeros로 대체
        if self.training and self.json_dropout_prob > 0:
            # (B, 1) 마스크: json_dropout_prob 확률로 0 (feat 제거)
            keep_mask = (
                torch.rand(feat.size(0), 1, device=feat.device)
                > self.json_dropout_prob
            ).float()
            feat = feat * keep_mask

        json_f = self.feat_branch(feat)
        return self.head(torch.cat([img_f, json_f], dim=1))


class SkinModel(_FusionBase):
    """Head: 1312→256→4 (cat 4클래스)"""
    def __init__(self, num_classes, json_feat_dim=JSON_FEAT_DIM):
        super().__init__(num_classes, json_feat_dim, head_hidden=[256])


class EyesModel(_FusionBase):
    """Head: 1312→512→6 (cat 6클래스)"""
    def __init__(self, num_classes, json_feat_dim=JSON_FEAT_DIM):
        super().__init__(num_classes, json_feat_dim, head_hidden=[512])

# ─────────────────────────────── TRANSFORMS ───────────────────────────────────
SKIN_TRANSFORM_TRAIN = transforms.Compose([
    transforms.Resize((IMG_RESIZE, IMG_RESIZE)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=90),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])
EYES_TRANSFORM_TRAIN = transforms.Compose([
    transforms.Resize((IMG_RESIZE, IMG_RESIZE)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
TRANSFORM_VAL = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─────────────────────────────── DATASETS ─────────────────────────────────────
class FusionDataset(Dataset):
    """
    학습/검증용 공용 데이터셋.

    JSON 전략:
      normal class → json_path 조회 안 함 → feat = zeros
      disease class → json_path 조회 → 실제값 (없으면 zeros)

    Modality Dropout은 모델 forward에서 처리.
    val/test에서도 동일 구조 사용 (val: 실제 JSON feat 그대로 사용 → 학습 품질 평가)
    """
    def __init__(self, samples, transform, normal_class_names=NORMAL_CLASS_NAMES,
                 class_list=None):
        self.samples = samples
        self.transform = transform
        # label index → normal 여부 판단
        self.normal_indices = set()
        if class_list:
            for i, name in enumerate(class_list):
                if name in normal_class_names:
                    self.normal_indices.add(i)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        p, l = self.samples[idx]
        img  = self.transform(Image.open(p).convert("RGB"))
        # normal class: JSON 없음 → zeros / disease class: JSON 읽기
        if l in self.normal_indices:
            feat = np.zeros(JSON_FEAT_DIM, dtype=np.float32)
        else:
            feat = _parse_json_feat(os.path.splitext(p)[0] + ".json")
        return img, torch.tensor(feat, dtype=torch.float32), l


class InferenceDataset(Dataset):
    """
    추론 전용 데이터셋.
    feat = zeros (항상): JSON 없음 상황을 재현.
    학습 시 Modality Dropout으로 이 상황을 경험했으므로 robust.
    """
    def __init__(self, samples, transform):
        self.samples = samples; self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        p, l = self.samples[idx]
        img  = self.transform(Image.open(p).convert("RGB"))
        feat = np.zeros(JSON_FEAT_DIM, dtype=np.float32)   # 항상 zeros
        return img, torch.tensor(feat, dtype=torch.float32), l

# ─────────────────────────── DATA PREPARATION ─────────────────────────────────
def _task_ready(name):
    p = os.path.join(WORK_DIR, "train", name)
    return os.path.isdir(p) and len(os.listdir(p)) > 0

def collect_copy_split(src_root, task_name, class_list):
    rng = random.Random(SEED)
    name_to_idx = {n: i for i, n in enumerate(class_list)}
    class_files = defaultdict(list); seen = set()
    for cls in class_list:
        d = os.path.join(src_root, cls)
        if not os.path.isdir(d): continue
        for f in os.listdir(d):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")): continue
            fp = os.path.join(d, f)
            if fp in seen: continue
            seen.add(fp); class_files[name_to_idx[cls]].append(fp)
    for split in ["train", "val", "test"]:
        for cls in class_list:
            os.makedirs(os.path.join(WORK_DIR, split, task_name, cls), exist_ok=True)
    train_s, val_s = [], []
    for idx, paths in class_files.items():
        cls = class_list[idx]; rng.shuffle(paths); n = len(paths)
        nv = max(1, int(n * VAL_RATIO)); nt = max(1, int(n * TEST_RATIO))
        splits_map = {"train": paths[:n-nv-nt], "val": paths[n-nv-nt:n-nt], "test": paths[n-nt:]}
        for sname, spaths in splits_map.items():
            dst = os.path.join(WORK_DIR, sname, task_name, cls)
            for src in spaths:
                d_img = os.path.join(dst, os.path.basename(src))
                if not os.path.exists(d_img): shutil.copy2(src, d_img)
                # disease class만 JSON 복사 (normal은 JSON 없음)
                if cls not in NORMAL_CLASS_NAMES:
                    json_src = os.path.splitext(src)[0] + ".json"
                    json_dst = os.path.splitext(d_img)[0] + ".json"
                    if os.path.exists(json_src) and not os.path.exists(json_dst):
                        shutil.copy2(json_src, json_dst)
            wp = [(os.path.join(dst, os.path.basename(s)), idx) for s in spaths]
            if sname == "train": train_s.extend(wp)
            elif sname == "val":  val_s.extend(wp)
    print(f"  ✅ {task_name}: train={len(train_s)} val={len(val_s)}")
    return train_s, val_s

def load_from_dir(task_dir, class_list):
    name_to_idx = {n: i for i, n in enumerate(class_list)}
    samples = []
    for cls in sorted(os.listdir(task_dir)):
        d = os.path.join(task_dir, cls)
        if not os.path.isdir(d) or cls not in name_to_idx: continue
        for f in os.listdir(d):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(d, f), name_to_idx[cls]))
    return samples

def class_weights_from_samples(samples, n_classes):
    labels = [l for _, l in samples]
    w = compute_class_weight('balanced', classes=np.arange(n_classes), y=labels)
    return torch.tensor(w, dtype=torch.float).to(DEVICE)

# ──────────────────────────────── HELPERS ─────────────────────────────────────
def make_loader(samples, transform, shuffle, drop_last=False, class_list=None):
    ds = FusionDataset(samples, transform, class_list=class_list)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True,
                      persistent_workers=True, prefetch_factor=2,
                      multiprocessing_context="fork", drop_last=drop_last)

def clear(): gc.collect(); torch.cuda.empty_cache()

# ─────────────────────────── HISTORY PLOT ─────────────────────────────────────
def _save_history_plot(history, best_acc):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    ax.plot([h["skin_train"] for h in history], 'b--', linewidth=1.5, alpha=0.7, label="Skin Train")
    ax.plot([h["skin_val"]   for h in history], 'b-',  linewidth=2.5, label="Skin Val")
    ax.set_ylim(0, 1); ax.set_title("Skin Accuracy (Train vs Val)"); ax.legend(); ax.grid(True, alpha=0.3)
    if len(history) > 1:
        ax.set_xlabel(f"Epoch  |  gap={(history[-1]['skin_train']-history[-1]['skin_val'])*100:.1f}%")
    ax = axes[1]
    ax.plot([h["eyes_train"] for h in history], 'r--', linewidth=1.5, alpha=0.7, label="Eyes Train")
    ax.plot([h["eyes_val"]   for h in history], 'r-',  linewidth=2.5, label="Eyes Val")
    ax.set_ylim(0, 1); ax.set_title("Eyes Accuracy (Train vs Val)"); ax.legend(); ax.grid(True, alpha=0.3)
    if len(history) > 1:
        ax.set_xlabel(f"Epoch  |  gap={(history[-1]['eyes_train']-history[-1]['eyes_val'])*100:.1f}%")
    ax = axes[2]
    ax.plot([h["skin_lr"] for h in history], 'b-', linewidth=1.5, label="Skin LR (head)")
    ax.plot([h["eyes_lr"] for h in history], 'r-', linewidth=1.5, label="Eyes LR (head)")
    ax.set_title("Learning Rate (WarmRestarts T₀=20)"); ax.legend(); ax.grid(True, alpha=0.3)
    for restart in [20, 40, 60, 80]:
        if restart < len(history):
            ax.axvline(x=restart, color='gray', linestyle=':', alpha=0.5)
    plt.suptitle(f"Cat Abnormal Omni | Best Avg {best_acc*100:.1f}%", fontweight="bold")
    plt.tight_layout(); plt.savefig("cat_abnormal_omni_history.png", dpi=150); plt.close()

# ─────────────────────────────── TRAIN/VAL ────────────────────────────────────
def val_loop(model, samples, transform, class_list):
    """
    val: 실제 JSON feat 사용 (학습 품질 평가용).
    Modality Dropout 미적용 (model.eval() → self.training=False).
    """
    loader = make_loader(samples, transform, shuffle=False,
                         drop_last=False, class_list=class_list)
    model.to(DEVICE).eval(); c, t = 0, 0
    with torch.no_grad():
        for imgs, feats, labels in loader:
            imgs, feats, labels = imgs.to(DEVICE), feats.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast("cuda"):
                out = model(imgs, feats)
            c += (out.argmax(1) == labels).sum().item(); t += labels.size(0)
    del loader; clear(); model.cpu(); clear()
    return c / t if t > 0 else 0.0

# ─────────────────────────────── TRAINING ─────────────────────────────────────
def train():
    if _task_ready("skin"):
        print("✅ skin ready")
        skin_train = load_from_dir(os.path.join(WORK_DIR, "train", "skin"), CAT_SKIN_CLASSES)
        skin_val   = load_from_dir(os.path.join(WORK_DIR, "val",   "skin"), CAT_SKIN_CLASSES)
    else:
        print("📦 Preparing skin (cat)...")
        skin_train, skin_val = collect_copy_split(SKIN_ROOT, "skin", CAT_SKIN_CLASSES)

    if _task_ready("eyes"):
        print("✅ eyes ready")
        eyes_train = load_from_dir(os.path.join(WORK_DIR, "train", "eyes"), CAT_EYES_CLASSES)
        eyes_val   = load_from_dir(os.path.join(WORK_DIR, "val",   "eyes"), CAT_EYES_CLASSES)
    else:
        print("📦 Preparing eyes (cat)...")
        eyes_train, eyes_val = collect_copy_split(EYES_ROOT, "eyes", CAT_EYES_CLASSES)

    skin_w    = class_weights_from_samples(skin_train, len(CAT_SKIN_CLASSES))
    eyes_w    = class_weights_from_samples(eyes_train, len(CAT_EYES_CLASSES))
    skin_crit = WeightedSmoothLoss(CAT_SKIN_CLASSES, class_weights=skin_w,
                                    smoothing=LABEL_SMOOTHING)
    eyes_crit = WeightedSmoothLoss(CAT_EYES_CLASSES, similar_groups=EYES_SIMILAR_GROUPS,
                                    class_weights=eyes_w, smoothing=LABEL_SMOOTHING_EYES)

    skin_model = SkinModel(len(CAT_SKIN_CLASSES), json_feat_dim=JSON_FEAT_DIM)
    eyes_model = EyesModel(len(CAT_EYES_CLASSES), json_feat_dim=JSON_FEAT_DIM)

    for m in [skin_model, eyes_model]:
        for p in m.backbone.parameters(): p.requires_grad = False
    print(f"  🔒 Backbone frozen for {FREEZE_EPOCHS} epochs")

    def make_opt(m):
        return torch.optim.AdamW([
            {"params": m.backbone.parameters(),
             "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
            {"params": [p for n, p in m.named_parameters() if "backbone" not in n],
             "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY},
        ])
    skin_opt = make_opt(skin_model); eyes_opt = make_opt(eyes_model)

    def make_sched(opt):
        return lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=1, eta_min=1e-7)
    skin_sched = make_sched(skin_opt); eyes_sched = make_sched(eyes_opt)

    skin_scaler = torch.amp.GradScaler("cuda")
    eyes_scaler = torch.amp.GradScaler("cuda")
    best_acc, history = 0.0, []

    for epoch in range(EPOCHS):
        print(f"\n{'='*55}\nEpoch {epoch+1}/{EPOCHS}\n{'='*55}")
        if epoch == FREEZE_EPOCHS:
            for m in [skin_model, eyes_model]:
                for p in m.backbone.parameters(): p.requires_grad = True
            print("  🔓 Backbone unfrozen")

        def train_task(name, model, samples, crit, transform, opt,
                       scaler, sched, epoch_float, class_list):
            model.to(DEVICE).train()
            loader = make_loader(samples, transform, shuffle=True,
                                 drop_last=True, class_list=class_list)
            ls, c, t = 0, 0, 0; n_batches = len(loader)
            for bi, (imgs, feats, labels) in enumerate(
                    tqdm(loader, desc=f"  {name}", ncols=100, leave=True)):
                imgs, feats, labels = imgs.to(DEVICE), feats.to(DEVICE), labels.to(DEVICE)
                opt.zero_grad()
                with torch.amp.autocast("cuda"):
                    # Modality Dropout은 model.forward 내부에서 자동 적용
                    # (self.training=True → 배치 내 각 샘플 50% 확률로 feat zeros)
                    out = model(imgs, feats); loss = crit(out, labels)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                prev = scaler.get_scale(); scaler.step(opt); scaler.update()
                if scaler.get_scale() == prev: sched.step(epoch_float + bi / n_batches)
                ls += loss.item(); c += (out.argmax(1)==labels).sum().item(); t += labels.size(0)
            del loader; clear(); model.cpu(); clear()
            return ls / (t // BATCH_SIZE + 1), c / t

        print("\n[1/2] Skin")
        s_loss, s_tacc = train_task("Skin", skin_model, skin_train, skin_crit,
                                    SKIN_TRANSFORM_TRAIN, skin_opt, skin_scaler, skin_sched,
                                    epoch, CAT_SKIN_CLASSES)
        print("\n[2/2] Eyes")
        e_loss, e_tacc = train_task("Eyes", eyes_model, eyes_train, eyes_crit,
                                    EYES_TRANSFORM_TRAIN, eyes_opt, eyes_scaler, eyes_sched,
                                    epoch, CAT_EYES_CLASSES)

        skin_vacc = val_loop(skin_model, skin_val, TRANSFORM_VAL, CAT_SKIN_CLASSES); clear()
        eyes_vacc = val_loop(eyes_model, eyes_val, TRANSFORM_VAL, CAT_EYES_CLASSES); clear()
        avg = (skin_vacc + eyes_vacc) / 2

        skin_lr = skin_opt.param_groups[1]["lr"]
        eyes_lr = eyes_opt.param_groups[1]["lr"]
        print(f"\n📊 Ep{epoch+1} | Skin Train {s_tacc*100:.1f}% Val {skin_vacc*100:.1f}% LR={skin_lr:.2e}"
              f" | Eyes Train {e_tacc*100:.1f}% Val {eyes_vacc*100:.1f}% LR={eyes_lr:.2e}"
              f" | Avg {avg*100:.1f}%")

        history.append({"epoch": epoch+1,
                         "skin_train": s_tacc, "skin_val": skin_vacc,
                         "eyes_train": e_tacc, "eyes_val": eyes_vacc,
                         "avg": avg, "skin_lr": skin_lr, "eyes_lr": eyes_lr})

        if avg > best_acc:
            best_acc = avg
            torch.save({
                "skin_model":         skin_model.state_dict(),
                "eyes_model":         eyes_model.state_dict(),
                "cat_skin_classes":   CAT_SKIN_CLASSES,
                "cat_eyes_classes":   CAT_EYES_CLASSES,
                "json_feat_dim":      JSON_FEAT_DIM,
                "json_train_dropout": JSON_TRAIN_DROPOUT,
                "best_epoch":         epoch+1,
                "best_acc":           best_acc,
                "history":            history,
            }, "cat_abnormal_omni_best.pth")
            print(f"  💾 Saved! Avg {best_acc*100:.1f}%")

        _save_history_plot(history, best_acc)

    print(f"\n🎉 Done! Best Avg: {best_acc*100:.1f}%")


if __name__ == "__main__":
    train()
