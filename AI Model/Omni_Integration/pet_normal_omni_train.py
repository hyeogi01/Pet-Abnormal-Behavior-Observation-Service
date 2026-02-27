import os
import random
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, get_linear_schedule_with_warmup
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image, ImageFile
import librosa
import numpy as np
from collections import defaultdict, Counter
from sklearn.utils.class_weight import compute_class_weight
import json
import gc

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# 0. 설정
# =========================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

BEHAVIOR_ROOT = "files/1_Animal_Behavior"
EMOTION_ROOT  = "files/2_Animal_emotions"
SOUND_ROOT    = "files/3_Animal_Sound"
PATELLA_ROOT  = "files/6_Animal_Patella"
WORK_DIR      = "files/work/normal_dataset"

BATCH_SIZE  = 64
EPOCHS      = 100
LR_VIDEO    = 5e-5
LR_AUDIO    = 1e-5
DEVICE      = "cuda:1" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 24
SR          = 16000
MAX_AUDIO_LEN = SR * 5

LOSS_WEIGHTS = {
    "behavior": 1.0,
    "emotion":  1.0,
    "sound":    1.0,  # [FIX v2] 0.6 → 1.0. 다운스케일이 sound 학습 억제 원인이었음.
    "patella":  1.0,
}

AUDIO_MODEL_NAME = "facebook/wav2vec2-base"
FEATURE_EXTRACTOR = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)

print(f"🎯 Device: {DEVICE}")

# =========================
# 🔥 Audio Augmentation
# =========================
def augment_audio(waveform, p=0.5):
    if random.random() > p:
        return waveform

    n_steps = random.uniform(-2, 2)
    waveform = librosa.effects.pitch_shift(waveform, sr=SR, n_steps=n_steps)

    rate = random.uniform(0.9, 1.1)
    stretched = librosa.effects.time_stretch(waveform, rate=rate)
    if len(stretched) > MAX_AUDIO_LEN:
        stretched = stretched[:MAX_AUDIO_LEN]
    else:
        stretched = np.pad(stretched, (0, MAX_AUDIO_LEN - len(stretched)))
    waveform = stretched

    noise = np.random.normal(0, 0.003, len(waveform))
    waveform = waveform * 0.99 + noise

    return waveform

# =========================
# 1. Dataset Preparation
# =========================
def collect_samples(root, exts):
    samples = []
    for class_dir in sorted(os.listdir(root)):
        class_path = os.path.join(root, class_dir)
        if not os.path.isdir(class_path):
            continue

        for root_dir, _, files in os.walk(class_path):
            for filename in files:
                if any(filename.lower().endswith(ext) for ext in exts):
                    file_path = os.path.join(root_dir, filename)
                    samples.append((class_dir, file_path))

    print(f"  → {len(samples)} samples, {len(set(s[0] for s in samples))} classes")
    return samples


def collect_patella_by_date(root):
    """
    날짜 폴더를 분리 단위로 사용하여 데이터 누수를 차단.

    구조: root / grade / date_dir / direction / *.jpg + *.json

    반환:
        { grade: [ (date_dir, [(img_path, json_path), ...]), ... ] }

    - 동일 날짜(= 동일 개체)의 이미지가 train/val/test에 분산되는 것을 방지.
    - JSON 쌍이 없는 이미지는 제외.
    """
    grade_date_map = defaultdict(list)

    for grade in sorted(os.listdir(root)):
        grade_path = os.path.join(root, grade)
        if not os.path.isdir(grade_path):
            continue

        for date_dir in sorted(os.listdir(grade_path)):
            date_path = os.path.join(grade_path, date_dir)
            if not os.path.isdir(date_path):
                continue

            date_samples = []
            # [FIX] 하드코딩된 direction 이름 대신 실제 존재하는 하위 폴더 전체 탐색.
            # 폴더명 대소문자/철자 차이로 샘플이 조용히 누락되는 문제 방지.
            for direction in sorted(os.listdir(date_path)):
                direction_path = os.path.join(date_path, direction)
                if not os.path.isdir(direction_path):
                    continue
                for filename in os.listdir(direction_path):
                    if filename.lower().endswith('.jpg'):
                        img_path  = os.path.join(direction_path, filename)
                        json_path = os.path.splitext(img_path)[0] + '.json'  # [FIX] splitext 사용
                        if os.path.exists(json_path):
                            date_samples.append((img_path, json_path))

            if date_samples:
                grade_date_map[grade].append((date_dir, date_samples))

    total_files = sum(
        len(samples)
        for dates in grade_date_map.values()
        for _, samples in dates
    )
    total_dates = sum(len(dates) for dates in grade_date_map.values())
    print(f"  → {total_files} files | {total_dates} date-folders | "
          f"{len(grade_date_map)} classes")
    for grade, dates in sorted(grade_date_map.items()):
        n_files = sum(len(s) for _, s in dates)
        print(f"    {grade}: {len(dates)} date-folders, {n_files} files")
    return grade_date_map


def split_and_copy_patella(grade_date_map):
    """
    날짜 폴더 단위로 train(70%) / val(15%) / test(15%) 분리 후 복사.

    [데이터 누수 방지]
    - 분리 단위 = 날짜 폴더(동일 개체의 촬영 묶음)
    - 동일 날짜의 이미지가 여러 split에 섞이는 문제 완전 차단.
    - 파일명 충돌 방지: {grade}_{date_dir}_{original_filename} 으로 저장.
    """
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(WORK_DIR, split, "patella"), exist_ok=True)

    total_stats = defaultdict(lambda: defaultdict(int))  # grade → split → file count

    for grade, date_list in grade_date_map.items():
        # 날짜 단위 셔플 (재현성을 위해 SEED 고정 후 호출됨)
        random.shuffle(date_list)

        n = len(date_list)
        n_train = int(n * 0.70)
        n_val   = int(n * 0.15)
        # test = 나머지 (약 15%)

        split_dates = {
            "train": date_list[:n_train],
            "val":   date_list[n_train : n_train + n_val],
            "test":  date_list[n_train + n_val :],
        }

        for split_name, dates in split_dates.items():
            dst_label_dir = os.path.join(WORK_DIR, split_name, "patella", grade)
            os.makedirs(dst_label_dir, exist_ok=True)

            for date_dir, samples in tqdm(
                dates,
                desc=f"patella/{split_name}/{grade}",
                leave=False
            ):
                for img_path, json_path in samples:
                    # 파일명 충돌 방지: grade_dateDir_originalFilename
                    base     = f"{grade}_{date_dir}_{os.path.basename(img_path)}"
                    dst_img  = os.path.join(dst_label_dir, base)
                    dst_json = os.path.splitext(dst_img)[0] + '.json'  # [FIX] splitext 사용
                    shutil.copy(img_path,  dst_img)
                    shutil.copy(json_path, dst_json)
                    total_stats[grade][split_name] += 1

    # 결과 출력
    print("\n  📊 Patella split result (date-folder unit):")
    grand_total = defaultdict(int)
    for grade in sorted(total_stats):
        s = total_stats[grade]
        total = sum(s.values())
        print(f"    {grade}: train={s['train']} | val={s['val']} | "
              f"test={s['test']} | total={total}")
        for split_name, cnt in s.items():
            grand_total[split_name] += cnt
    print(f"    [ALL] train={grand_total['train']} | val={grand_total['val']} | "
          f"test={grand_total['test']} | "
          f"total={sum(grand_total.values())}")


def sample_balanced(samples):
    """샘플링 없이 전체 데이터 반환. 불균형은 학습 시 class_weight로 보정."""
    class_counts = defaultdict(int)
    for label, _ in samples:
        class_counts[label] += 1
    print(f"  📊 {len(class_counts)} classes, total {len(samples)} samples (all used)")
    for label, cnt in sorted(class_counts.items()):
        print(f"    {label}: {cnt}")
    return samples

def sample_balanced_audio(samples, target_per_class=200):
    """
    [FIX v2] 오디오 오버샘플링 적용.
    - 기존: 전체 반환 (788개 그대로) → pet_sound_train.py 대비 학습 데이터 절대량 부족
    - 수정: 소수 클래스를 target_per_class까지 random.choices로 오버샘플링
    - pet_sound_train.py에서 동일 방식으로 150 epoch 기준 90%+ 달성 확인됨.
    - target_per_class=200: dog_bark(202개) 기준으로 전 클래스 균등화
    """
    class_samples = defaultdict(list)
    for label, path in samples:
        class_samples[label].append((label, path))

    result = []
    print(f"  📊 Audio oversampling (target={target_per_class}/class):")
    for label, items in sorted(class_samples.items()):
        original_cnt = len(items)
        if original_cnt < target_per_class:
            # 부족한 만큼 반복 샘플링 (augmentation 효과)
            oversampled = random.choices(items, k=target_per_class)
        else:
            oversampled = items
        result.extend(oversampled)
        print(f"    {label}: {original_cnt} → {len(oversampled)}")

    print(f"  📊 Total after oversampling: {len(result)} samples")
    return result


def _dedup_samples(samples):
    """
    파일 경로 기준 중복 제거.
    os.walk로 재귀 수집 시 동일 파일이 중복 등록되는 경우를 방지.
    → train/val/test 분리 전에 반드시 호출해 데이터 누수 원천 차단.
    """
    seen = set()
    deduped = []
    for label, path in samples:
        if path not in seen:
            seen.add(path)
            deduped.append((label, path))
    removed = len(samples) - len(deduped)
    if removed > 0:
        print(f"  ⚠️  중복 파일 {removed}개 제거 (총 {len(deduped)}개 사용)")
    return deduped


def split_and_copy(samples, task_name, original_samples=None):
    """
    train/val/test 분리 후 WORK_DIR로 파일 복사.

    [데이터 누수 방지]
    1. _dedup_samples()로 파일 경로 중복 제거 후 분리.
    2. sound 태스크: test 파일을 original_samples에서 먼저 확정하고,
       해당 경로들을 오버샘플 pool(samples)에서 사전 제거.
       → train/val ↔ test 겹침 완전 차단.
    """
    # ── 중복 경로 제거 ──
    samples = _dedup_samples(samples)

    random.shuffle(samples)
    class_samples = defaultdict(list)

    for label, path in samples:
        class_samples[label].append(path)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(WORK_DIR, split, task_name), exist_ok=True)

    # ── sound: test 파일 먼저 확정 → train/val pool에서 제거 ──
    if original_samples is not None:
        orig_class = defaultdict(list)
        for label, path in _dedup_samples(original_samples):
            orig_class[label].append(path)

        test_items_by_label = {}
        excluded_paths = set()
        for label, paths in orig_class.items():
            n_test = max(10, len(paths) // 5)
            test_paths = paths[:n_test]
            test_items_by_label[label] = test_paths
            excluded_paths.update(test_paths)

        # 오버샘플 pool에서 test 파일 제거
        filtered_class_samples = defaultdict(list)
        for label, paths in class_samples.items():
            filtered_class_samples[label] = [p for p in paths if p not in excluded_paths]
        class_samples = filtered_class_samples
    else:
        test_items_by_label = None

    for label, items in class_samples.items():
        n = len(items)
        n_train = int(n * 0.8)
        n_val   = int(n * 0.1)

        if test_items_by_label is not None:
            train_items = items[:n_train]
            val_items   = items[n_train:n_train + n_val]
            test_items  = test_items_by_label.get(label, [])
        else:
            train_items = items[:n_train]
            val_items   = items[n_train:n_train + n_val]
            test_items  = items[n_train + n_val:]

        split_map = {"train": train_items, "val": val_items, "test": test_items}

        for split_name, split_items in split_map.items():
            dst_label_dir = os.path.join(WORK_DIR, split_name, task_name, label)
            os.makedirs(dst_label_dir, exist_ok=True)

            for item in tqdm(split_items, desc=f"{task_name}/{split_name}/{label}", leave=False):
                dst_path = os.path.join(dst_label_dir, f"{label}_{os.path.basename(item)}")
                shutil.copy(item, dst_path)


def _task_ready(task_name):
    """해당 task의 train 폴더가 존재하고 비어있지 않으면 True"""
    task_train = os.path.join(WORK_DIR, "train", task_name)
    return os.path.isdir(task_train) and len(os.listdir(task_train)) > 0


def prepare_dataset():
    need_behavior = not _task_ready("behavior")
    need_emotion  = not _task_ready("emotion")
    need_sound    = not _task_ready("sound")
    need_patella  = not _task_ready("patella")

    if not any([need_behavior, need_emotion, need_sound, need_patella]):
        print("✅ All file-copy tasks already prepared, skipping.")
        return

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(WORK_DIR, split), exist_ok=True)

    if need_behavior:
        print("\n📦 Collecting behavior (all samples)...")
        behavior_all = collect_samples(BEHAVIOR_ROOT, ['.jpg', '.png', '.jpeg'])
        behavior = sample_balanced(behavior_all)
        print("  📋 Splitting & Copying behavior...")
        split_and_copy(behavior, "behavior")
    else:
        print("✅ behavior already prepared, skipping.")

    if need_emotion:
        print("\n📦 Collecting emotion (all samples)...")
        emotion_all = collect_samples(EMOTION_ROOT, ['.jpg', '.png', '.jpeg'])
        emotion = sample_balanced(emotion_all)
        print("  📋 Splitting & Copying emotion...")
        split_and_copy(emotion, "emotion")
    else:
        print("✅ emotion already prepared, skipping.")

    if need_sound:
        print("\n📦 Collecting sound (all samples)...")
        sound_all = collect_samples(SOUND_ROOT, ['.wav', '.mp3', '.m4a'])
        sound = sample_balanced_audio(sound_all)
        print("  📋 Splitting & Copying sound...")
        split_and_copy(sound, "sound", original_samples=sound_all)
    else:
        print("✅ sound already prepared, skipping.")

    if need_patella:
        print("\n📦 Collecting patella luxation (date-folder split)...")
        grade_date_map = collect_patella_by_date(PATELLA_ROOT)
        print("  📋 Splitting & Copying patella (by date-folder)...")
        split_and_copy_patella(grade_date_map)
    else:
        print("✅ patella already prepared, skipping.")

    print("\n✅ Dataset preparation complete.")


# =========================
# 2. Dataset Classes
# =========================
class ImageDataset(Dataset):
    def __init__(self, task_dir, augment=False):
        self.samples = []
        self.label_to_id = {}

        for label in sorted(os.listdir(task_dir)):
            label_dir = os.path.join(task_dir, label)
            if not os.path.isdir(label_dir):
                continue

            self.label_to_id[label] = len(self.label_to_id)

            for file in os.listdir(label_dir):
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.samples.append((os.path.join(label_dir, file), label))

        print(f"  📊 {os.path.basename(task_dir)}: {len(self.samples)} samples, {len(self.label_to_id)} classes")

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, self.label_to_id[label]


class PatellaDataset(Dataset):
    def __init__(self, task_dir, augment=False):
        self.samples = []

        # [FIX v2] Patella grade는 순서가 있는 ordinal 데이터.
        # sorted()는 문자열 기준 정렬이라 "normal" < "1" < "2"... 이 아닌 임의 순서가 됨.
        # → 명시적 순서 지정: normal=0, grade1=1, grade2=2, grade3=3, grade4=4
        # 이 순서를 지키면 인접 grade 혼동 시 패널티가 작아 학습에 유리함.
        PATELLA_ORDER = ["normal", "1", "2", "3", "4"]
        available = [d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))]
        ordered = [g for g in PATELLA_ORDER if g in available]
        # PATELLA_ORDER에 없는 미정의 등급은 뒤에 추가 (안전장치)
        for g in sorted(available):
            if g not in ordered:
                ordered.append(g)
        self.label_to_id = {label: idx for idx, label in enumerate(ordered)}

        # ordered 순서대로 샘플 수집 (label_to_id 순서와 일치 보장)
        for label in ordered:
            label_dir = os.path.join(task_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for file in os.listdir(label_dir):
                if file.lower().endswith('.jpg'):
                    img_path  = os.path.join(label_dir, file)
                    json_path = os.path.splitext(img_path)[0] + '.json'
                    if os.path.exists(json_path):
                        self.samples.append((img_path, json_path, label))

        print(f"  📊 {os.path.basename(task_dir)}: {len(self.samples)} samples, {len(self.label_to_id)} classes")

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size  # 정규화를 위해 원본 크기 저장
        img = self.transform(img)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # [FIX v2] KP_DIM 27 → 39 으로 수정.
        # 진단 결과 실제 데이터의 keypoint 수 분포:
        #   5~13개 혼재, 대부분 12개(99/200건), 최대 13개(1건)
        # 기존 KP_DIM=27(9개 기준)은 12개 keypoint 샘플의 뒤 9차원(3 keypoints)을
        # 모두 잘라버려 모델이 불완전한 정보로 학습하는 버그가 있었음.
        # → 최대값 13개 기준 13×3=39로 확장하여 모든 keypoint를 온전히 사용.
        # - visibility=1.0: 실제 keypoint, visibility=0.0: 패딩(원점과 구별 가능)
        # ※ annotation_info 내 키 이름(x, y)이 실제 JSON 스키마와 다를 경우
        #   아래 .get('x', ...) 부분을 실제 키 이름으로 수정하세요.
        KP_DIM = 39  # 13 keypoints × 3 (x, y, vis) — 실제 데이터 최대값 기준
        keypoints = []
        for annotation in data.get('annotation_info', []):
            x = float(annotation.get('x', 0)) / orig_w  # [0, 1] 정규화
            y = float(annotation.get('y', 0)) / orig_h  # [0, 1] 정규화
            keypoints.extend([x, y, 1.0])               # visibility = 1 (실제 keypoint)

        # 부족한 자리는 (0, 0, 0) 으로 패딩 → vis=0 이므로 모델이 무시 가능
        while len(keypoints) < KP_DIM:
            keypoints.append(0.0)

        keypoints = torch.tensor(keypoints[:KP_DIM], dtype=torch.float32)

        return img, keypoints, self.label_to_id[label]


class AudioDataset(Dataset):
    def __init__(self, task_dir, augment=False):
        self.samples = []
        self.label_to_id = {}
        self.id_to_label = {}
        self.augment = augment
        next_id = 0

        for label in sorted(os.listdir(task_dir)):
            label_dir = os.path.join(task_dir, label)
            if not os.path.isdir(label_dir):
                continue

            self.label_to_id[label] = next_id
            self.id_to_label[next_id] = label
            next_id += 1

            for file in os.listdir(label_dir):
                if file.lower().endswith(('.wav', '.mp3', '.m4a')):
                    self.samples.append((os.path.join(label_dir, file), label))

        print(f"  📊 {os.path.basename(task_dir)}: {len(self.samples)} samples, {len(self.label_to_id)} classes, augment={augment}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            waveform, _ = librosa.load(path, sr=SR, mono=True)
        except Exception:
            waveform = np.zeros(MAX_AUDIO_LEN)

        if self.augment:
            waveform = augment_audio(waveform)

        if len(waveform) > MAX_AUDIO_LEN:
            waveform = waveform[:MAX_AUDIO_LEN]
        else:
            waveform = np.pad(waveform, (0, MAX_AUDIO_LEN - len(waveform)))

        inputs = FEATURE_EXTRACTOR(waveform, sampling_rate=SR, return_tensors="pt")
        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(self.label_to_id[label], dtype=torch.long)
        }


def collate_fn_audio(batch):
    input_values = torch.stack([item["input_values"] for item in batch])
    labels       = torch.stack([item["labels"]       for item in batch])
    return {"input_values": input_values, "labels": labels}


# =========================
# 3. Individual Models
# =========================
def _resnet50_backbone():
    """ResNet50 backbone. fc를 Identity로 교체하고 feat_dim(2048) 반환."""
    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_features = backbone.fc.in_features  # 2048
    backbone.fc = nn.Identity()
    return backbone, in_features


class BehaviorModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone, in_features = _resnet50_backbone()
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.head(self.backbone(x))


class EmotionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone, in_features = _resnet50_backbone()
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.head(self.backbone(x))


class PatellaModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone, in_features = _resnet50_backbone()

        # [FIX v2] keypoint 임베딩 입력 차원: 27 → 39 (x, y, visibility) × 13 keypoints
        # 진단 결과 실제 데이터 최대 13개 keypoint → KP_DIM=39에 맞춰 수정.
        # backbone feature(2048)에 묻히지 않도록 별도 임베딩 후 fusion.
        # ⚠️  PatellaDataset 의 KP_DIM = 39 와 반드시 일치해야 함.
        KP_INPUT_DIM = 39
        self.kp_embed = nn.Sequential(
            nn.Linear(KP_INPUT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(in_features + 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, keypoints):
        feat    = self.backbone(x)
        kp_feat = self.kp_embed(keypoints)
        return self.head(torch.cat([feat, kp_feat], dim=1))


class AudioModel(nn.Module):
    def __init__(self, num_classes, freeze_backbone=False):
        super().__init__()
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            AUDIO_MODEL_NAME,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        if freeze_backbone:
            for param in self.model.wav2vec2.parameters():
                param.requires_grad = False

    def forward(self, input_values, labels=None):
        return self.model(input_values=input_values, labels=labels)


# =========================
# 4. Helper Functions
# =========================
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()


def make_loader(dataset, shuffle, is_audio=False):
    """
    공통 DataLoader 생성 헬퍼.
    - persistent_workers=True: epoch마다 worker 재생성 오버헤드 제거
    - prefetch_factor=4: GPU 대기 시간 감소
    """
    workers = 2 if is_audio else NUM_WORKERS
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
        prefetch_factor=4 if workers > 0 else None,
        collate_fn=collate_fn_audio if is_audio else None,
    )


# =========================
# 5. Sequential Training (메모리 효율적)
# =========================
def train():
    prepare_dataset()

    # label_to_id 미리 로드
    print("\n🔄 Pre-loading label mappings...")
    temp_b = ImageDataset(os.path.join(WORK_DIR, "train", "behavior"), augment=False)
    temp_e = ImageDataset(os.path.join(WORK_DIR, "train", "emotion"),  augment=False)
    temp_s = AudioDataset(os.path.join(WORK_DIR, "train", "sound"),    augment=False)
    temp_p = PatellaDataset(os.path.join(WORK_DIR, "train", "patella"), augment=False)

    behavior_label_to_id = temp_b.label_to_id
    emotion_label_to_id  = temp_e.label_to_id
    sound_label_to_id    = temp_s.label_to_id
    sound_id_to_label    = temp_s.id_to_label
    patella_label_to_id  = temp_p.label_to_id

    # Behavior class_weight  [FIX] 기존 누락 → 추가 (25 클래스 불균형 보정)
    behavior_labels_list = [temp_b.label_to_id[label] for _, label in temp_b.samples]
    behavior_class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(len(behavior_label_to_id)),
        y=behavior_labels_list
    )
    behavior_class_weights_tensor = torch.tensor(behavior_class_weights, dtype=torch.float)

    # Emotion class_weight
    emotion_labels_list = [temp_e.label_to_id[label] for _, label in temp_e.samples]
    emotion_class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(len(emotion_label_to_id)),
        y=emotion_labels_list
    )
    emotion_class_weights_tensor = torch.tensor(emotion_class_weights, dtype=torch.float)

    # Patella class_weight
    patella_labels_list = [temp_p.label_to_id[label] for _, _, label in temp_p.samples]
    patella_class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(len(patella_label_to_id)),
        y=patella_labels_list
    )
    patella_class_weights = np.clip(patella_class_weights, 0.5, 2.5)  # 극단값 제한
    patella_class_weights_tensor = torch.tensor(patella_class_weights, dtype=torch.float)
    print(f"  📊 Patella class weights (clamped): {dict(zip(sorted(patella_label_to_id.keys()), patella_class_weights.round(3)))}")

    del temp_b, temp_e, temp_s, temp_p
    clear_memory()

    # 모델 초기화 (CPU에 먼저 생성)
    print("\n🔄 Initializing models...")
    behavior_model = BehaviorModel(len(behavior_label_to_id))
    emotion_model  = EmotionModel(len(emotion_label_to_id))
    patella_model  = PatellaModel(len(patella_label_to_id))
    audio_model    = AudioModel(len(sound_label_to_id), freeze_backbone=False)

    # Optimizers
    behavior_opt = torch.optim.AdamW(behavior_model.parameters(), lr=LR_VIDEO, weight_decay=0.01)
    emotion_opt  = torch.optim.AdamW(emotion_model.parameters(),  lr=LR_VIDEO, weight_decay=0.01)
    patella_opt  = torch.optim.AdamW(patella_model.parameters(),  lr=1e-4,     weight_decay=0.01)
    audio_opt    = torch.optim.AdamW(audio_model.parameters(),    lr=LR_AUDIO, weight_decay=0.01)

    # Patella LR Scheduler (CosineAnnealing)
    patella_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        patella_opt, T_max=EPOCHS, eta_min=1e-6
    )

    # Audio LR Warmup Scheduler
    _temp_sound = AudioDataset(os.path.join(WORK_DIR, "train", "sound"), augment=False)
    _approx_sound_steps = (len(_temp_sound) // BATCH_SIZE) * EPOCHS
    del _temp_sound
    # [FIX] warmup_steps: 100 → 20
    # sound 데이터는 ~12 배치/epoch 이므로 warmup=100이면 약 8 epoch 동안 LR이
    # 거의 0에 머물러 초기 epoch 정확도가 고착되는 문제가 있었음.
    # 20 steps(≈ 1.5 epoch)으로 줄여 빠른 초기 학습을 유도.
    audio_scheduler = get_linear_schedule_with_warmup(
        audio_opt,
        num_warmup_steps=20,
        num_training_steps=_approx_sound_steps
    )
    clear_memory()

    # Scalers - 모든 task 독립 (한 task의 inf/nan이 다른 task에 전파되지 않도록)
    # [FIX] behavior/emotion 공유 video_scaler → 각각 독립 scaler 분리
    behavior_scaler = torch.amp.GradScaler("cuda")
    emotion_scaler  = torch.amp.GradScaler("cuda")
    patella_scaler  = torch.amp.GradScaler("cuda")
    audio_scaler    = torch.amp.GradScaler("cuda")

    # Loss
    # [FIX] behavior도 class_weight 적용 (기존: 가중치 없음 → 25 클래스 불균형 무보정)
    criterion = nn.CrossEntropyLoss(
        weight=behavior_class_weights_tensor.to(DEVICE),
        label_smoothing=0.1
    )
    criterion_emotion = nn.CrossEntropyLoss(
        weight=emotion_class_weights_tensor.to(DEVICE),
        label_smoothing=0.1
    )
    criterion_patella = nn.CrossEntropyLoss(
        weight=patella_class_weights_tensor.to(DEVICE),
        label_smoothing=0.1
    )

    best_avg_acc = 0
    history = []

    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")

        loss_b, loss_e, loss_s, loss_p = 0, 0, 0, 0

        # ========== 1. Behavior ==========
        print(f"\n🐾 Training Behavior...")
        behavior_model.to(DEVICE)
        behavior_model.train()

        behavior_train  = ImageDataset(os.path.join(WORK_DIR, "train", "behavior"), augment=True)
        behavior_loader = make_loader(behavior_train, shuffle=True)

        for imgs, labels in tqdm(behavior_loader, desc="Behavior", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            behavior_opt.zero_grad()
            with torch.amp.autocast("cuda"):
                imgs, labels_a, labels_b, lam, _ = mixup_data(imgs, labels)
                logits = behavior_model(imgs)
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)

            behavior_scaler.scale(loss).backward()
            behavior_scaler.unscale_(behavior_opt)  # [FIX] 누락된 unscale_ 추가
            torch.nn.utils.clip_grad_norm_(behavior_model.parameters(), 1.0)  # [FIX] 누락된 grad clip 추가
            behavior_scaler.step(behavior_opt)
            behavior_scaler.update()

            loss_b += loss.item()

        loss_b /= len(behavior_loader)
        print(f"  → Avg Loss: {loss_b:.4f}")

        behavior_model.cpu()
        del behavior_train, behavior_loader
        clear_memory()

        # ========== 2. Emotion ==========
        print(f"\n😊 Training Emotion...")
        emotion_model.to(DEVICE)
        emotion_model.train()

        emotion_train  = ImageDataset(os.path.join(WORK_DIR, "train", "emotion"), augment=True)
        emotion_loader = make_loader(emotion_train, shuffle=True)

        for imgs, labels in tqdm(emotion_loader, desc="Emotion", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            emotion_opt.zero_grad()
            with torch.amp.autocast("cuda"):
                imgs, labels_a, labels_b, lam, _ = mixup_data(imgs, labels)
                logits = emotion_model(imgs)
                loss = (lam * criterion_emotion(logits, labels_a)
                        + (1 - lam) * criterion_emotion(logits, labels_b))
                loss = loss * LOSS_WEIGHTS["emotion"]

            emotion_scaler.scale(loss).backward()
            emotion_scaler.unscale_(emotion_opt)  # [FIX] 누락된 unscale_ 추가
            torch.nn.utils.clip_grad_norm_(emotion_model.parameters(), 1.0)  # [FIX] 누락된 grad clip 추가
            emotion_scaler.step(emotion_opt)
            emotion_scaler.update()

            loss_e += loss.item()

        loss_e /= len(emotion_loader)
        print(f"  → Avg Loss: {loss_e:.4f}")

        emotion_model.cpu()
        del emotion_train, emotion_loader
        clear_memory()

        # ========== 3. Sound ==========
        print(f"\n🔊 Training Sound...")
        audio_model.to(DEVICE)
        audio_model.train()

        sound_train = AudioDataset(os.path.join(WORK_DIR, "train", "sound"), augment=True)

        sound_labels_list = [item[1] for item in sound_train.samples]
        sound_label_ids   = [sound_train.label_to_id[l] for l in sound_labels_list]
        class_weights = compute_class_weight(
            'balanced',
            classes=np.arange(len(sound_train.label_to_id)),
            y=sound_label_ids
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

        sound_loader = make_loader(sound_train, shuffle=True, is_audio=True)

        # [FIX] class_weight 적용 방식 수정.
        # 기존: outputs.loss(이미 평균된 scalar)에 per_sample_w.mean()을 곱하는 방식
        #       → loss scale만 흔들릴 뿐 클래스별 가중치가 정확히 반영되지 않음.
        # 수정: CrossEntropyLoss(weight=...)를 직접 생성해 logits에 적용.
        criterion_sound = nn.CrossEntropyLoss(weight=class_weights_tensor)

        for batch in tqdm(sound_loader, desc="Sound", leave=False):
            audios = batch["input_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            audio_opt.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs = audio_model(input_values=audios)          # labels 제거 → 내부 loss 미사용
                loss = criterion_sound(outputs.logits, labels) * LOSS_WEIGHTS["sound"]

            audio_scaler.scale(loss).backward()
            audio_scaler.unscale_(audio_opt)
            torch.nn.utils.clip_grad_norm_(audio_model.parameters(), 1.0)
            audio_scaler.step(audio_opt)
            audio_scaler.update()
            audio_scheduler.step()  # optimizer.step() 이후 호출 (순서 경고 수정)

            loss_s += loss.item()

        loss_s /= len(sound_loader)
        print(f"  → Avg Loss: {loss_s:.4f}")

        audio_model.cpu()
        del sound_train, sound_loader, class_weights_tensor
        clear_memory()

        # ========== 4. Patella ==========
        print(f"\n🦴 Training Patella...")
        patella_model.to(DEVICE)
        patella_model.train()

        patella_train  = PatellaDataset(os.path.join(WORK_DIR, "train", "patella"), augment=True)
        patella_loader = make_loader(patella_train, shuffle=True)

        for imgs, keypoints, labels in tqdm(patella_loader, desc="Patella", leave=False):
            imgs, keypoints, labels = imgs.to(DEVICE), keypoints.to(DEVICE), labels.to(DEVICE)

            patella_opt.zero_grad()
            with torch.amp.autocast("cuda"):
                imgs, labels_a, labels_b, lam, index = mixup_data(imgs, labels)
                mixed_kp = lam * keypoints + (1 - lam) * keypoints[index]  # keypoints도 동일 index로 mix
                logits = patella_model(imgs, mixed_kp)
                loss = lam * criterion_patella(logits, labels_a) + (1 - lam) * criterion_patella(logits, labels_b)

            patella_scaler.scale(loss).backward()
            patella_scaler.unscale_(patella_opt)
            torch.nn.utils.clip_grad_norm_(patella_model.parameters(), 1.0)
            patella_scaler.step(patella_opt)
            patella_scaler.update()

            loss_p += loss.item()

        loss_p /= len(patella_loader)
        patella_scheduler.step()
        print(f"  → Avg Loss: {loss_p:.4f} | LR: {patella_scheduler.get_last_lr()[0]:.2e}")

        patella_model.cpu()
        del patella_train, patella_loader
        clear_memory()

        # ========== Validation ==========
        print(f"\n🔍 Validation...")

        # Behavior Val
        behavior_model.to(DEVICE)
        behavior_model.eval()
        behavior_val        = ImageDataset(os.path.join(WORK_DIR, "val", "behavior"), augment=False)
        behavior_val_loader = make_loader(behavior_val, shuffle=False)

        correct_b, total_b = 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(behavior_val_loader, desc="Val Behavior", leave=False):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = behavior_model(imgs)
                pred = logits.argmax(-1)
                correct_b += (pred == labels).sum().item()
                total_b   += labels.size(0)
        acc_b = correct_b / total_b

        behavior_model.cpu()
        del behavior_val, behavior_val_loader
        clear_memory()

        # Emotion Val
        emotion_model.to(DEVICE)
        emotion_model.eval()
        emotion_val        = ImageDataset(os.path.join(WORK_DIR, "val", "emotion"), augment=False)
        emotion_val_loader = make_loader(emotion_val, shuffle=False)

        correct_e, total_e = 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(emotion_val_loader, desc="Val Emotion", leave=False):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = emotion_model(imgs)
                pred = logits.argmax(-1)
                correct_e += (pred == labels).sum().item()
                total_e   += labels.size(0)
        acc_e = correct_e / total_e

        emotion_model.cpu()
        del emotion_val, emotion_val_loader
        clear_memory()

        # Sound Val
        audio_model.to(DEVICE)
        audio_model.eval()
        sound_val        = AudioDataset(os.path.join(WORK_DIR, "val", "sound"), augment=False)
        sound_val_loader = make_loader(sound_val, shuffle=False, is_audio=True)

        correct_s, total_s = 0, 0
        with torch.no_grad():
            for batch in tqdm(sound_val_loader, desc="Val Sound", leave=False):
                audios = batch["input_values"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                outputs = audio_model(input_values=audios, labels=labels)
                pred = outputs.logits.argmax(-1)
                correct_s += (pred == labels).sum().item()
                total_s   += labels.size(0)
        acc_s = correct_s / total_s

        audio_model.cpu()
        del sound_val, sound_val_loader
        clear_memory()

        # Patella Val
        patella_model.to(DEVICE)
        patella_model.eval()
        patella_val        = PatellaDataset(os.path.join(WORK_DIR, "val", "patella"), augment=False)
        patella_val_loader = make_loader(patella_val, shuffle=False)

        correct_p, total_p = 0, 0
        with torch.no_grad():
            for imgs, keypoints, labels in tqdm(patella_val_loader, desc="Val Patella", leave=False):
                imgs, keypoints, labels = imgs.to(DEVICE), keypoints.to(DEVICE), labels.to(DEVICE)
                logits = patella_model(imgs, keypoints)
                pred = logits.argmax(-1)
                correct_p += (pred == labels).sum().item()
                total_p   += labels.size(0)
        acc_p = correct_p / total_p

        patella_model.cpu()
        del patella_val, patella_val_loader
        clear_memory()

        avg_acc = (acc_b + acc_e + acc_s + acc_p) / 4

        print(f"\n📊 Results:")
        print(f"  Behavior: Loss {loss_b:.4f} | Acc {acc_b:.4f} ({acc_b*100:.1f}%)")
        print(f"  Emotion:  Loss {loss_e:.4f} | Acc {acc_e:.4f} ({acc_e*100:.1f}%)")
        print(f"  Sound:    Loss {loss_s:.4f} | Acc {acc_s:.4f} ({acc_s*100:.1f}%)")
        print(f"  Patella:  Loss {loss_p:.4f} | Acc {acc_p:.4f} ({acc_p*100:.1f}%)")
        print(f"  Average Acc: {avg_acc:.4f} ({avg_acc*100:.1f}%)")

        history.append({
            'epoch' : epoch + 1,
            'loss_b': loss_b, 'loss_e': loss_e,
            'loss_s': loss_s, 'loss_p': loss_p,
            'acc_b' : acc_b,  'acc_e' : acc_e,
            'acc_s' : acc_s,  'acc_p' : acc_p,
            'acc_avg': avg_acc,
        })

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            torch.save({
                "behavior_model":           behavior_model.state_dict(),
                "emotion_model":            emotion_model.state_dict(),
                "audio_model":              audio_model.state_dict(),
                "patella_model":            patella_model.state_dict(),
                "behavior_label_to_id":     behavior_label_to_id,
                "emotion_label_to_id":      emotion_label_to_id,
                "sound_label_to_id":        sound_label_to_id,
                "sound_id_to_label":        sound_id_to_label,
                "patella_label_to_id":      patella_label_to_id,
                "patella_scheduler":        patella_scheduler.state_dict(),
                "best_epoch":               epoch + 1,
                "best_acc":                 best_avg_acc,
                "history":                  history,
            }, "pet_normal_omni_best.pth")
            print(f"  💾 Saved new best model! (Acc: {best_avg_acc:.4f})")

    # 학습 곡선 시각화
    print("\n📈 Generating training history plot...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    tasks = [
        ('acc_b', 'loss_b', 'b-',     'Behavior'),
        ('acc_e', 'loss_e', 'r-',     'Emotion'),
        ('acc_s', 'loss_s', 'g-',     'Sound'),
        ('acc_p', 'loss_p', 'purple', 'Patella'),
    ]

    for i, (acc_key, loss_key, color, title) in enumerate(tasks):
        axes[0, i].plot([h[loss_key] for h in history], color=color, linewidth=2)
        axes[0, i].set_title(f'{title} Loss')
        axes[0, i].set_xlabel('Epoch')
        axes[0, i].set_ylabel('Loss')
        axes[0, i].grid(True, alpha=0.3)

        axes[1, i].plot([h[acc_key] for h in history], color=color, linewidth=2)
        axes[1, i].set_title(f'{title} Accuracy')
        axes[1, i].set_xlabel('Epoch')
        axes[1, i].set_ylabel('Accuracy')
        axes[1, i].set_ylim(0, 1)
        axes[1, i].grid(True, alpha=0.3)

    plt.suptitle('Pet Normal Omni Model Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pet_normal_omni_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: pet_normal_omni_history.png")

    print(f"\n🎉 Training Finished!")
    print(f"  Best Average Acc: {best_avg_acc:.4f} ({best_avg_acc*100:.1f}%)")


if __name__ == "__main__":
    train()
