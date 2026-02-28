"""
check_pet_sound_split.py
────────────────────────────────────────────────────────────────────
files/6_Pet_Sound 하위 클래스별 샘플 수를 확인하고,
cat_normal_omni_train.py의 collect_and_split 로직 기준으로
train / val / test 분할 결과를 미리 출력합니다.
────────────────────────────────────────────────────────────────────
"""

import os

PET_SOUND_ROOT = "files/3_Animal_Sound"    # 실제 경로에 맞게 수정
OVERSAMPLE_MIN = 150                     # cat_normal_omni_train.py 기준값
AUDIO_EXTS     = ('.wav', '.mp3', '.m4a')

# ── 1. 클래스별 파일 수 집계 ────────────────────────────────────────
print(f"\n📂 스캔 경로: {PET_SOUND_ROOT}\n{'='*55}")

class_counts = {}
if not os.path.isdir(PET_SOUND_ROOT):
    print(f"❌ 경로를 찾을 수 없습니다: {PET_SOUND_ROOT}")
    exit(1)

for cls in sorted(os.listdir(PET_SOUND_ROOT)):
    cls_dir = os.path.join(PET_SOUND_ROOT, cls)
    if not os.path.isdir(cls_dir):
        continue
    count = sum(
        1 for root, _, files in os.walk(cls_dir)
        for f in files if f.lower().endswith(AUDIO_EXTS)
    )
    class_counts[cls] = count

# ── 2. 클래스별 split 계산 ──────────────────────────────────────────
print(f"{'클래스':<25} {'원본':>6} {'train':>8} {'val':>6} {'test':>6}  {'오버샘플 후 train':>18}")
print("-" * 75)

total_raw   = {"train": 0, "val": 0, "test": 0}
total_final = {"train": 0, "val": 0, "test": 0}

for cls, n in sorted(class_counts.items()):
    n_val   = max(1, int(n * 0.1))
    n_test  = max(1, int(n * 0.1))
    n_train = n - n_val - n_test
    n_train_os = max(n_train, OVERSAMPLE_MIN) if OVERSAMPLE_MIN > 0 else n_train

    total_raw["train"]   += n_train
    total_raw["val"]     += n_val
    total_raw["test"]    += n_test
    total_final["train"] += n_train_os
    total_final["val"]   += n_val
    total_final["test"]  += n_test

    os_note = f"{n_train_os:>6}  ← 오버샘플" if n_train_os > n_train else f"{n_train_os:>6}"
    print(f"  {cls:<23} {n:>6} {n_train:>8} {n_val:>6} {n_test:>6}  {os_note}")

# ── 3. 합계 ────────────────────────────────────────────────────────
print("=" * 75)
print(f"  {'[원본 합계]':<23} {sum(class_counts.values()):>6} "
      f"{total_raw['train']:>8} {total_raw['val']:>6} {total_raw['test']:>6}")
print(f"  {'[오버샘플 후 합계]':<23} {'':>6} "
      f"{total_final['train']:>8} {total_final['val']:>6} {total_final['test']:>6}")
print()

# ── 4. BatchNorm 경고 체크 ────────────────────────────────────────
BATCH_SIZE = 32
print("⚠️  BatchNorm 위험 체크 (val 샘플 수 < BATCH_SIZE)")
for cls, n in sorted(class_counts.items()):
    n_val = max(1, int(n * 0.1))
    if n_val < BATCH_SIZE:
        print(f"   {cls}: val={n_val}개  →  배치 1개 미만, drop_last=False 이므로 OK")
print("   (val은 drop_last=False 이므로 에러 없음, 정보 참고용)\n")
