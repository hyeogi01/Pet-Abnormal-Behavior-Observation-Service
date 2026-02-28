"""
reorganize_dog_sound.py
────────────────────────────────────────────────────────────────────
files/3_Animal_Sound 클래스 재구성

[변경 내용]
  병합: dog_bark     ← dog_bark + dog_growl
  병합: dog_respiratory_event ← dog_cough + dog_tracheal_collapse
                                 + dog_breath + dog_sneeze
  제거: dog_playing
  유지: cat_aggressive, cat_positive, dog_howling, dog_whining

[동작 방식]
  - 원본 폴더는 절대 삭제/수정하지 않음 (--dry-run으로 미리 확인 가능)
  - 이미 존재하는 파일은 건너뜀 (중복 실행 안전)
  - 작업 완료 후 최종 구조 요약 출력
────────────────────────────────────────────────────────────────────
"""

import os, shutil, argparse
from collections import defaultdict

SOUND_ROOT = "files/3_Animal_Sound"

# ── 병합/유지/제거 규칙 정의 ──────────────────────────────────────
MERGE_RULES = {
    # 결과 클래스명          : [원본 클래스명, ...]
    "dog_bark"               : ["dog_bark", "dog_growl"],
    "dog_respiratory_event"  : ["dog_cough", "dog_tracheal_collapse",
                                 "dog_breath", "dog_sneeze"],
    "dog_howling"            : ["dog_howling"],
    "dog_whining"            : ["dog_whining"],
    "cat_aggressive"         : ["cat_aggressive"],
    "cat_positive"           : ["cat_positive"],
}
REMOVE_CLASSES = ["dog_playing"]

def count_files(directory):
    exts = ('.wav', '.mp3', '.m4a')
    return sum(
        1 for _, _, files in os.walk(directory)
        for f in files if f.lower().endswith(exts)
    )

def reorganize(dry_run=False):
    mode = "🔍 [DRY-RUN]" if dry_run else "🔨 [EXECUTE]"
    print(f"\n{mode} 시작 | 경로: {SOUND_ROOT}\n{'='*60}")

    if not os.path.isdir(SOUND_ROOT):
        print(f"❌ 경로를 찾을 수 없습니다: {SOUND_ROOT}")
        return

    moved_counts = defaultdict(int)
    skipped_counts = defaultdict(int)

    # ── 1. 병합 작업 ───────────────────────────────────────────────
    for dst_cls, src_classes in MERGE_RULES.items():
        dst_dir = os.path.join(SOUND_ROOT, dst_cls)
        if not dry_run:
            os.makedirs(dst_dir, exist_ok=True)

        for src_cls in src_classes:
            src_dir = os.path.join(SOUND_ROOT, src_cls)
            if not os.path.isdir(src_dir):
                print(f"  ⚠️  원본 폴더 없음 (건너뜀): {src_cls}")
                continue

            # 이미 같은 폴더면 파일 이동 불필요
            if src_cls == dst_cls:
                n = count_files(src_dir)
                print(f"  ✅ 유지: {src_cls:<30} ({n}개)")
                continue

            files = [
                os.path.join(root, f)
                for root, _, fs in os.walk(src_dir)
                for f in fs
                if f.lower().endswith(('.wav', '.mp3', '.m4a'))
            ]

            for src_path in files:
                fname    = os.path.basename(src_path)
                # 충돌 방지: 원본클래스명을 prefix로 붙임
                dst_name = f"{src_cls}__{fname}"
                dst_path = os.path.join(dst_dir, dst_name)

                if os.path.exists(dst_path):
                    skipped_counts[dst_cls] += 1
                    continue

                if not dry_run:
                    shutil.copy2(src_path, dst_path)
                moved_counts[dst_cls] += 1

            print(f"  {'(예정)' if dry_run else '완료'} "
                  f"{src_cls:<28} → {dst_cls}  "
                  f"({len(files)}개 복사)")

    # ── 2. 제거 대상 안내 (실제 삭제는 하지 않음) ─────────────────
    print(f"\n{'─'*60}")
    print("🗑️  제거 대상 클래스 (수동 삭제 권장):")
    for cls in REMOVE_CLASSES:
        cls_dir = os.path.join(SOUND_ROOT, cls)
        if os.path.isdir(cls_dir):
            n = count_files(cls_dir)
            print(f"   {cls_dir}  ({n}개)  ← 직접 삭제하세요")
        else:
            print(f"   {cls}: 이미 없음")

    # ── 3. 원본 병합 소스 폴더 안내 ───────────────────────────────
    merged_sources = [
        src for src_list in MERGE_RULES.values()
        for src in src_list
        if src not in MERGE_RULES  # 결과 클래스와 이름이 다른 것만
    ]
    merged_sources = [s for s in merged_sources
                      if s not in list(MERGE_RULES.keys())]
    unique_sources = sorted(set(merged_sources))

    print(f"\n📦 병합 완료 후 원본 폴더 (수동 삭제 권장):")
    for src_cls in unique_sources:
        src_dir = os.path.join(SOUND_ROOT, src_cls)
        if os.path.isdir(src_dir):
            print(f"   {src_dir}")

    # ── 4. 최종 구조 요약 ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("📊 최종 클래스 구조 요약:")
    print(f"  {'클래스':<30} {'파일 수':>8}")
    print(f"  {'-'*40}")
    total = 0
    for dst_cls in sorted(MERGE_RULES.keys()):
        dst_dir = os.path.join(SOUND_ROOT, dst_cls)
        n = count_files(dst_dir) if os.path.isdir(dst_dir) else 0
        note = f"  ← +{moved_counts[dst_cls]}개 추가됨" if moved_counts[dst_cls] else ""
        print(f"  {dst_cls:<30} {n:>8}{note}")
        total += n
    print(f"  {'합계':<30} {total:>8}")
    print()

    if dry_run:
        print("💡 실제 실행하려면: python reorganize_dog_sound.py")
    else:
        print("✅ 완료! 위 원본 폴더들을 확인 후 수동으로 삭제하세요.")
        print("   이후 cat_normal_omni_train.py / dog_normal_omni_train.py의")
        print("   DOG_SOUND_CLASSES를 아래로 업데이트하세요:")
        print()
        print('   DOG_SOUND_CLASSES = [')
        for cls in sorted(k for k in MERGE_RULES if k.startswith("dog")):
            print(f'       "{cls}",')
        print('   ]')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 변경 없이 예상 결과만 출력")
    args = parser.parse_args()
    reorganize(dry_run=args.dry_run)
