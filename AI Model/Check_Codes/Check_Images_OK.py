# PIL 상 잘못된 이미지 확인

import os
from PIL import Image, ImageFile
import shutil
from pathlib import Path

def find_corrupt_images(root_dir, output_dir='corrupt_images', dry_run=False):
    """
    손상된 이미지 파일을 찾아 이동/보고
    
    Args:
        root_dir: 검사할 루트 디렉토리
        output_dir: corrupt 파일 이동할 폴더 (기본: corrupt_images/)
        dry_run: True면 이동 안하고 목록만 출력
    
    Returns:
        list: (파일경로, 에러메시지) 목록
    """
    ImageFile.LOAD_TRUNCATED_IMAGES = False  # 엄격 검사
    
    os.makedirs(output_dir, exist_ok=True)
    corrupt_files = []
    total_files = 0
    
    print(f"🔍 검사 시작: {root_dir}")
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                total_files += 1
                img_path = os.path.join(root, file)
                
                try:
                    # 1단계: 열기
                    with Image.open(img_path) as img:
                        # 2단계: 무결성 검증
                        img.verify()
                        # 3단계: 실제 데이터 로드 (verify 후 재로드 필요)
                        img = Image.open(img_path)
                        img.load()
                        img.getexif()  # EXIF 오류도 잡음
                except Exception as e:
                    corrupt_files.append((img_path, str(e)[:100]))
                    
                    if not dry_run:
                        # 상대경로 유지하며 corrupt 폴더로 이동
                        rel_path = Path(img_path).relative_to(root_dir)
                        dst_path = Path(output_dir) / rel_path
                        dst_dir = dst_path.parent
                        os.makedirs(dst_dir, exist_ok=True)
                        
                        shutil.move(img_path, dst_path)
                        print(f"🗑️  [{len(corrupt_files)}] {rel_path}")
                    else:
                        print(f"⚠️  [{len(corrupt_files)}] {img_path}: {str(e)[:80]}")
    
    print(f"\n✅ 검사 완료!")
    print(f"   총 파일: {total_files:,}")
    print(f"   손상 파일: {len(corrupt_files):,}개 ({len(corrupt_files)/total_files*100:.2f}%)")
    
    if corrupt_files:
        print("\n📋 손상 파일 상위 5개:")
        for path, error in corrupt_files[:5]:
            print(f"  - {Path(path).name}: {error}")
    
    return corrupt_files

# 실행 예시
if __name__ == "__main__":
    WORK_DIR = "files/6_Animal_Patella"
    
    # 1단계: dry_run으로 먼저 확인
    print("=== DRY RUN (이동 안함) ===")
    corrupts = find_corrupt_images(WORK_DIR, dry_run=True)
    
    # 2단계: 문제없으면 실제 이동
    if input("\n실제로 corrupt 파일 이동? (y/N): ").lower() == 'y':
        find_corrupt_images(WORK_DIR)