# PIL 상 잘못된 파일 복구 코드

import cv2
from pathlib import Path

def repair_all_images(root_dir):
    """모든 JPEG를 cv2로 재인코딩"""
    total = 0
    repaired = 0
    
    for class_dir in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_dir)
        if not os.path.isdir(class_path): continue
            
        for filename in os.listdir(class_path):
            if filename.lower().endswith('.jpg'):
                img_path = os.path.join(class_path, filename)
                
                # cv2로 읽기 성공?
                img_bgr = cv2.imread(img_path)
                if img_bgr is None: continue
                
                # 재인코딩 저장
                cv2.imwrite(img_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                repaired += 1
                total += 1
                
                if total % 1000 == 0:
                    print(f"복구 중... {total}개 처리")
    
    print(f"✅ {repaired}/{total}개 파일 복구 완료!")

# 실행
repair_all_images("files/4_Animal_Skin")
repair_all_images("files/5_Animal_Eyes")