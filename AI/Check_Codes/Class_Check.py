# 현재 폴더의 class 구성 확인

import os
from collections import defaultdict

root = "files/5_Animal_Eyes"  # 현재 폴더
class_counts = defaultdict(int)

for class_dir in os.listdir(root):
    class_path = os.path.join(root, class_dir)
    if os.path.isdir(class_path):
        img_count = sum(1 for f in os.listdir(class_path) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        class_counts[class_dir] = img_count
        print(f"{class_dir}: {img_count:,} images")

total = sum(class_counts.values())
print(f"\n📊 Total: {total:,} images across {len(class_counts)} classes")
