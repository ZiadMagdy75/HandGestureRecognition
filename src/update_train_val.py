import os
import random

filtered_dir = "data/sign_data/filtered"
train_file = "data/train.txt"
val_file = "data/val.txt"

# نحصل على كل الصور المتبقية
all_images = [f for f in os.listdir(filtered_dir) if f.lower().endswith(".jpg")]
random.shuffle(all_images)

# نقسمهم 80% تدريب - 20% اختبار
split_idx = int(0.8 * len(all_images))
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

# نكتب المسارات الجديدة
with open(train_file, "w") as f:
    for img in train_images:
        f.write(f"sign_data/filtered/{img}\n")

with open(val_file, "w") as f:
    for img in val_images:
        f.write(f"sign_data/filtered/{img}\n")

print(f"✅ تم تحديث train.txt و val.txt بناءً على الصور في filtered/")
print(f"عدد صور التدريب: {len(train_images)}")
print(f"عدد صور التحقق (validation): {len(val_images)}")
