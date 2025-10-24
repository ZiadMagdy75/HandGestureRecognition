import os

train_file = "data/train.txt"
aug_dir = "data/sign_data/augmented"

# نقرأ الملف القديم
with open(train_file, "r") as f:
    existing = [line.strip() for line in f.readlines()]

# نضيف المسارات الجديدة
augmented_paths = [f"sign_data/augmented/{f}" for f in os.listdir(aug_dir) if f.lower().endswith(".jpg")]

# ندمجهم ونحذف التكرار
combined = list(set(existing + augmented_paths))

# نكتبهم في train.txt من جديد
with open(train_file, "w") as f:
    for path in combined:
        f.write(path + "\n")

print(f"✅ تم تحديث train.txt وإضافة {len(augmented_paths)} صورة Augmented جديدة.")
print(f"📁 إجمالي الصور في التدريب الآن: {len(combined)}")
