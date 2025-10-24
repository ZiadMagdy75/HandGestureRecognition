import cv2
import numpy as np
import os

input_dir = "data/sign_data/cleaned_grabcut"
output_dir = "data/sign_data/filtered"
os.makedirs(output_dir, exist_ok=True)

removed = 0
kept = 0

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(".jpg"):
        continue

    path = os.path.join(input_dir, filename)
    img = cv2.imread(path)
    if img is None:
        continue

    # نحول الصورة لـGray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # نحسب نسبة البكسلات الغير بيضاء
    white = np.sum(gray > 240)  # البكسلات البيضاء جدًا
    total = gray.size
    non_white_ratio = 1 - (white / total)

    # لو أقل من 2% مش أبيض → معناها الصورة كلها يد أو جسم
    if non_white_ratio < 0.02:  
        # الخلفية كلها بيضاء تقريبًا → نحذف
        removed += 1
        continue

    # لو أكتر من 90% من الصورة مش أبيض → احتمال صورة جسم مش يد
    if non_white_ratio > 0.9:
        removed += 1
        continue

    # نحفظ الصور المقبولة
    cv2.imwrite(os.path.join(output_dir, filename), img)
    kept += 1

print(f"✅ تم الاحتفاظ بـ {kept} صورة وإزالة {removed} صورة غير مناسبة.")
print(f"📁 الصور النظيفة محفوظة في: {output_dir}")
