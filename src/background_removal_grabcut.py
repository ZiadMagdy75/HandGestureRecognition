import cv2
import numpy as np
import os

input_dir = "data/sign_data/processed"
output_dir = "data/sign_data/cleaned_grabcut"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)
    if img is None:
        continue

    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # تحديد منطقة اليد التقريبية (وسط الصورة)
    h, w = img.shape[:2]
    rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))

    # تنفيذ GrabCut
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # إنشاء الماسك النهائي
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = img * mask2[:, :, np.newaxis]

    # جعل الخلفية بيضاء
    white_bg = np.full_like(img, 255)
    cleaned = np.where(result == 0, white_bg, result)

    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, cleaned)

print("✅ تمت إزالة الخلفية بنجاح باستخدام GrabCut! الصور محفوظة في cleaned_grabcut/")
