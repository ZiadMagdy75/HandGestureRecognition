import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# المسارات
input_dir = "data/sign_data/filtered"
output_dir = "data/sign_data/augmented"
os.makedirs(output_dir, exist_ok=True)

# إعدادات الـAugmentation
datagen = ImageDataGenerator(
    rotation_range=20,          # تدوير عشوائي ±20 درجة
    width_shift_range=0.1,      # تحريك أفقي بسيط
    height_shift_range=0.1,     # تحريك رأسي بسيط
    zoom_range=0.15,            # تكبير أو تصغير بسيط
    brightness_range=[0.8, 1.2],# تغيير في الإضاءة
    horizontal_flip=True,       # قلب أفقي
    fill_mode='nearest'
)

# عدد النسخ لكل صورة (تقدر تغيره حسب وقت المعالجة)
NUM_AUG_PER_IMAGE = 3

count = 0
for filename in os.listdir(input_dir):
    if not filename.lower().endswith(".jpg"):
        continue

    path = os.path.join(input_dir, filename)
    img = cv2.imread(path)
    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0)

    i = 0
    for batch in datagen.flow(img, batch_size=1,
                              save_to_dir=output_dir,
                              save_prefix=f"aug_{filename.split('.')[0]}",
                              save_format='jpg'):
        i += 1
        if i >= NUM_AUG_PER_IMAGE:
            break
    count += 1

print(f"✅ تم إنشاء صور Augmentation جديدة بنجاح!")
print(f"📸 عدد الصور الأصلية: {count}")
print(f"📁 الصور الجديدة محفوظة في: {output_dir}")
