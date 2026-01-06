"""
===============================================================================
Module: BACKGROUND REMOVER
Mô tả: Sử dụng MediaPipe để xóa phông nền ảnh (selfie segmentation)

Môn: IE212 - Công nghệ Dữ liệu Lớn
===============================================================================
"""

import os
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BG_COLOR = (192, 192, 192)  # gray - Màu nền sau khi xóa phông
MASK_COLOR = (255, 255, 255)  # white - Màu mask

# Đường dẫn đến model (cùng thư mục với file này)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "selfie_segmenter.tflite")

# Kiểm tra file model tồn tại
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Không tìm thấy model tại: {model_path}")

# Khởi tạo Image Segmenter
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
segmenter = vision.ImageSegmenter.create_from_options(options)

def remove_background(frame: np.ndarray) -> np.ndarray:
    # creating mp image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    segmentation_result = segmenter.segment(mp_image)
    category_mask = segmentation_result.category_mask

    # remove the background
    image_data = mp_image.numpy_view()
    fg_image = np.zeros(image_data.shape, dtype=np.uint8)
    fg_image[:] = MASK_COLOR
    bg_image = np.zeros(image_data.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
    output_frame = np.where(condition, bg_image, image_data)

    return output_frame
