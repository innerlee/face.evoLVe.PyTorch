#!/usr/bin/env python
from PIL import Image
import numpy as np
from evolveface import detect_faces, show_results
from evolveface import get_reference_facial_points, warp_and_crop_face
import time

crop_size = 112
scale = crop_size / 112.
reference = get_reference_facial_points(default_square=True) * scale

for img in [
        "play/img_00005.jpg", "play/img_00276.jpg", "play/img_00277.jpg", "play/img_00278.jpg", "play/img_00279.jpg",
        "play/img_00280.jpg", "play/img_00281.jpg", "play/img_00282.jpg", "play/img_00283.jpg", "play/img_00284.jpg",
        "play/img_00285.jpg", "play/img_00286.jpg", "play/img_00287.jpg", "play/img_00288.jpg", "play/img_00323.jpg",
        "play/img_01718.jpg", "play/img_02692.jpg", "play/img_03152.jpg", "play/img_05397.jpg"
]:
    bounding_boxes, landmarks = detect_faces(Image.open(img).convert('RGB'))
