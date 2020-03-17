#!/usr/bin/env python
import argparse
from PIL import Image
import numpy as np
from evolveface import detect_faces, show_results
from evolveface import get_reference_facial_points, warp_and_crop_face
import time

parser = argparse.ArgumentParser(description='find face')
parser.add_argument("-i", "--input", help="input image", type=str, default='play/1.jpg')
parser.add_argument("-o", "--out", help="output image file", default='x.jpg')
parser.add_argument("--crop_size", help="specify size of aligned faces", default=112, type=int)
args = parser.parse_args()
crop_size = args.crop_size
scale = crop_size / 112.
reference = get_reference_facial_points(default_square=True) * scale

img = Image.open(args.input).convert('RGB')
start = time.time()
bounding_boxes, landmarks = detect_faces(img)
print(bounding_boxes)
print(len(bounding_boxes))
end = time.time()
print(end - start)
show_results(img, bounding_boxes, landmarks).save(args.out)

for i in range(len(landmarks)):
    facial5points = [[landmarks[i][j], landmarks[i][j + 5]] for j in range(5)]
    warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
    Image.fromarray(warped_face).save(f'y{i}.jpg')
