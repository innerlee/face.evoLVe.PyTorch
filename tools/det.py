#!/usr/bin/env python
import argparse
from PIL import Image
import numpy as np
from evolveface import detect_faces, show_results
from evolveface import get_reference_facial_points, warp_and_crop_face
import time
import glob

parser = argparse.ArgumentParser(description='find face')
parser.add_argument("-i", "--input", help="input image", type=str, default='play/1.jpg')
parser.add_argument("-o", "--out", help="output image file", default='x.jpg')
parser.add_argument("--crop_size", help="specify size of aligned faces", default=112, type=int)
args = parser.parse_args()
crop_size = args.crop_size
scale = crop_size / 112.
reference = get_reference_facial_points(default_square=True) * scale

files = sorted(glob.glob('data/anli1/*jpg'))

start = time.time()
for f in files:
    img = Image.open(f).convert('RGB')
    bounding_boxes, landmarks = detect_faces(img)
    np.save(f + '.npy', bounding_boxes)
    print(f, len(bounding_boxes))
end = time.time()
print(end - start)
