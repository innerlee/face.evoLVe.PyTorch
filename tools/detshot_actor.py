#!/usr/bin/env python
import argparse
import glob
import os
import time

import numpy as np
from PIL import Image

from evolveface import detect_faces, get_reference_facial_points
from evolveface import extract_feature_IR50A

parser = argparse.ArgumentParser(description='find face')
parser.add_argument("actor", help="which actor", type=str)
parser.add_argument("--crop_size", help="specify size of aligned faces", default=112, type=int)
args = parser.parse_args()

actor = args.actor
files = glob.glob(f'data/photo/{actor}/*')
print(f'processing actor {actor}, {len(files)} photo')

crop_size = args.crop_size
scale = crop_size / 112.
reference = get_reference_facial_points(default_square=True) * scale

start = time.time()
result = []
for t, f in enumerate(files):
    img = Image.open(f).convert('RGB')
    bounding_boxes, landmarks = detect_faces(img)
    if len(bounding_boxes) > 1:
        n = np.argmax(bounding_boxes[:, -1])
        bounding_boxes = bounding_boxes[n:n+1, :]
        landmarks = landmarks[n:n+1, :]
    if len(bounding_boxes) > 0:
        features = extract_feature_IR50A(img, landmarks)
    else:
        features = []
    result.append(dict(
        t = t,
        bounding_boxes = bounding_boxes,
        landmarks = landmarks,
        features = features
    ))
    print(f, len(bounding_boxes))
np.save(f'out/face_{actor}.npy', result)

end = time.time()
print(end - start)
