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
parser.add_argument("shot", help="which shot", type=int)
parser.add_argument("--input", help="input folder", default='data/sports', type=str)
parser.add_argument("--crop_size", help="specify size of aligned faces", default=112, type=int)
parser.add_argument("--shot_info",
                    help="specify path to shot info",
                    default='data/sports_shots/shot_movie/sports.txt',
                    type=str)
args = parser.parse_args()

shot = args.shot
t_start, t_end = np.loadtxt(args.shot_info, int)[shot, :2]
print(f'processing shot {shot}, frames {t_start} -- {t_end}')

crop_size = args.crop_size
scale = crop_size / 112.
reference = get_reference_facial_points(default_square=True) * scale

start = time.time()
result = []
for t in range(t_start, t_end+1):
    f = os.path.join(args.input, f'img_{t:05}.jpg')
    img = Image.open(f).convert('RGB')
    bounding_boxes, landmarks = detect_faces(img)
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
np.save(f'out/shot_{shot}.npy', result)

end = time.time()
print(end - start)
