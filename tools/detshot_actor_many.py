#!/usr/bin/env python
import argparse
import glob
import os
import time

from scipy.spatial.distance import cosine
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

# find the true one
features = [r['features'] for r in result]
faces = np.concatenate(features)
for (i, f) in enumerate(faces):
    found = np.zeros(len(features))
    goodfaces = []
    for (j, ff) in enumerate(features):
        dists = [cosine(f, g) for g in ff]
        if min(dists) < 0.5:
            found[j] = 1
            goodfaces.append(ff[np.argmin(dists)])
    if sum(found) == len(found):
        break

goodfaces = np.array(goodfaces)
np.save(f'out/face_{actor}.npy', goodfaces)
print(f'found {len(goodfaces)} faces for {actor}')

end = time.time()
print(end - start)
