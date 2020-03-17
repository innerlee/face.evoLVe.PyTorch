#!/usr/bin/env python
import argparse
import glob
import time

import numpy as np
from PIL import Image

from evolveface import detect_faces, get_reference_facial_points

parser = argparse.ArgumentParser(description='find face')
parser.add_argument("input", help="input folder", type=str)
parser.add_argument("--crop_size", help="specify size of aligned faces", default=112, type=int)
args = parser.parse_args()

crop_size = args.crop_size
scale = crop_size / 112.
reference = get_reference_facial_points(default_square=True) * scale
files = sorted(glob.glob(args.input + '/*jpg'))

start = time.time()

for f in files:
    img = Image.open(f).convert('RGB')
    bounding_boxes, landmarks = detect_faces(img)
    np.save(f + '.npy', bounding_boxes)
    print(f, len(bounding_boxes))

end = time.time()
print(end - start)
