import argparse
from PIL import Image
from evolveface import detect_faces
from evolveface import show_results

parser = argparse.ArgumentParser(description='find face')
parser.add_argument("input", help="input image", type=str)
parser.add_argument("-o", "--out", help="output image file", default='face.jpg')
args = parser.parse_args()

img = Image.open(args.input)
bounding_boxes, landmarks = detect_faces(img)
show_results(img, bounding_boxes, landmarks).save('x.jpg')
