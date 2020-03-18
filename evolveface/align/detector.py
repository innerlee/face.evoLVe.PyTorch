import numpy as np
from PIL import Image
import torch
import torchvision as tv
from .get_nets import PNET, RNET, ONET
from .box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from .first_stage import run_first_stage


def detect_faces(image, min_face_size=20.0, thresholds=[0.6, 0.7, 0.8], nms_thresholds=[0.7, 0.7, 0.7]):
    """
    Arguments:
        image: an instance of PIL.Image.
        min_face_size: a float number.
        thresholds: a list of length 3.
        nms_thresholds: a list of length 3.

    Returns:
        two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
        bounding boxes and facial landmarks.
    """
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')

    img_array = np.array(image)

    # LOAD MODELS
    pnet = PNET
    rnet = RNET
    onet = ONET

    # BUILD AN IMAGE PYRAMID
    width, height = image.size
    min_length = min(height, width)

    min_detection_size = 12
    factor = 0.707  # sqrt(0.5)

    # scales for scaling the image
    scales = []

    # scales the image so that
    # minimum size that we can detect equals to
    # minimum face size that we want to detect
    m = min_detection_size / min_face_size
    min_length *= m

    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m * factor**factor_count)
        min_length *= factor
        factor_count += 1

    # STAGE 1
    # run P-Net on different scales
    args = []
    for s in scales:
        args.append((img_array, image.size, pnet, s, thresholds[0]))

    bounding_boxes = run_first_stage(args)
    # collect boxes (and offsets, and scores) from different scales
    bounding_boxes = [i for i in bounding_boxes if i is not None]
    if len(bounding_boxes) == 0:
        return [], []
    bounding_boxes = torch.cat(bounding_boxes)

    # keep = nms(bounding_boxes.cpu().numpy()[:, 0:5], nms_thresholds[0])
    keep = tv.ops.nms(bounding_boxes[:, :4], bounding_boxes[:, 4], nms_thresholds[0])
    bounding_boxes = bounding_boxes[keep].cpu().numpy()

    # use offsets predicted by pnet to transform bounding boxes
    bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
    # shape [n_boxes, 5]

    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 2

    img_boxes = get_image_boxes(bounding_boxes, img_array, width, height, size=24)
    img_boxes = torch.FloatTensor(img_boxes).to("cuda:0")
    with torch.no_grad():
        output = rnet(img_boxes)
    offsets = output[0].cpu().numpy()  # shape [n_boxes, 4]
    probs = output[1].cpu().numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[1])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1, ))
    offsets = offsets[keep]

    keep = nms(bounding_boxes, nms_thresholds[1])
    # import ipdb; ipdb.set_trace()
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 3

    img_boxes = get_image_boxes(bounding_boxes, img_array, width, height, size=48)
    if len(img_boxes) == 0:
        return [], []
    img_boxes = torch.FloatTensor(img_boxes).to("cuda:0")
    with torch.no_grad():
        output = onet(img_boxes)
    landmarks = output[0].cpu().numpy()  # shape [n_boxes, 10]
    offsets = output[1].cpu().numpy()  # shape [n_boxes, 4]
    probs = output[2].cpu().numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[2])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1, ))
    offsets = offsets[keep]
    landmarks = landmarks[keep]

    # compute landmark points
    width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
    height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
    xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
    landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
    landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

    bounding_boxes = calibrate_box(bounding_boxes, offsets)
    keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
    # import ipdb; ipdb.set_trace()
    bounding_boxes = bounding_boxes[keep]
    landmarks = landmarks[keep]

    return bounding_boxes, landmarks
