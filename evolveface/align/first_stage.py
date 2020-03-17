import torch
from torch.autograd import Variable
import math
from PIL import Image
import numpy as np
from .box_utils import nms, _preprocess, _preprocess_gpu
import cv2


def run_first_stage(args):
    """Run P-Net, generate bounding boxes, and do NMS.

    Arguments:
        image: an instance of PIL.Image.
        net: an instance of pytorch's nn.Module, P-Net.
        scale: a float number,
            scale width and height of the image by this number.
        threshold: a float number,
            threshold on the probability of a face when generating
            bounding boxes from predictions of the net.

    Returns:
        a float numpy array of shape [n_boxes, 9],
            bounding boxes with scores and offsets (4 + 1 + 4).
    """
    imgs = []
    imgt = args[0][0]
    for image, size, net, scale, threshold in args:
        # scale the image and convert it to a float array
        width, height = size
        sw, sh = math.ceil(width * scale), math.ceil(height * scale)
        img = cv2.resize(image, (sw, sh), cv2.INTER_LINEAR)
        img = torch.from_numpy(img).to("cuda:0")
        img = _preprocess_gpu(img)
        imgs.append(img)

    outputs = []
    with torch.no_grad():
        for img in imgs:
            outputs.append(net(img))

    allboxes = []
    for output, (_, _, _, scale, threshold) in zip(outputs, args):
        probs = output[1][0, 1, :, :]
        offsets = output[0]
        # probs: probability of a face at each sliding window
        # offsets: transformations to true bounding boxes
        boxes = _generate_bboxes(probs, offsets, scale, threshold)
        if len(boxes) == 0:
            continue
        keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
        allboxes.append(boxes[keep])

    return allboxes


def _generate_bboxes(probs, offsets, scale, threshold):
    """Generate bounding boxes at places
    where there is probably a face.

    Arguments:
        probs: a float numpy array of shape [n, m].
        offsets: a float numpy array of shape [1, 4, n, m].
        scale: a float number,
            width and height of the image were scaled by this number.
        threshold: a float number.

    Returns:
        a float numpy array of shape [n_boxes, 9]
    """

    # applying P-Net is equivalent, in some sense, to
    # moving 12x12 window with stride 2
    stride = 2
    cell_size = 12
    # indices of boxes where there is probably a face
    inds = torch.where(probs > threshold)

    if inds[0].size(0) == 0:
        return np.array([])

    # transformations of bounding boxes
    tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
    # they are defined as:
    # w = x2 - x1 + 1
    # h = y2 - y1 + 1
    # x1_true = x1 + tx1*w
    # x2_true = x2 + tx2*w
    # y1_true = y1 + ty1*h
    # y2_true = y2 + ty2*h

    offsets = torch.stack([tx1, ty1, tx2, ty2])
    score = probs[inds[0], inds[1]]
    # P-Net is applied to scaled images
    # so we need to rescale bounding boxes back
    bounding_boxes = torch.cat([
        torch.round((stride * inds[1].float() + 1.0) / scale)[None, :],
        torch.round((stride * inds[0].float() + 1.0) / scale)[None, :],
        torch.round((stride * inds[1].float() + 1.0 + cell_size) / scale)[None, :],
        torch.round((stride * inds[0].float() + 1.0 + cell_size) / scale)[None, :], score[None, :], offsets
    ]).permute(1, 0).cpu().numpy()
    # why one is added?

    return bounding_boxes
