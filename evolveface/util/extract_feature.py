# Helper function for extracting features from pre-trained models
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os
from PIL import Image
from .dataset_memory import DatasetMemory
from evolveface import IR_50
from evolveface import get_reference_facial_points, warp_and_crop_face

BACKBONE_IR50A = None


def l2_normalize(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def extract_feature_IR50A(img, landmarks):
    img_np = np.array(img)
    batch_size = 256

    # load backbone from a checkpoint
    global BACKBONE_IR50A
    if BACKBONE_IR50A is None:
        model_path = os.path.join(os.path.dirname(__file__), '../models/backbone_ir50_asia.pth')
        assert (os.path.exists(model_path))
        # print("Loading Backbone Checkpoint '{}'".format(model_path))
        BACKBONE_IR50A = IR_50([112, 112])
        BACKBONE_IR50A.load_state_dict(torch.load(model_path))
        BACKBONE_IR50A.to("cuda:0")
        BACKBONE_IR50A.eval()  # set to evaluation mode
    backbone = BACKBONE_IR50A

    # get faces from landmarks
    scale = 128. / 112.
    reference = get_reference_facial_points(default_square=True) * scale
    faces = []
    for i in range(len(landmarks)):
        facial5points = [[landmarks[i][j], landmarks[i][j + 5]] for j in range(5)]
        warped_face = Image.fromarray(warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(128, 128)))
        faces.append(warped_face)
        faces.append(transforms.functional.hflip(warped_face))

    # define data loader
    transform = transforms.Compose([
        transforms.Resize([128, 128]),  # smaller side resized
        transforms.CenterCrop([112, 112]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = DatasetMemory(faces, transform)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=0,
                                         drop_last=False)

    # extract features
    idx = 0
    features = torch.zeros([len(loader.dataset), 512])
    with torch.no_grad():
        iter_loader = iter(loader)
        while idx + batch_size <= len(loader.dataset):
            batch, _ = iter_loader.next()
            features[idx:idx + batch_size] = backbone(batch.to("cuda:0"))
            idx += batch_size

        if idx < len(loader.dataset):
            batch, _ = iter_loader.next()
            features[idx:] = backbone(batch.to("cuda:0"))

    features = l2_normalize(features[::2] + features[1::2]).cpu().numpy()
    return features
