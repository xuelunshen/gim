import cv2
import csv
import torch
import numpy as np

CLS_DICT = {}
with open('weights/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        name = row[5].split(";")[0]
        if name == 'screen':
            name = '_'.join(row[5].split(";")[:2])
        CLS_DICT[name] = int(row[0]) - 1

exclude = ['person', 'sky', 'car']


def read_deeplab_image(img, size):
    width, height = img.shape[1], img.shape[0]

    if max(width, height) > size:
        if width > height:
            img = cv2.resize(img, (size, int(size * height / width)), interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (int(size * width / height), size), interpolation=cv2.INTER_AREA)

    img = (torch.from_numpy(img.copy()).float() / 255).permute(2, 0, 1)[None]

    return img


def read_segmentation_image(img, size):
    img = read_deeplab_image(img, size=size)[0]
    # img = (torch.from_numpy(img).float() / 255).permute(2, 0, 1)
    img = img - torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    img = img / torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return img


def segment(rgb, size, device, segmentation_module):
    img_data = read_segmentation_image(rgb, size=size)
    singleton_batch = {'img_data': img_data[None].to(device)}
    output_size = img_data.shape[1:]
    # Run the segmentation at the highest resolution.
    scores = segmentation_module(singleton_batch, segSize=output_size)
    # Get the predicted scores for each pixel
    _, pred = torch.max(scores, dim=1)
    return pred.cpu()[0].numpy().astype(np.uint8)
