import ast

import cv2
import numpy as np
import pandas as pd
import torch

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


# Thanks https://www.kaggle.com/tanlikesmath/siim-covid-19-detection-a-simple-eda
def dicom2array(path, voi_lut=True, fix_monochrome=True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return data


def box2arr(boxes):
    result = []

    for box in boxes:
        result.append([
            box['x'], box['y'], box['x'] + box['width'], box['y'] + box['height']
        ])

    result = np.array(result).astype(np.int32)
    result[result < 0] = 0
    return result


def make_mask(img, boxes):
    mask = np.zeros_like(img)
    for i in range(len(boxes)):
        x, y, x2, y2 = boxes[i]

        mask[:, y:y2, x:x2] = 1
    mask = mask.squeeze()
    mask = cv2.resize(mask, (32, 32))
    mask = mask.reshape(1, 512, 512)
    return mask


class DatasetRetriever:

    def __init__(self, df, aug, cls_mode=False):
        super().__init__()
        self.cls_mode = cls_mode
        self.df = df
        self.aug = aug

    def __getitem__(self, index: int):
        image, boxes = self.load_image_and_boxes(index)

        pneumo_type = self.get_label(index)

        if self.cls_mode:
            if 'dom' in self.df.columns and self.df['dom'][index] == 'train':
                dom = 0
            else:
                dom = 1
            mask = make_mask(img=image, boxes=boxes)
            return image, pneumo_type, dom, mask

        labels = torch.ones((boxes.shape[0],), dtype=torch.int64) * pneumo_type

        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([index])}

        return image, target

    def get_label(self, index):
        if self.df['Negative for Pneumonia'][index] == 1:
            return 0
        if self.df['Typical Appearance'][index] == 1:
            return 1
        elif self.df['Indeterminate Appearance'][index] == 1:
            return 2
        elif self.df['Atypical Appearance'][index] == 1:
            return 3

    def __len__(self) -> int:
        return self.df.shape[0]

    def load_image_and_boxes(self, idx):

        dcm_path = self.df.path[idx]

        img = dicom2array(path=dcm_path)

        if not self.cls_mode:
            if pd.isna(self.df['boxes'][idx]):
                boxes = []
            else:
                boxes = ast.literal_eval(self.df['boxes'][idx])
            boxes = box2arr(boxes)
        else:
            boxes = np.array([0, 0, 1, 1])

        augm = self.aug(image=img, bboxes=[boxes], labels=[0])

        img = augm['image']
        boxes = np.array(augm['bboxes']).astype(np.int32)

        # if len(boxes) > 0:
        #     boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        #     boxes[:, 3] = boxes[:, 1] + boxes[:, 3] # already performed in box2arr
        # boxes = np.clip(boxes, 0, 1) # not required

        return img, boxes

    # def load_cutmix_image_and_boxes(self, index, imsize=1024):
    #     """
    #     This implementation of cutmix author:  https://www.kaggle.com/nvnnghia
    #     Refactoring and adaptation: https://www.kaggle.com/shonenkov
    #     """
    #     w, h = imsize, imsize
    #     s = imsize // 2
    #
    #     xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
    #     indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]
    #
    #     result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
    #     result_boxes = []
    #
    #     for i, index in enumerate(indexes):
    #         image, boxes = self.load_image_and_boxes(index)
    #         if i == 0:
    #             x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
    #             x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
    #         elif i == 1:  # top right
    #             x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
    #             x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
    #         elif i == 2:  # bottom left
    #             x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
    #             x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
    #         elif i == 3:  # bottom right
    #             x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
    #             x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
    #         result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
    #         padw = x1a - x1b
    #         padh = y1a - y1b
    #
    #         boxes[:, 0] += padw
    #         boxes[:, 1] += padh
    #         boxes[:, 2] += padw
    #         boxes[:, 3] += padh
    #
    #         result_boxes.append(boxes)
    #
    #     result_boxes = np.concatenate(result_boxes, 0)
    #     np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
    #     result_boxes = result_boxes.astype(np.int32)
    #     result_boxes = result_boxes[
    #         np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)]
    #     return result_image, result_boxes
