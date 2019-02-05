import numpy as np
from itertools import product


def compute_IoU(box1, box2):
    return box1.intersection(box2).area / box1.union(box2).area


def compute_ious(pred, gt):
    ious = []
    iou_matrix = np.array(
        np.split(
            np.array([compute_IoU(x1, x2) for x1, x2 in product(pred, gt)]),
            len(pred)))
    while True:
        if iou_matrix.shape[1] == 0 or iou_matrix.shape[0] == 0:
            print(iou_matrix)
            break
        if iou_matrix.shape[1] == 1:
            ious.extend(iou_matrix.reshape(-1).tolist())
            break
        max_row = iou_matrix.max(axis=1).argmax()
        max_col = iou_matrix[max_row].argmax()
        ious.append(iou_matrix[max_row, max_col])
        iou_matrix = np.delete(iou_matrix, max_row, axis=0)
        iou_matrix = np.delete(iou_matrix, max_col, axis=1)
    return np.array(ious)


def mAP(pred, gt, range_start=0.5, range_end=0.95, range_step=0.05):
    if len(pred) == 0:
        return 0
    iou_matrix = np.array(
        np.split(
            np.array([compute_IoU(x1, x2) for x1, x2 in product(pred, gt)]),
            len(pred)))
    if range_start < range_end:
        thresholds = np.arange(range_start, range_end, range_step)
    elif range_start == range_end:
        thresholds = [range_start]
    precision = 0
    for thresh in thresholds:
        tp = (iou_matrix.max(axis=0) > thresh).sum()
        fn = len(gt) - tp
        fp = (iou_matrix.max(axis=1) < thresh).sum()
        precision += tp / (tp + fn + fp)
    return precision / len(thresholds)


def per_class_mAP(pred, gt, range_start=0.5, range_end=0.95, range_step=0.05):
    gt_sale_idx = gt[:, 0] == 'sale_description'
    gt_section_idx = gt[:, 0] == 'section_author'
    if len(pred) <= 0:
        if len(gt) > 0:
            return [0.0, 0.0]
        else:
            return [1.0, 1.0]
    pred_sale_idx = pred[:, 0] == 'sale_description'
    pred_section_idx = pred[:, 0] == 'section_author'
    if sum(gt_sale_idx) == 0:
        if sum(pred_sale_idx) > 0:
            mAP_sale = 0.0
        else:
            mAP_sale = 1.0
    elif sum(pred_sale_idx) > 0:
        gt_sale = gt[gt_sale_idx][:, 1]
        pred_sale = pred[pred_sale_idx][:, 1]
        mAP_sale = mAP(pred_sale, gt_sale, range_start, range_end, range_step)
    else:
        mAP_sale = 0.0

    if sum(gt_section_idx) == 0:
        if sum(pred_section_idx) > 0:
            mAP_section = 0.0
        else:
            mAP_section = 1.0
    elif sum(pred_section_idx) > 0:
        gt_pred = gt[gt_section_idx][:, 1]
        pred_section = pred[pred_section_idx][:, 1]
        mAP_section = mAP(pred_section, gt_pred, range_start, range_end,
                          range_step)
    else:
        mAP_section = 0.0

    return [mAP_sale, mAP_section]