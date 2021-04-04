import sys
sys.path.append('..')
import threading
import torch
import numpy as np
import torch.nn.functional as F
from util.IOU import *

labels = [
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
    (111, 74, 0),
    (81, 0, 81),
    (128, 64, 128),
    (244, 35, 232),
    (250, 170, 160),
    (230, 150, 140),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (180, 165, 180),
    (150, 100, 100),
    (150, 120, 90),
    (153, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 0, 90),
    (0, 0, 110),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
    (0, 0, 142),
]


class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scroes"""

    def __init__(self, nclass):
        self.nclass = nclass
        self.total_iou = []
        self.hist = np.zeros((nclass, nclass))
        self.lock = threading.Lock()
        self.reset()

    def update(self, labels, preds):
        def evaluate_worker(self, label, pred):
            correct, labeled = batch_pix_accuracy(
                pred, label)
            # inter, union = batch_intersection_union(
            #     pred, label, self.nclass)
            IOU = batch_intersection_union(pred, label, self.nclass)
            with self.lock:
                self.total_correct += correct
                self.total_label += labeled
                # self.total_inter += inter
                # self.total_union += union
                # self.hist +=hist
                self.total_iou.append(IOU)
            return

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, label, pred),
                                        )
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            raise NotImplemented

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        # IoU = self.total_inter / (1e-10 + self.total_union)
        mIoU = np.array(self.total_iou).mean()
        # ious = per_class_iu(self.hist) * 100
        return pixAcc, mIoU  # IoU.mean()

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        return


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    output = torch.squeeze(output, 0).detach().cpu().numpy()
    target = torch.squeeze(target, 0).detach().cpu().numpy()
    # print(output.ndim)
    # assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = 255
    target[np.where(target == ignore_index)[0]] = 255
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def batch_pix_accuracy(output, target):
    """Batch Pixel Accuracy
    Args:
        predict: input 4D tensor
        target: label 3D tensor
    """
    # predict = torch.max(output, 1)[1]
    # label: 0, 1, ..., nclass - 1
    # Note: 0 is background
    output = output.data
    target = target.data
    predict = output[0].detach().cpu().numpy()
    target = target[0].detach().cpu().numpy()

    predict = (np.transpose(predict, (1, 2, 0)) + 1)
    target = (np.transpose(target, (1, 2, 0)) + 1)

    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((np.isclose(predict, target, rtol=0, atol=(0.06, 0.06, 0.06))) * (target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    # output = output.data
    # target = target.data
    # predict = output[0].detach().cpu().numpy()
    # target = target[0].detach().cpu().numpy()
    #
    # im_pred = (np.transpose(predict, (1, 2, 0)) + 1)
    # im_lab = (np.transpose(target, (1, 2, 0)) + 1)
    #
    # im_pred = im_pred * (im_lab > 0)
    # # Compute area intersection:
    # intersection = im_pred * (im_pred == im_lab)
    # area_inter, _ = np.histogram(intersection, bins=nclass - 1,
    #                              range=(1, nclass - 1))
    # # Compute area union:
    # area_pred, _ = np.histogram(im_pred, bins=nclass - 1,
    #                             range=(1, nclass - 1))
    # area_lab, _ = np.histogram(im_lab, bins=nclass - 1,
    #                            range=(1, nclass - 1))
    # area_union = area_pred + area_lab - area_inter
    #
    # return area_inter, area_union
    # mini = 1
    # maxi = nclass - 1
    # nbins = nclass - 1
    #
    # # label is: 0, 1, 2, ..., nclass-1
    # # Note: 0 is background
    # predict = torch.squeeze(output, 0).detach().cpu().numpy().astype('int64') + 1
    # target = torch.squeeze(target, 0).detach().cpu().numpy().astype('int64')+1
    #
    # predict = predict * (target > 0).astype(predict.dtype)
    # intersection = predict * (predict == target)
    #
    # # areas of intersection and union
    # area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    # area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    # area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    # area_union = area_pred + area_lab - area_inter
    # assert (area_inter <= area_union).all(), \
    #     "Intersection area should be smaller than Union area"
    # return area_inter, area_union

    # predict = torch.squeeze(output, 0).detach().cpu().numpy()+1
    # target = torch.squeeze(target, 0).detach().cpu().numpy()+1

    # return area_inter, area_union
    # final = output[0]
    # final = resize_4d_tensor(final, 256, 256)
    # pred = final.argmax(axis=0)
    # print(pred.shape)
    # label = target.detach().cpu().numpy()
    # hist = fast_hist(pred.flatten(), label.flatten(), nclass)
    # return hist

    output = output.data
    target = target.data
    predict = output[0].detach().cpu().numpy()
    target = target[0].detach().cpu().numpy()

    predict = (np.transpose(predict, (1, 2, 0)) + 1)
    target = (np.transpose(target, (1, 2, 0)) + 1)
    # predict = cv2.cvtColor(predict, cv2.COLOR_RGB2GRAY)
    # target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)

    ious = []

    for cls in np.arange(1, len(labels)):  # This goes from 1:n_classes-1 -> class "0" is ignored
        cls = labels[cls]
        pred_inds = np.isclose(predict, cls, rtol=1e-7, atol=(0.06, 0.06, 0.06))
        target_inds = np.isclose(target, cls, rtol=0, atol=(0, 0, 0))
        intersection = np.nan_to_num((pred_inds[target_inds]).sum())  # Cast to long to prevent overflows

        union = np.nan_to_num(pred_inds.sum()) + np.nan_to_num(target_inds.sum()) - intersection
        if union == 0:
            ious.append(float(0))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))

    ious = np.array(ious)

    if np.count_nonzero(np.array(ious > 0, dtype=np.uint8)) == 0:
        return 0

    return np.mean(ious[ious > 0])


# ref https://github.com/CSAILVision/sceneparsing/blob/master/evaluationCode/utils_eval.py
def pixel_accuracy(im_pred, im_lab):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)

    # Remove classes from unlabeled pixels in gt image. 
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(im_lab > 0)
    pixel_correct = np.sum((im_pred == im_lab) * (im_lab > 0))

    return pixel_correct, pixel_labeled


def intersection_and_union(im_pred, im_lab, num_class):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)
    # Remove classes from unlabeled pixels in gt image. 
    im_pred = im_pred * (im_lab > 0)
    # Compute area intersection:
    intersection = im_pred * (im_pred == im_lab)
    area_inter, _ = np.histogram(intersection, bins=num_class - 1,
                                 range=(1, num_class - 1))
    # Compute area union: 
    area_pred, _ = np.histogram(im_pred, bins=num_class - 1,
                                range=(1, num_class - 1))
    area_lab, _ = np.histogram(im_lab, bins=num_class - 1,
                               range=(1, num_class - 1))
    area_union = area_pred + area_lab - area_inter
    return area_inter, area_union


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def rel_abs_vol_diff(y_true, y_pred):
    return np.abs((y_pred.sum() / y_true.sum() - 1) * 100)


def get_boundary(data, img_dim=2, shift=-1):
    data = data > 0
    edge = np.zeros_like(data)
    for nn in range(img_dim):
        edge += ~(data ^ np.roll(~data, shift=shift, axis=nn))
    return edge.astype(int)


def numpy_dice(y_true, y_pred, axis=None, smooth=1.0):
    intersection = y_true * y_pred
    return (2. * intersection.sum(axis=axis) + smooth) / (
            np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis) + smooth)


def dice_coefficient(input, target, smooth=1.0):
    assert smooth > 0, 'Smooth must be greater than 0.'

    probs = F.softmax(input, dim=1)

    encoded_target = probs.detach() * 0
    encoded_target.scatter_(1, target.unsqueeze(1), 1)
    encoded_target = encoded_target.float()

    num = probs * encoded_target  # b, c, h, w -- p*g
    num = torch.sum(num, dim=3)  # b, c, h
    num = torch.sum(num, dim=2)  # b, c

    den1 = probs * probs  # b, c, h, w -- p^2
    den1 = torch.sum(den1, dim=3)  # b, c, h
    den1 = torch.sum(den1, dim=2)  # b, c

    den2 = encoded_target * encoded_target  # b, c, h, w -- g^2
    den2 = torch.sum(den2, dim=3)  # b, c, h
    den2 = torch.sum(den2, dim=2)  # b, c

    dice = (2 * num + smooth) / (den1 + den2 + smooth)  # b, c

    return dice.mean().mean()
