import numpy as np
import torch
import torch.nn.functional as F
from hausdorff import hausdorff_distance
SMOOTH = 1e-5
'''
def mean_iou(y_true_in, y_pred_in, print_table=False):
    if True: #not np.sum(y_true_in.flatten()) == 0:
        labels = y_true_in
        y_pred = y_pred_in

        true_objects = 2
        pred_objects = 2

        intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

        # Compute areas (needed for finding the union between all objects)
        area_true = np.histogram(labels, bins = true_objects)[0]
        area_pred = np.histogram(y_pred, bins = pred_objects)[0]
        area_true = np.expand_dims(area_true, -1)
        area_pred = np.expand_dims(area_pred, 0)

        # Compute union
        union = area_true + area_pred - intersection

        # Exclude background from the analysis
        intersection = intersection[1:,1:]
        union = union[1:,1:]
        union[union == 0] = 1e-9

        # Compute the intersection over union
        iou = intersection / union

        # Precision helper function
        def precision_at(threshold, iou):
            matches = iou > threshold
            true_positives = np.sum(matches, axis=1) == 1   # Correct objects
            false_positives = np.sum(matches, axis=0) == 0  # Missed objects
            false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
            tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
            return tp, fp, fn

        # Loop over IoU thresholds
        prec = []
        if print_table:
            print("Thresh\tTP\tFP\tFN\tPrec.")
        for t in np.arange(0.5, 1.0, 0.05):
            tp, fp, fn = precision_at(t, iou)
            if (tp + fp + fn) > 0:
                p = tp / (tp + fp + fn)
            else:
                p = 0
            if print_table:
                print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
            prec.append(p)

        if print_table:
            print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
        return np.mean(prec)

    else:
        if np.sum(y_pred_in.flatten()) == 0:
            return 1
        else:
            return 0

def batch_iou(output, target):
    output = torch.sigmoid(output).data.cpu().numpy() > 0.5
    target = (target.data.cpu().numpy() > 0.5).astype('int')
    output = output[:,0,:,:]
    target = target[:,0,:,:]

    ious = []
    for i in range(output.shape[0]):
        ious.append(mean_iou(output[i], target[i]))

    return np.mean(ious)

def mean_iou(output, target):
    smooth = 1e-5

    # output = torch.sigmoid(output).data.cpu().numpy()
    output = output.data.cpu().numpy()
    target = target.data.cpu().numpy()
    ious = []
    for t in np.arange(0.5, 1.0, 0.05):
        output_ = output > t
        target_ = target > t
        intersection = (output_ & target_).sum()
        union = (output_ | target_).sum()
        iou = (intersection + smooth) / (union + smooth)
        # print(iou)
        ious.append(iou)

    return np.mean(ious)

def iou_score(output, target, thr=0.5):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > thr
    target_ = target > thr
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)
def dice_coef(output, target):
    # smooth = 1e-5

    # if torch.is_tensor(output):
    #     output = torch.sigmoid(output).data.cpu().numpy()
    # if torch.is_tensor(target):
    #     target = target.data.cpu().numpy()
    # #output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    # #target = target.view(-1).data.cpu().numpy()

    # intersection = (output * target).sum()

    # return (2. * intersection + smooth) / \
    #     (output.sum() + target.sum() + smooth)
    smooth = 1e-5
    num = output.shape[0]
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output_1 = output[:,0,:,:]
    output_2 = output[:,1,:,:]

    target_1 = target[:,0,:,:]
    target_2 = target[:,1,:,:]

    # input_1 = input_1.view(num, -1)
    # target_1 = target_1.view(num, -1)

    intersection_1 = (output_1 * target_1)
    intersection_2 = (output_2 * target_2)

    dice_1 = (2. * intersection_1.sum() + smooth) / (output_1.sum() + target_1.sum() + smooth)
    dice_2 = (2. * intersection_2.sum() + smooth) / (output_2.sum() + target_2.sum() + smooth)
    # if dice_1 > 1.:
    #     print(output_1.shape, output_1.min(), output_1.max(), output_1.sum())
    #     print(target_1.shape, target_1.min(), target_1.max(), target_1.sum())
    #     print(intersection_1.shape, intersection_1.min(), intersection_1.max())
    #     print(2 * intersection_1.sum(), output_1.sum() + target_1.sum())
    #     assert 1>4

    return dice_1, dice_2
'''

class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """
    def __init__(self, num_classes): #我们应该是2
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes)) # matrix 2*2
 
    def _fast_hist(self, label_pred, label_true): # 计算一行(1*256)的混淆矩阵
        # 找出标签中需要计算的类别 去掉背景
        mask = (label_true >= 0) & (label_true < self.num_classes)
        # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes) ## core code
        t = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask], minlength=self.num_classes ** 2)
        # print(t)
        # assert 1>4
        return hist
 
    def add_batch(self, predictions, gts):# 计算一张256*256图的混淆矩阵 print[[65536. 0.][0. 0.]]
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())# flatten()按照行展成一行

    def evaluate(self): # 对单张图像
        np.seterr(divide='ignore',invalid='ignore')
        # print(np.diag(self.hist).sumtarget(), self.hist.sum())
        # print('--->', np.diag(self.hist), self.hist.sum(axis=1), self.hist.sum(axis=0))
        # print(self.hist)
        Accuracy = (np.diag(self.hist).sum() + SMOOTH) / (self.hist.sum() + SMOOTH) # PA = 识别正确的像素/全部像素  精确率
        IoU = (np.diag(self.hist) + SMOOTH) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist) + SMOOTH)  #IoU 并集SA + SB - (SA交SB)，前景背景交换也求了一次iou

        mIoU = np.nanmean(IoU)
        return Accuracy, mIoU

def dice_coef(output, target):
    # if torch.is_tensor(output):
    #     output = output.cpu().numpy()
    # if torch.is_tensor(target):
    #     target = target..cpu().numpy()
    intersection = (output * target).sum()
    # print(f"{intersection}/{output.sum()}+{target.sum()}")
    return (2. * intersection + SMOOTH) / \
        (output.sum() + target.sum() + SMOOTH)

def accuracy(output, target):
    return (output == target).sum() / len(output.flatten())

def ppv(output, target):
    # smooth = 1e-5
    # if torch.is_tensor(output):
    #     output = torch.sigmoid(output).data.cpu().numpy()
    # if torch.is_tensor(target):
    #     target = target.data.cpu().numpy()
    intersection = (output * target).sum()
    return  (intersection + SMOOTH) / \
           (output.sum() + SMOOTH)

def sensitivity(output, target):
    # smooth = 1e-5
    # if torch.is_tensor(output):
    #     output = torch.sigmoid(output).data.cpu().numpy()
    # if torch.is_tensor(target):
    #     target = target.data.cpu().numpy()

    intersection = (output * target).sum()
    return (intersection + SMOOTH) / \
        (target.sum() + SMOOTH)

def get_metric(predict, mask, thr):
    """
    predict: output of the network, shape of [b, h, w]
    mask: ground truth , shape of [b, h ,w]
    thr: thresh hold
    """
    batch_size = mask.shape[0]

    predict = predict.data.cpu().numpy()
    mask = mask.data.cpu().numpy()

    predict[predict >= thr] = 1
    predict[predict < thr] = 0
    mask[mask >= thr] = 1
    mask[mask < thr] = 0

    predict = predict.astype(np.int16)
    mask = mask.astype(np.int16)

    m_dsc = []
    m_iou = []
    m_acc = []
    m_ppv = []
    m_sen = []
    m_hd = []

    for b in range(batch_size):
        m_dsc.append(dice_coef(predict[b], mask[b]))
        m_ppv.append(ppv(predict[b], mask[b]))
        m_sen.append(sensitivity(predict[b], mask[b]))
        m_hd.append(hausdorff_distance(predict[b], mask[b]))

        Iou = IOUMetric(2)
        Iou.add_batch(predict[b], mask[b])
        acc, miou = Iou.evaluate() 
        m_iou.append(miou)
        m_acc.append(acc)

    return np.nanmean(m_dsc), np.nanmean(m_iou), np.nanmean(m_acc), np.nanmean(m_ppv), np.nanmean(m_sen), np.nanmean(m_hd)

if __name__ == "__main__":
    # ## test dice_coef
    # torch.manual_seed(100)
    # output = torch.randn(1,2,2,2).sigmoid() # b, c, h, w
    # mask = torch.rand(1,2,2,2).ge(0.4).int()
    # print(output)
    # print(mask)
    # print(iou_score(output, mask))
    # print(dice_coef(output, mask))

    # test get_metric
    torch.manual_seed(0)
    output = torch.randn(2, 256, 256) # .sigmoid() # b, h, w
    mask = torch.rand(2, 256, 256).ge(0.5).int()
    dsc, miou, acc, ppv, sen, hd = get_metric(output, mask, 0.4)
    print(dsc, miou, acc, ppv, sen, hd)