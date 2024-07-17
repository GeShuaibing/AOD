import math
import torch


def bbox_iou4(box1,
             box2,
             feat_sz,
             xywh=True,
             GIoU=False,
             DIoU=False,
             CIoU=False,
             SIoU=False,
             EIoU=False,
             WIoU=False,
             MPDIoU=False,
             alpha=1,
             scale=False,
             monotonous=False,
             ratio=1.0,
             eps=1e-7):
    """
    计算bboxes iou
    Args:
        feat_sz: 特征图大小
        box1: predict bboxes
        box2: target bboxes
        xywh: 将bboxes转换为xyxy的形式
        GIoU: 为True时计算GIoU LOSS (yolov5自带)
        DIoU: 为True时计算DIoU LOSS (yolov5自带)
        CIoU: 为True时计算CIoU LOSS (yolov5自带，默认使用)
        SIoU: 为True时计算SIoU LOSS (新增)
        EIoU: 为True时计算EIoU LOSS (新增)
        WIoU: 为True时计算WIoU LOSS (新增)
        MPDIoU: 为True时计算MPDIoU LOSS (新增)
        alpha: AlphaIoU中的alpha参数，默认为1，为1时则为普通的IoU，如果想采用AlphaIoU，论文alpha默认值为3，此时设置CIoU=True则为AlphaCIoU
        scale: scale为True时，WIoU会乘以一个系数
        monotonous: 3个输入分别代表WIoU的3个版本，None: origin v1, True: monotonic FM v2, False: non-monotonic FM v3
        ratio: Inner-IoU对应的是尺度因子，通常取范围为[0.5，1.5],原文中VOC数据集对应的Inner-CIoU和Inner-SIoU设置在[0.7，0.8]之间有较大提升，
        数据集中大目标多则设置<1，小目标多设置>1
        eps: 防止除0
    Returns:
        iou
    """
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps
    if scale:
        wise_scale = WIoU_Scale(1 - (inter / union), monotonous=monotonous)

    # IoU
    # iou = inter / union # ori iou
    iou = torch.pow(inter / (union + eps), alpha)  # alpha iou
    feat_h, feat_w = feat_sz

    # Inner-IoU
    if xywh:
        inner_b1_x1, inner_b1_x2, inner_b1_y1, inner_b1_y2 = x1 - w1_ * ratio, x1 + w1_ * ratio, \
                                                             y1 - h1_ * ratio, y1 + h1_ * ratio
        inner_b2_x1, inner_b2_x2, inner_b2_y1, inner_b2_y2 = x2 - w2_ * ratio, x2 + w2_ * ratio, \
                                                             y2 - h2_ * ratio, y2 + h2_ * ratio
    else:
        x1, y1, x2, y2 = b1_x1, b1_y1, b2_x1, b2_y1
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        inner_b1_x1, inner_b1_x2, inner_b1_y1, inner_b1_y2 = x1 - w1_ * ratio, x1 + w1_ * ratio, \
                                                             y1 - h1_ * ratio, y1 + h1_ * ratio
        inner_b2_x1, inner_b2_x2, inner_b2_y1, inner_b2_y2 = x2 - w2_ * ratio, x2 + w2_ * ratio, \
                                                             y2 - h2_ * ratio, y2 + h2_ * ratio
    inner_inter = (torch.min(inner_b1_x2, inner_b2_x2) - torch.max(inner_b1_x1, inner_b2_x1)).clamp(0) * \
                  (torch.min(inner_b1_y2, inner_b2_y2) - torch.max(inner_b1_y1, inner_b2_y1)).clamp(0)
    inner_union = w1 * ratio * h1 * ratio + w2 * ratio * h2 * ratio - inner_inter + eps
    inner_iou = inner_inter / inner_union

    if CIoU or DIoU or GIoU or EIoU or SIoU or WIoU or MPDIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        c_area = cw * ch + eps  # convex area
        if CIoU or DIoU or EIoU or SIoU or WIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = (cw ** 2 + ch ** 2) ** alpha + eps  # convex diagonal squared
            rho2 = (((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (
                    b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4) ** alpha  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha_ciou = v / (v - iou + (1 + eps))
                return inner_iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha))  # CIoU
            elif EIoU:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = torch.pow(cw ** 2 + eps, alpha)
                ch2 = torch.pow(ch ** 2 + eps, alpha)
                return inner_iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)  # EIou
            elif SIoU:
                # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
                s_cw, s_ch = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps, (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
                sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
                sin_alpha_1, sin_alpha_2 = torch.abs(s_cw) / sigma, torch.abs(s_ch) / sigma
                threshold = pow(2, 0.5) / 2
                sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
                angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
                rho_x, rho_y = (s_cw / cw) ** 2, (s_ch / ch) ** 2
                gamma = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
                omiga_w, omiga_h = torch.abs(w1 - w2) / torch.max(w1, w2), torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                return inner_iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha)  # SIou
            elif WIoU:
                if scale:
                    return getattr(WIoU_Scale, '_scaled_loss')(wise_scale), (1 - inner_iou) * torch.exp(
                        (rho2 / c2)), inner_iou  # WIoU v3 https://arxiv.org/abs/2301.10051
                return inner_iou, torch.exp((rho2 / c2))  # WIoU v1
            return inner_iou - rho2 / c2  # DIoU
        elif MPDIoU:
            d1 = (b2_x1 - b1_x1) ** 2 + (b2_y1 - b1_y1) ** 2
            d2 = (b2_x2 - b1_x2) ** 2 + (b2_y2 - b1_y2) ** 2
            mpdiou_hw_pow = feat_h ** 2 + feat_w ** 2
            return inner_iou - d1 / mpdiou_hw_pow - d2 / mpdiou_hw_pow - torch.pow((c_area - union) / c_area + eps,
                                                                                   alpha)  # MPDIoU
        # c_area = cw * ch + eps  # convex area
        return inner_iou - torch.pow((c_area - union) / c_area + eps,
                                     alpha)  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


class WIoU_Scale:
    """
    monotonous: {
            None: origin v1
            True: monotonic FM v2
            False: non-monotonic FM v3
        }
        momentum: The momentum of running mean
    """
    iou_mean = 1.
    _momentum = 1 - pow(0.5, exp=1 / 7000)
    _is_train = True

    def __init__(self, iou, monotonous=False):
        self.iou = iou
        self.monotonous = monotonous
        self._update(self)

    @classmethod
    def _update(cls, self):
        if cls._is_train: cls.iou_mean = (1 - cls._momentum) * cls.iou_mean + \
                                         cls._momentum * self.iou.detach().mean().item()

    @classmethod
    def _scaled_loss(cls, self, gamma=1.9, delta=3):
        if isinstance(self.monotonous, bool):
            if self.monotonous:
                return (self.iou.detach() / self.iou_mean).sqrt()
            else:
                beta = self.iou.detach() / self.iou_mean
                alpha = delta * torch.pow(gamma, beta - delta)
                return beta / alpha
        return 1

#------------------------------------------------------------------------------
# iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
# lbox += (1.0 - iou).mean()  # iou loss

obj_sz = (tobj.size()[2], tobj.size()[3])
iou = bbox_iou(pbox, tbox[i], feat_sz=obj_sz, CIoU=True, ratio=0.8)
if isinstance(iou, tuple):
    if len(iou) == 2:
        lbox += (iou[1].detach().squeeze() * (1 - iou[0].squeeze())).mean()
        iou = iou[0].squeeze()
    else:
        lbox += (iou[0] * iou[1]).mean()
        iou = iou[2].squeeze()
else:
    lbox += (1.0 - iou.squeeze()).mean()  # iou loss
    iou = iou.squeeze()