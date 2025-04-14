from PIL import Image
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T


class AugmentWithoutResize:
    """
    適用於 Faster R-CNN 預設的 GeneralizedRCNNTransform。
    """

    def __init__(self):
        self.jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)

    def __call__(self, image, target):
        image = self.jitter(image)  # 僅做顏色增強
        return F.to_tensor(image), target


# 以下仍保留原有結構供 Swin 等固定輸入尺寸架構使用
class ResizeWithPaddingForTraining:
    def __init__(self, target_size=256, fill=114):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, image, target):
        orig_w, orig_h = image.size
        scale = self.target_size / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        image = image.resize((new_w, new_h), resample=Image.BILINEAR)
        pad_x = (self.target_size - new_w) // 2
        pad_y = (self.target_size - new_h) // 2
        padded_image = Image.new(
            "RGB", (self.target_size, self.target_size), color=(self.fill,) * 3
        )
        padded_image.paste(image, (pad_x, pad_y))

        if "boxes" in target:
            boxes = target["boxes"]
            boxes = boxes * scale
            boxes[:, [0, 2]] += pad_x
            boxes[:, [1, 3]] += pad_y
            target["boxes"] = boxes

        return F.to_tensor(padded_image), target


class ResizeWithPaddingAndBox:
    """
    用於 inference 階段：加入 resize_info 以供還原 bbox 座標。
    """

    def __init__(self, target_size=256, fill=114):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, image, target):
        orig_w, orig_h = image.size
        scale = self.target_size / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        image = image.resize((new_w, new_h), resample=Image.BILINEAR)
        pad_x = (self.target_size - new_w) // 2
        pad_y = (self.target_size - new_h) // 2
        padded_image = Image.new(
            "RGB", (self.target_size, self.target_size), color=(self.fill,) * 3
        )
        padded_image.paste(image, (pad_x, pad_y))

        if "boxes" in target:
            boxes = target["boxes"].clone()
            boxes = boxes * scale
            boxes[:, [0, 2]] += pad_x
            boxes[:, [1, 3]] += pad_y
            target["boxes"] = boxes

        target["resize_info"] = {
            "scale": scale,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "orig_size": (orig_w, orig_h),
        }

        image = F.to_tensor(padded_image)
        return image, target


def restore_boxes(boxes, resize_info):
    """
    還原經過 ResizeWithPaddingAndBox 之後的預測框為原圖座標。
    """
    scale = resize_info["scale"]
    pad_x = resize_info["pad_x"]
    pad_y = resize_info["pad_y"]

    restored = boxes.clone()
    restored[:, [0, 2]] -= pad_x
    restored[:, [1, 3]] -= pad_y
    restored /= scale
    return restored
