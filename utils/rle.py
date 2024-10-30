import numpy as np
import pycocotools.mask as mask_utils


def rle2mask(rle):
    return np.array(mask_utils.decode(rle), dtype=np.uint8) > 0


def mask2rle(mask):
    rle = mask_utils.encode(np.array(mask[:, :, None], order="F", dtype=np.uint8))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def mask2img(mask):
    h, w = mask.shape
    img = np.zeros([h, w, 3], dtype=np.uint8)
    img[mask] = 255
    return img
