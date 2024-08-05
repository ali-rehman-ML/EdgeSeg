import numpy as np
import cv2

def bilinear_upsample(tensor, new_height, new_width):
    n, c, h, w = tensor.shape
    upsampled = np.zeros((n, c, new_height, new_width), dtype=tensor.dtype)
    for i in range(n):
        for j in range(c):
            upsampled[i, j] = cv2.resize(tensor[i, j], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return upsampled

def bicubic_upsample(tensor, new_height, new_width):
    n, c, h, w = tensor.shape
    upsampled = np.zeros((n, c, new_height, new_width), dtype=tensor.dtype)
    for i in range(n):
        for j in range(c):
            upsampled[i, j] = cv2.resize(tensor[i, j], (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return upsampled



def resize_torch(
    x,
    size: any or None = None,
    scale_factor: list[float] or None = None,
    mode: str = "bicubic",
    align_corners: bool or None = False,
):
    import torch.nn.functional as F
    import torch
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")
def post_process_torch(x,size=(1024,2048)):
    resize_torch(x,size=size)

def post_process_numpy(x,size=(1024,2048)):
    return bicubic_upsample(x,size[0],size[1])


def post_process_output(x,size=(1024,2048)):
    if isinstance(x, np.ndarray):
        o=post_process_numpy(x,size)
        o=np.argmax(o,dim=1)
        return np.squeeze(o)
    else:
        import torch
        o=post_process_torch(x,size)
        o=torch.argmax(o,dim=1).squeeze().numpy()
        return o











