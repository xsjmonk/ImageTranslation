import numpy as np
import torch
from PIL import Image
def preprocess_pixels(pixels: np.ndarray, size: int, mean, std, padding_rgb=(0,0,0)) -> torch.Tensor:
    if pixels.ndim != 3 or pixels.shape[2] != 3: raise ValueError("expected RGB pixels")
    h,w = pixels.shape[:2]; scale = min(size/w, size/h)
    image = Image.fromarray(pixels.astype("uint8"), "RGB")
    image = image.resize((max(1,round(w*scale)), max(1,round(h*scale))), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size,size), tuple(padding_rgb))
    canvas.paste(image, ((size-image.width)//2, (size-image.height)//2))
    a = np.asarray(canvas, dtype=np.float32) / 255
    return torch.from_numpy(((a-np.asarray(mean))/np.asarray(std)).transpose(2,0,1)).float()
