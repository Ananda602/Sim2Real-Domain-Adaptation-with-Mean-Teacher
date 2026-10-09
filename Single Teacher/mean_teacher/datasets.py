import numpy as np
import random
from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms
from torchvision.transforms import functional as F  # low-level ops
from torchvision.transforms import InterpolationMode

# Optional export decorator (CuriousAI style)
try:
    from .utils import export
except Exception:
    def export(fn):  # no-op
        return fn

class TransformTwice:
    """Apply the same transform twice; returns (view1, view2)."""
    def __init__(self, transform): self.transform = transform
    def __call__(self, x): return self.transform(x), self.transform(x)

class TransformTwiceWeakStrong:
    """Apply weak and strong transforms; returns (weak_view, strong_view)."""
    def __init__(self, weak, strong): self.weak, self.strong = weak, strong
    def __call__(self, x): return self.weak(x), self.strong(x)

class GaussianBlur:
    """Random Gaussian blur with radius sampled from a range."""
    def __init__(self, radius_range=(0.1, 2.0)): self.radius_range = radius_range
    def __call__(self, img):
        r = float(np.random.uniform(*self.radius_range))
        return img.filter(ImageFilter.GaussianBlur(r))

class CopyPasteAug:
    """
    Lightweight Copy-Paste augmentation.
    Randomly copies one mask region and pastes it into another random background crop.
    """
    def __init__(self, prob=0.5, scale=(0.8, 1.2)):
        self.prob = prob
        self.scale = scale

    def __call__(self, img, mask):
        if random.random() > self.prob:
            return img, mask

        # Random affine jitter for pasted instance
        scale_factor = random.uniform(*self.scale)
        img_paste = img.resize([int(img.width * scale_factor), int(img.height * scale_factor)])
        mask_paste = mask.resize([int(mask.width * scale_factor), int(mask.height * scale_factor)])

        # Paste at a random offset (keeping within image bounds)
        x_off = random.randint(0, max(0, img.width - mask_paste.width))
        y_off = random.randint(0, max(0, img.height - mask_paste.height))

        # Paste mask and region
        img = img.copy()
        mask = mask.copy()
        img.paste(img_paste, (x_off, y_off), mask_paste)
        mask.paste(mask_paste, (x_off, y_off), mask_paste)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img, mask


class PairedWeakStrongTransform:
    """
    Apply the SAME geometry (resize + one shared random hflip)
    to both weak and strong views, then apply different photometric augs.

    weak:  light color jitter
    strong: stronger color jitter + Gaussian blur
    """
    def __init__(
        self,
        size=128,
        hflip_p: float = 0.5,
        weak_color = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        strong_color = transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2),
        add_blur=True
    ):
        self.resize = transforms.Resize((size, size), interpolation=InterpolationMode.BILINEAR)
        self.hflip_p = hflip_p

        weak_list = [weak_color, transforms.ToTensor()]
        strong_list = [strong_color]
        if add_blur:
            strong_list.append(GaussianBlur())
        strong_list.append(transforms.ToTensor())

        self.weak_photo = transforms.Compose(weak_list)
        self.strong_photo = transforms.Compose(strong_list)

    def __call__(self, img):
        # Shared geometry
        img = self.resize(img)
        do_flip = (np.random.rand() < self.hflip_p)
        if do_flip:
            img = ImageOps.mirror(img)  # horizontal flip

        # Photometric-only differences
        weak   = self.weak_photo(img)
        strong = self.strong_photo(img)
        return (weak, strong)

@export
def myforest_dataset():
    size = 256  # working size

    # Training: paired weak/strong with shared geometry
    train_transformation = PairedWeakStrongTransform(
        size=size,
        hflip_p=0.5,
        # You can tweak color jitter/blur via constructor if needed
    )

    # Evaluation: deterministic resize + ToTensor (no strong augs)
    eval_tf = transforms.Compose([
        transforms.Resize((size, size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])

    return {
        "train_transformation": train_transformation,  # returns (weak, strong)
        "eval_transformation": eval_tf,
        "datadir": "Dataset/cropped/",
        "num_classes": 1
    }

