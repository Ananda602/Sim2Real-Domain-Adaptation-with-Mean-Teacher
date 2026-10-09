import os
import itertools
import logging
import json
import random
import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image, ImageOps
from tqdm import tqdm
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision.transforms import InterpolationMode
from .datasets import myforest_dataset, CopyPasteAug
LOG = logging.getLogger('main')

# ---------------------------
# Two-Stream Batch Sampler (primary=unlabeled, secondary=labeled)
# ---------------------------
class TwoStreamBatchSampler(Sampler):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = list(primary_indices)
        self.secondary_indices = list(secondary_indices)

        LOG.info(f"[TwoStreamBatchSampler] primary_indices (unlabeled) = {len(self.primary_indices)}, "
                 f"secondary_indices (labeled) = {len(self.secondary_indices)}")

        if len(self.primary_indices) == 0:
            LOG.warning("No primary (unlabeled) samples found!")
        if len(self.secondary_indices) == 0:
            LOG.warning("No secondary (labeled) samples found!")

        sec_req = int(max(0, secondary_batch_size))
        self.secondary_batch_size = min(sec_req, len(self.secondary_indices))
        self.primary_batch_size = int(batch_size) - self.secondary_batch_size
        assert self.primary_batch_size > 0, "primary_batch_size must be > 0"

        if self.secondary_batch_size == 0:
            LOG.warning("TwoStreamBatchSampler: secondary_batch_size==0; sampling primary only.")

    def __iter__(self):
        primary_iter = _iterate_once(self.primary_indices)
        if self.secondary_batch_size > 0 and len(self.secondary_indices) > 0:
            secondary_iter = _iterate_eternally(self.secondary_indices)
            return (
                list(p) + list(s)
                for (p, s) in zip(
                    _grouper(primary_iter, self.primary_batch_size),
                    _grouper(secondary_iter, self.secondary_batch_size)
                )
            )
        else:
            return (list(p) for p in _grouper(primary_iter, self.primary_batch_size))

    def __len__(self):
        return max(1, len(self.primary_indices) // max(1, self.primary_batch_size))


def _iterate_once(indices): return np.random.permutation(indices)

def _iterate_eternally(indices):
    def infinite():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite())

def _grouper(iterable, n):
    iters = [iter(iterable)] * n
    return zip(*iters)


def _decode_any_mask(segmentation, H, W):
    if segmentation is None:
        return None

    # 1) Polygon list
    if isinstance(segmentation, list):
        try:
            rles = mask_utils.frPyObjects(segmentation, H, W)
            rle = mask_utils.merge(rles)
            m = mask_utils.decode(rle)
            return m.astype('uint8').squeeze()
        except Exception:
            return None

    # 2) Dict (RLE-like)
    if isinstance(segmentation, dict):
        counts = segmentation.get('counts', None)

        # 2a) COCO compressed RLE (string/bytes)
        if isinstance(counts, (str, bytes)):
            try:
                m = mask_utils.decode(segmentation)
                return m.astype('uint8').squeeze()
            except Exception:
                return None

        # 2b) counts as list: either uncompressed COCO runs or (start,len) pairs
        if isinstance(counts, list):
            try:
                total = H * W
                looks_like_pairs = (len(counts) % 2 == 0) and (len(counts) > 0 and max(counts) >= total // 4)

                if looks_like_pairs:
                    # Pairs (start,len), fill in Fortran order
                    m = np.zeros((H, W), dtype=np.uint8, order='F')
                    flatF = m.ravel(order='F')
                    it = iter(counts)
                    for s in it:
                        l = next(it, 0)
                        s = int(s); l = int(l)
                        if l <= 0:
                            continue
                        s = max(0, min(s, total))
                        e = max(s, min(s + l, total))
                        flatF[s:e] = 1
                    return m

                # Otherwise: treat as uncompressed COCO run-lengths
                rle = mask_utils.frPyObjects({'size': [H, W], 'counts': counts}, H, W)
                m = mask_utils.decode(rle)
                return m.astype('uint8').squeeze()

            except Exception:
                return None

    return None

class CocoCropDataset(Dataset):
    """
    entries: list of dicts:
      {img_path, bbox(xywh ints), H,W, seg (or None), ann_id}
    __getitem__ -> (views, mask, tag)
      - views: (weak,strong) tensors
      - mask : [1,H,W] if labeled else None
      - tag  : "<file>#ann<id>"
    """
    def __init__(self, entries, img_transform=None, mask_transform=None):
        self.entries = entries
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.copy_paste_aug = CopyPasteAug(prob=0.5, scale=(0.8, 1.2))

    def __len__(self): return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        img = Image.open(e['img_path']).convert('RGB')

        try:
            exif = img.getexif()  # returns a dict-like Exif object or empty
        except Exception:
            exif = None
        if exif:
            orientation = exif.get(0x0112, 1)  # 0x0112 = Orientation tag
            if orientation == 3:      # rotated 180
                img = img.rotate(180, expand=True)
            elif orientation == 6:    # rotated 270° CCW -> needs 270 to become upright
                img = img.rotate(270, expand=True)
            elif orientation == 8:    # rotated 90° CCW -> needs 90 to become upright
                img = img.rotate(90, expand=True)
                
        x, y, w, h = e['bbox']
        crop = img.crop((x, y, x + w, y + h))  # PIL uses left,top,right,bottom

        if crop.mode != "RGB":
            crop = crop.convert("RGB")

        mask_t = None
        if e.get('seg', None) is not None:
            mask_full = _decode_any_mask(e['seg'], e['H'], e['W'])
            if mask_full is not None:
                mask_crop = mask_full[y:y+h, x:x+w]
                if mask_crop.sum() > 0:
                    mask_img = Image.fromarray((mask_crop * 255).astype(np.uint8), mode='L')
                    if random.random() < 0.5:
                        crop, mask_img = self.copy_paste_aug(crop, mask_img)
                    mask_t = self.mask_transform(mask_img) if self.mask_transform else T.ToTensor()(mask_img)

        views = self.img_transform(crop) if self.img_transform else (T.ToTensor()(crop), T.ToTensor()(crop))
        tag = f"{os.path.basename(e['img_path'])}#ann{e['ann_id']}"
        return views, mask_t, tag


def _collect_entries_from_one_coco(images_dir, annotations_path):
    with open(annotations_path, 'r') as f:
        coco = json.load(f)

    # use height/width from JSON (avoid cv2.imread)
    id2info = {img['id']: (img['file_name'], img.get('height'), img.get('width'))
               for img in coco.get('images', [])}

    labeled, unlabeled = [], []

    for ann in tqdm(coco.get('annotations', []), desc=f"Indexing COCO: {os.path.basename(images_dir)}"):
        if 'bbox' not in ann:
            continue

        info = id2info.get(ann['image_id'])
        if not info:
            continue
        fn, H, W = info
        path = os.path.join(images_dir, fn)
        if not os.path.isfile(path) or H is None or W is None:
            # fallback (rare): skip or lazily read later if you must
            continue

        x, y, w, h = list(map(int, ann['bbox']))
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > W or y + h > H:
            continue

        e = {
            'img_path': path,
            'bbox': (x, y, w, h),
            'H': H, 'W': W,
            'ann_id': ann.get('id', -1),
            'seg': ann.get('segmentation', None)
        }

        # Light-touch labeling: trust presence of 'seg' as labeled; avoid decoding here
        if e['seg']:
            labeled.append(e)
        else:
            unlabeled.append(e)

    return labeled, unlabeled

def create_coco_datasets(pairs,size=256,mode="train",                 # "train", "eval"
    return_indices=False,         # return (labeled_idxs, unlabeled_idxs) when mode in {"train"}
    tqdm_bar=False
):
    """
    Unified builder for COCO crop datasets.

    mode="train" -> returns (train_dataset, [labeled_idxs, unlabeled_idxs]?)
    mode="eval"  -> returns eval_dataset (labeled-only)
    
    pairs: list[(images_dir, annotations_json)]
    """
    cfg = myforest_dataset()
    train_tf = cfg.get('train_transformation', None)   # (weak,strong) for mean-teacher
    eval_tf  = cfg.get('eval_transformation',  None)

    # ---- collect entries once ----
    all_labeled, all_unlabeled = [], []
    it = pairs
    if tqdm_bar:
        it = tqdm(pairs, desc="Collecting COCO pairs")

    for img_dir, ann_path in it:
        if not (os.path.isdir(img_dir) and os.path.isfile(ann_path)):
            LOG.info(f"[skip] {img_dir} (missing folder or annotation.json)")
            continue
        L, U = _collect_entries_from_one_coco(img_dir, ann_path)
        all_labeled.extend(L)
        all_unlabeled.extend(U)

    mask_tf = T.Compose([
        T.Resize((size, size), interpolation=InterpolationMode.NEAREST),
        T.ToTensor()
    ])

    if mode == "train":
        all_entries = all_labeled + all_unlabeled
        train_dataset = CocoCropDataset(all_entries, img_transform=train_tf, mask_transform=mask_tf)
        LOG.info(f"[data/train] labeled={len(all_labeled)} unlabeled={len(all_unlabeled)} total={len(all_entries)}")

        # Compute bbox areas and duplicate indices of small-object crops
        if len(all_labeled) > 0:
            areas = [e["bbox"][2] * e["bbox"][3] for e in all_labeled]
            median_area = np.median(areas)
            small_indices = [i for i, e in enumerate(all_labeled) if (e["bbox"][2] * e["bbox"][3]) < median_area]

            upsample_factor = 2   # duplicate small-object samples (tunable)
            weighted_labeled_idxs = labeled_idxs = list(range(len(all_labeled)))
            weighted_labeled_idxs += small_indices * (upsample_factor - 1)
            np.random.shuffle(weighted_labeled_idxs)

            LOG.info(f"[Sampler] Boosted {len(small_indices)} small-object samples "
                    f"(median area={median_area:.1f}, upsample×{upsample_factor})")
        else:
            weighted_labeled_idxs = []

        # Unlabeled indices remain the same
        unlabeled_idxs = list(range(len(all_labeled), len(all_labeled) + len(all_unlabeled)))

        if return_indices:
            labeled_idxs   = list(range(len(all_labeled)))
            unlabeled_idxs = list(range(len(all_labeled), len(all_labeled) + len(all_unlabeled)))
            return train_dataset, labeled_idxs, unlabeled_idxs
        return train_dataset

    elif mode == "eval":
        eval_dataset = CocoCropDataset(all_labeled, img_transform=eval_tf, mask_transform=mask_tf)
        LOG.info(f"[data/eval] labeled-only={len(all_labeled)}")
        return eval_dataset

    else:
        raise ValueError("mode must be one of {'train','eval'}")