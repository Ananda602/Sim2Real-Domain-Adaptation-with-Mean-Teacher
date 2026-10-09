import os
import os.path as osp
import shutil
import time
import sys
import csv
import logging
from math import exp 
from datetime import datetime
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torch.amp import autocast, GradScaler  # modern AMP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from mean_teacher import architectures, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import TwoStreamBatchSampler, create_coco_datasets
from mean_teacher.visuals import dump_teacher_pseudomasks, dump_dual_teacher_pseudomasks, plot_from_csv,inspect_unlabeled_batch, _preds_to_semantic_prob
from mean_teacher.metrics import compute_ap_set, segmentation_metrics, format_eta, get_model_device
from mean_teacher.utils import *  # AverageMeterSet, parameters_string, etc.
LOG = logging.getLogger('main')

# AMP device type
AMP_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_config_file(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

args = None
best_iou = 0.0
global_step = 0
VIS_DIR = None  # set in main(context)

def setup_logging(run_dir="runs"):
    os.makedirs(run_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(run_dir, f"train_{timestamp}.log")

    # Log to both file and stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # --- NEW BLOCK: capture uncaught exceptions into the same log ---
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception
    # ---------------------------------------------------------------

    return log_path

def main(context):
    global global_step, best_iou, VIS_DIR
    global device 

    checkpoint_path = context.transient_dir
    training_log = context.create_train_log("training")
    validation_log = context.create_train_log("validation")
    ema_validation_log = context.create_train_log("ema_validation")

    device_id = getattr(args, 'gpu_id', 0)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)  # <--- force current device
    LOG.info(f"Using device: {device}")

    # where to save teacher pseudo-mask previews
    VIS_DIR = os.path.join(checkpoint_path, "pseudo_vis")
    os.makedirs(VIS_DIR, exist_ok=True)

    cudnn.benchmark = True

    # ----------------------------
    # DATA ROOTS & TASK LAYOUT
    # ----------------------------
    # Backward compatibility: allow legacy --data-root OR the new --real-root/--sim-root
    if (not hasattr(args, "real_root") or args.real_root is None) and (not hasattr(args, "data_root") or args.data_root is None):
        raise ValueError("Provide --real-root (preferred) or legacy --data-root.")

    REAL_ROOT = getattr(args, "real_root", None) or getattr(args, "data_root", None)
    SIM_ROOT  = getattr(args, "sim_root", None)  # optional

    # tasks (directory names under each part)
    REAL_TASK = getattr(args, "real_task", "tree")
    SIM_TASKS = getattr(args, "sim_tasks", None)
    if SIM_TASKS is None:
        # default simulated tasks
        SIM_TASKS = ["whole_tree", "tree_trunk"]
    elif isinstance(SIM_TASKS, str):
        SIM_TASKS = [t.strip() for t in SIM_TASKS.split(",") if t.strip()]

    train_split = getattr(args, "train_split", "train")
    eval_split  = getattr(args, "eval_split",  "val")  # NOTE: eval will be real-only below

    ann_file_map = {
    "train": ["annotation_partial_anand.json",
              "annotation_partial_sim.json"],
    "val":   ["coco_annotations_clean_for_anand.json",
              "coco_annotations_cleaned_for_anand.json"],
    "test":  ["coco_annotations_clean_for_anand.json",
              "coco_annotations_cleaned_for_anand.json"],
    }

    def _resolve_ann_path(base_dir, split, default="annotation_partial.json"):
        """Pick the first annotation file that actually exists."""
        candidates = ann_file_map.get(split, [default])
        for fname in candidates:
            path = os.path.join(base_dir, fname)
            if os.path.exists(path):
                return path
        # fallback: return last option even if missing
        return os.path.join(base_dir, candidates[-1])

    def _list_parts(root_dir):
        """List part folders (e.g., RPTU, …) under a root."""
        if not (root_dir and os.path.isdir(root_dir)):
            return []
        return sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])


    def _pairs_for(root_dir, task, split):
        """Generate all (images_dir, annotation_json) pairs for a given root/task/split across its parts."""
        pairs = []
        if not (root_dir and os.path.isdir(root_dir)):
            LOG.warning(f"Root not found: {root_dir}")
            return pairs

        parts = _list_parts(root_dir)
        candidates = ann_file_map.get(split, "annotation_partial.json")
        if isinstance(candidates, str):
            candidates = [candidates]

        for part in parts:
            img_dir = os.path.join(root_dir, part, task, split, "images")

            ann_js = None
            for fname in candidates:
                cand = os.path.join(root_dir, part, task, split, fname)
                if os.path.exists(cand):
                    ann_js = cand
                    break

            if img_dir and os.path.exists(img_dir) and ann_js and os.path.exists(ann_js):
                t0 = time.time()
                # ⚡ Lazy-load the COCO just to time JSON parsing
                _ = COCO(ann_js)
                elapsed = time.time() - t0
                LOG.info(f"Loading {ann_js} took {elapsed:.2f} seconds.")
                pairs.append((img_dir, ann_js))
            else:
                LOG.warning(f"Skipping missing dataset -> images: {img_dir}, anns: {candidates}")
        return pairs


    # ----------------------------
    # BUILD TRAIN / EVAL PAIRS
    # ----------------------------
    train_pairs = []
    eval_pairs  = []

    # Real-world TRAIN
    train_pairs.extend(_pairs_for(REAL_ROOT, REAL_TASK, train_split))

    # Simulated TRAIN (if provided)
    if SIM_ROOT:
        for t in SIM_TASKS:
            train_pairs.extend(_pairs_for(SIM_ROOT, t, train_split))

    # Validation = REAL-ONLY VAL
    eval_pairs.extend(_pairs_for(REAL_ROOT, REAL_TASK, eval_split))

    # --- Sanity peek ---
    def peek(pairs, name, k=3):
        print(f"[peek] {name}: {len(pairs)} pairs")
        for (img_dir, ann) in pairs[:k]:
            print("   ", img_dir, "|", ann)

    peek(train_pairs, "train_pairs")
    peek(eval_pairs,  "eval_pairs")

    # Ensure no overlap of image dirs between train and eval (should hold since eval is real-only val)
    train_dirs = set(d for d, _ in train_pairs)
    eval_dirs  = set(d for d, _ in eval_pairs)
    leak = train_dirs & eval_dirs
    if leak:
        print("[WARNING] overlap between train/eval image dirs:", leak)
    else:
        print("[OK] no overlap between train/eval image dirs")

    print("INFO: dataloader initiated")

    def count_images_per_part(pairs):
        counts = {}
        for img_dir, ann_file in pairs:
            try:
                part_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(img_dir))))
            except Exception:
                part_name = "UNKNOWN_PART"
            num_images = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            counts[part_name] = counts.get(part_name, 0) + num_images
        return counts

    # Report counts (per source)
    real_train_counts = count_images_per_part(_pairs_for(REAL_ROOT, REAL_TASK, train_split))
    real_val_counts   = count_images_per_part(_pairs_for(REAL_ROOT, REAL_TASK, eval_split))
    sim_train_counts  = {}
    if SIM_ROOT:
        for t in SIM_TASKS:
            c = count_images_per_part(_pairs_for(SIM_ROOT, t, train_split))
            # merge per-part across sim tasks for a concise summary
            for k, v in c.items():
                sim_train_counts[k] = sim_train_counts.get(k, 0) + v

    print("INFO: -------- Dataset Summary --------")
    for part in sorted(set(list(real_train_counts.keys()) + list(real_val_counts.keys()))):
        print(f"INFO: REAL {part} -> train: {real_train_counts.get(part, 0)} images, val: {real_val_counts.get(part, 0)} images")
    if SIM_ROOT:
        for part in sorted(sim_train_counts.keys()):
            print(f"INFO: SIM  {part} -> train: {sim_train_counts.get(part, 0)} images")
    print("INFO: ---------------------------------")


    # ------------------------------------------------------------
    # 1) Datasets & loaders
    # ------------------------------------------------------------
    train_dataset, labeled_idxs, unlabeled_idxs = create_coco_datasets(
    train_pairs, size=256, mode="train", return_indices=True)

    eval_dataset = create_coco_datasets(eval_pairs, size=256, mode="eval")


    os.makedirs("output", exist_ok=True)
    with open("output/labeled_indices.txt", "w") as f:
        for idx in labeled_idxs:
            f.write(str(train_dataset.entries[idx]['img_path']) + "\n")
    with open("output/unlabeled_indices.txt", "w") as f:
        for idx in unlabeled_idxs:
            f.write(str(train_dataset.entries[idx]['img_path']) + "\n")

    LOG.info(f"[splits] train_pairs={len(train_pairs)} | eval_pairs={len(eval_pairs)} "
         f"| train_split='{train_split}' | eval_split='{eval_split}'")

    LOG.info(f"Dataset sizes -> labeled: {len(labeled_idxs)} | unlabeled: {len(unlabeled_idxs)}")


    lbl_bs = int(getattr(args, "labeled_batch_size", max(1, int(args.batch_size * 0.25))))
    lbl_bs = min(lbl_bs, max(1, len(labeled_idxs)))
    unlbl_bs = args.batch_size - lbl_bs
    assert unlbl_bs > 0, "Batch must include at least one unlabeled sample."
    LOG.info(f"[TwoStream] batch_size={args.batch_size} -> unlabeled={unlbl_bs}, labeled={lbl_bs}")

    batch_sampler = TwoStreamBatchSampler(
        primary_indices=unlabeled_idxs,   # UNLABELED
        secondary_indices=labeled_idxs,   # LABELED
        batch_size=args.batch_size,
        secondary_batch_size=lbl_bs
    )

    def collate_fn(batch):
        views, masks, tags = zip(*batch)

        if isinstance(views[0], (tuple, list)) and len(views[0]) == 2:
            weak   = torch.stack([v[0] for v in views], dim=0)   # [B,3,H,W]
            strong = torch.stack([v[1] for v in views], dim=0)   # [B,3,H,W]
        else:
            imgs = torch.stack(list(views), dim=0)
            weak, strong = imgs, imgs

        H, W = weak.shape[-2], weak.shape[-1]

        mask_list, is_labeled = [], []
        for m in masks:
            if m is None:
                mask_list.append(torch.zeros(1, H, W))
                is_labeled.append(0)
            else:
                mask_list.append(m)
                is_labeled.append(1)
        masks_t = torch.stack(mask_list, dim=0)                 # [B,1,H,W]
        is_lbl_t = torch.tensor(is_labeled, dtype=torch.uint8)  # [B]
        return (weak, strong), (masks_t, is_lbl_t), tags

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=collate_fn
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(2, args.workers // 2),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False,
        collate_fn=collate_fn
    )

    # ------------------------------------------------------------
    # 2) Models
    # ------------------------------------------------------------
    model, ema_model, ema_model_b = create_models(
    arch_student=args.arch,
    arch_teacher=args.arch_teacher,
    num_classes=args.num_classes,
    teacher_kwargs={
        "cfg_path": args.teacher_cfg,
        "ckpt_path": args.teacher_ckpt_a,
        "ckpt_path_b": args.teacher_ckpt_b,
        "conf_thresh_a": getattr(args, "teacher_conf_a", 0.0),
        "conf_thresh_b": getattr(args, "teacher_conf_b", 0.0),
    }
    )

        # --- Teacher compatibility test ---
    try:
        _okA = check_teacher_compat(ema_model,   tag="Teacher A")
        _okB = check_teacher_compat(ema_model_b, tag="Teacher B")
        if not (_okA and _okB):
            LOG.warning("[Compat] One or both teachers did not pass the compatibility check. KD fusion may fail.")
    except Exception as _e:
        LOG.warning(f"[Compat] Teacher check could not run: {_e}")

    LOG.info(parameters_string(model))
    LOG.info("Teacher config path (cfg_path): %s", args.teacher_cfg)
    LOG.info("Teacher A checkpoint path (ckpt_path): %s ", args.teacher_ckpt_a)
    LOG.info("Teacher B checkpoint path (ckpt_path_b): %s", args.teacher_ckpt_b)

    # ------------------------------------------------------------
    # 3) Optimizer + AMP scaler (new API)
    # ------------------------------------------------------------

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim < 2 or n.endswith(".bias") or "bn" in n or "norm" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    wd = getattr(args, "weight_decay", 0.01)
    betas = tuple(getattr(args, "optimizer", {}).get("betas", [0.9, 0.999]))

    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=betas
    )

    # GradScaler (safe on CPU/GPU)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    # Cosine scheduler with 1-epoch warmup
    warmup_epochs = int(getattr(args, "warmup_epochs", 0))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=args.lr_min)

    # ------------------------------------------------------------
    # 4) Resume
    # ------------------------------------------------------------
    if getattr(args, 'resume', None):
        assert osp.isfile(args.resume), f"=> no checkpoint found at '{args.resume}'"
        LOG.info("=> loading checkpoint '%s'", args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')

        # bookkeeping
        args.start_epoch = int(checkpoint.get('epoch', 0))
        global_step      = int(checkpoint.get('global_step', 0))
        best_iou         = float(checkpoint.get('best_iou', 0.0))

        # -------- helper to normalize DP prefixes --------
        def _maybe_add_module(sd, want_module_prefix: bool):
            if not isinstance(sd, dict) or len(sd) == 0:
                return sd
            has_module = next(iter(sd)).startswith("module.")
            if want_module_prefix and not has_module:
                return {f"module.{k}": v for k, v in sd.items()}
            if not want_module_prefix and has_module:
                return {k[len("module."):]: v for k, v in sd.items()}
            return sd

        # -------- student (REQUIRED) --------
        student_sd = checkpoint.get('state_dict', checkpoint)  # tolerate raw state-dict
        want_module = next(iter(model.state_dict())).startswith("module.")
        student_sd = _maybe_add_module(student_sd, want_module)
        missing, unexpected = model.load_state_dict(student_sd, strict=False)
        if missing or unexpected:
            LOG.warning("Student load non-strict: missing=%d unexpected=%d", len(missing), len(unexpected))

        # -------- teacher (OPTIONAL) --------
        # If you're in OFFLINE DISTILLATION (teacher_ckpt given), DO NOT overwrite the frozen teacher.
        if not (getattr(args, "teacher_ckpt_a", None) or getattr(args, "teacher_ckpt_b", None)):
            ema_sd = checkpoint.get('ema_state_dict', None)
            if ema_sd is not None:
                want_module_t = next(iter(ema_model.state_dict())).startswith("module.")
                ema_sd = _maybe_add_module(ema_sd, want_module_t)
                try:
                    missing_t, unexpected_t = ema_model.load_state_dict(ema_sd, strict=False)
                    LOG.info("=> loaded Teacher A from resume [missing=%d unexpected=%d]",
                            len(missing_t), len(unexpected_t))
                except Exception as e:
                    LOG.warning("Resume: Teacher A present but could not be loaded (%s). Keeping current.", e)

            ema_sd_b = checkpoint.get('ema_state_dict_b', None)
            if ema_sd_b is not None:
                want_module_t_b = next(iter(ema_model_b.state_dict())).startswith("module.")
                ema_sd_b = _maybe_add_module(ema_sd_b, want_module_t_b)
                try:
                    missing_tb, unexpected_tb = ema_model_b.load_state_dict(ema_sd_b, strict=False)
                    LOG.info("=> loaded Teacher B from resume [missing=%d unexpected=%d]",
                            len(missing_tb), len(unexpected_tb))
                except Exception as e:
                    LOG.warning("Resume: Teacher B present but could not be loaded (%s). Keeping current.", e)
            else:
                LOG.info("=> no ema_state_dict_b in checkpoint; keeping current Teacher B.")
        else:
            LOG.info("=> offline distill active (teacher_ckpt set); skipping teacher restore from resume.")


        # -------- optimizer  --------
        if 'optimizer' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception as e:
                LOG.warning("Could not load optimizer state: %s", e)

        # -------- AMP scaler  --------
        if 'scaler' in checkpoint:
            try:
                scaler.load_state_dict(checkpoint['scaler'])
            except Exception as e:
                LOG.warning("Could not load AMP scaler state: %s", e)

        LOG.info("=> loaded checkpoint '%s' (epoch %d, global_step %d, best_iou %.4f)",
                args.resume, args.start_epoch, global_step, best_iou)


        if getattr(args, 'evaluate', False):
            LOG.info("Evaluating the Student model:")
            validate(eval_loader, model, validation_log, global_step, args.start_epoch, scaler=None)

            if getattr(args, "teacher_ckpt_a", None):
                LOG.info("Evaluating Teacher A (frozen):")
                validate(eval_loader, ema_model, ema_validation_log, global_step, args.start_epoch, scaler=None)

            if getattr(args, "teacher_ckpt_b", None):
                # make a second log for teacher B, or reuse ema_validation_log if you don't need a separate CSV
                ema_validation_log_b = context.create_train_log("ema_validation_b")
                LOG.info("Evaluating Teacher B (frozen):")
                validate(eval_loader, ema_model_b, ema_validation_log_b, global_step, args.start_epoch, scaler=None)
            return

    primary_csv = osp.join(checkpoint_path, 'primary_metrics.csv')

    def append_metrics_csv(csv_path, epoch_num, iou, dice, ap50=None, ap75=None, map95=None):
        first = not osp.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            if first:
                w.writerow(['epoch', 'iou', 'dice', 'mAP@50', 'mAP@75', 'mAP@[.5:.95]'])
            w.writerow([epoch_num, iou, dice, ap50, ap75, map95])


    # ------------------------------------------------------------
    # 5) Train
    # ------------------------------------------------------------
    for epoch in range(args.start_epoch, args.epochs):
        t0 = time.time()
        train_one_epoch(train_loader, model, ema_model, ema_model_b, optimizer, epoch, training_log, scaler)
        LOG.info("--- training epoch in %.1fs ---", time.time() - t0)

        if warmup_epochs > 0 and epoch < warmup_epochs:
            # linear warmup: start from args.lr_min up to args.lr
            lr_start = getattr(args, "lr_min", 1e-5)
            lr_target = getattr(args, "lr", 1e-4)
            lr = lr_start + (lr_target - lr_start) * (epoch / warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        else:
            scheduler.step()

        is_best = False
        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            t1 = time.time()
            LOG.info("Evaluating the Student model:")
            iou, dice, ap_stats = validate(eval_loader, model, validation_log, global_step, epoch + 1, scaler=None)
            append_metrics_csv(primary_csv, epoch + 1, iou, dice,
                   ap_stats['mAP@50'], ap_stats['mAP@75'], ap_stats['mAP@[.5:.95]'])

            # quick unlabeled inspection (teacher vs student)
            try:
                inspect_out = os.path.join(checkpoint_path, "inspect_unlabeled")
                inspect_unlabeled_batch(train_loader, model, ema_model, inspect_out,
                                        max_images=20, thresh=args.conf_threshold)
            except Exception as e:
                LOG.warning(f"inspect_unlabeled_batch failed: {e}")

            LOG.info("--- validation in %.1fs ---", time.time() - t1)
            is_best = iou > best_iou
            best_iou = max(iou, best_iou)

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                #'ema_state_dict': ema_model.state_dict(),
                'best_iou': best_iou,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint_path, epoch + 1)

def create_models(arch_student: str,arch_teacher: str,num_classes: int,teacher_kwargs: dict | None = None,):
    teacher_kwargs = teacher_kwargs or {}
    LOG.info("=> creating student (%s) + teachers (%s)", arch_student, arch_teacher)

    # --- Student factory ---
    try:
        factory_s = architectures.__dict__[arch_student]
    except KeyError:
        LOG.warning("Student arch '%s' not found; fallback to 'maskrcnn'", arch_student)
        factory_s = architectures.__dict__["maskrcnn"]

    # --- Teacher factory ---
    try:
        factory_t = architectures.__dict__[arch_teacher]
    except KeyError:
        LOG.warning("Teacher arch '%s' not found; fallback to 'maskrcnn'", arch_teacher)
        factory_t = architectures.__dict__["maskrcnn"]

    #Student
    model = factory_s(num_classes=num_classes, in_channels=3)

    # Teacher A
    ema_model = factory_t(
        num_classes=num_classes, in_channels=3,
        cfg_path=teacher_kwargs.get("cfg_path"),
        ckpt_path=teacher_kwargs.get("ckpt_path"),
        conf_thresh=teacher_kwargs.get("conf_thresh_a", 0.0)
    )

    # Teacher B
    ema_model_b = factory_t(
        num_classes=num_classes, in_channels=3,
        cfg_path=teacher_kwargs.get("cfg_path"),
        ckpt_path=teacher_kwargs.get("ckpt_path_b"),
        conf_thresh=teacher_kwargs.get("conf_thresh_b", 0.0)
    )

    model       = model.to(device)
    ema_model   = ema_model.to(device)
    ema_model_b = ema_model_b.to(device)

    for p in ema_model.parameters(): p.requires_grad_(False)
    for p in ema_model_b.parameters(): p.requires_grad_(False)
    ema_model.eval(); ema_model_b.eval()


    # --- Describe models with backbone ---
    def describe_model(detector, tag):
        m = detector.module if hasattr(detector, "module") else detector
        bb = getattr(m, "backbone", None)
        bb_name = type(bb).__name__ if bb is not None else "UnknownBackbone"
        LOG.info("=> %s built: %s with backbone %s", tag, type(m).__name__, bb_name)

    describe_model(model, "Student")
    describe_model(ema_model, "Teacher A")
    describe_model(ema_model_b, "Teacher B")

    return model, ema_model, ema_model_b

def check_teacher_compat(teacher, tag="Teacher", device=None):
    """
    Run a tiny forward pass and log shapes to ensure this MMDet teacher
    produces 'masks' [N,1,H,W] and 'scores' [N] as expected.
    Returns True if shapes look OK, else False.
    """
    try:
        teacher.eval()
        try:
            dev = device or next(teacher.parameters()).device
        except StopIteration:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Single RGB image (random)
        from PIL import Image
        import torchvision.transforms as T

        test_img = Image.open("/home/jsathyan/noadsm/trees/1/tree_trunk/train/images/Demo.0423.jpg").convert("RGB")
        transform = T.Compose([T.Resize((256,256)), T.ToTensor()])
        img = transform(test_img).to(dev)


        with torch.no_grad():
            outs = teacher([img])

        ok = isinstance(outs, (list, tuple)) and len(outs) > 0 and isinstance(outs[0], dict)
        if not ok:
            LOG.warning(f"[Compat] {tag}: unexpected output type {type(outs)}")
            return False

        masks  = outs[0].get("masks", None)
        scores = outs[0].get("scores", None)
        if masks is None or scores is None:
            LOG.warning(f"[Compat] {tag}: no 'masks' or 'scores' in output keys={list(outs[0].keys())}")
            return False

        if not (masks.ndim == 4 and masks.shape[1] == 1):
            LOG.warning(f"[Compat] {tag}: masks shape {tuple(masks.shape)} (expected [N,1,H,W])")
            return False

        LOG.info(f"[Compat] {tag}: masks {tuple(masks.shape)}, scores {tuple(scores.shape)}, "
                 f"device={masks.device}, dtype={masks.dtype}")
        return True

    except Exception as e:
        LOG.error(f"[Compat] {tag} forward failed: {e}")
        return False

# --- helpers  ------------------------------------------
def _check_imgs_list(tag, imgs, device=None):
    assert isinstance(imgs, (list, tuple)) and len(imgs) > 0, f"{tag}: need non-empty list"
    C, H, W = None, None, None
    for t in imgs:
        assert torch.is_tensor(t) and t.ndim == 3, f"{tag}: expect tensor [C,H,W]"
        assert t.size(0) in (1, 3), f"{tag}: channels {t.size(0)} not in {1,3}"
        if device is not None:
            assert t.device == device, f"{tag}: on {t.device}, expected {device}"
        if C is None: C, H, W = t.shape
        else: assert t.shape == (C, H, W), f"{tag}: shape mismatch {t.shape} vs {(C,H,W)}"
        assert torch.isfinite(t).all(), f"{tag}: NaN/Inf in image"
    return C, H, W

def _check_prob_map(tag, P, B, H, W):
    assert torch.is_tensor(P) and P.ndim == 4, f"{tag}: expect [B,1,H,W]"
    assert P.shape[0] == B and P.shape[1] == 1 and P.shape[2] == H and P.shape[3] == W, \
        f"{tag}: got {list(P.shape)}, want [B,1,{H},{W}]"
    assert torch.isfinite(P).all(), f"{tag}: NaN/Inf in {tag}"
    m, M = float(P.min()), float(P.max())
    assert m >= -1e-3 and M <= 1.0 + 1e-3, f"{tag}: not probs in [0,1], min={m} max={M}"

def _student_predict_no_targets(model, images, amp_device):
    """Temporarily use eval() to get detections (no targets) while keeping grads."""
    was_training = model.training
    model.eval()
    with autocast(device_type=AMP_DEVICE, enabled=(amp_device is not None)):
        preds = model(images)  # list[dict]
    if was_training:
        model.train()
    return preds
# ------------------------------------------------------------------------------

def dilate(prob, k=5):
    # morphological dilation via max-pool on probability map
    return torch.nn.functional.max_pool2d(prob, kernel_size=k, stride=1, padding=k//2)

def fuse_granularity(P_whole, P_trunk, rule_tree="hier"):
    # Tree probability (student space)
    if rule_tree == "hier":
        # trunk only trusted inside whole; else, use whole
        P_tree = torch.maximum(P_whole, P_trunk * (P_whole > 0.5).float())
    elif rule_tree == "noisy_or":
        P_tree = 1.0 - (1.0 - P_whole) * (1.0 - P_trunk)
    else:  # "max"
        P_tree = torch.maximum(P_whole, P_trunk)

    # Crown probability (for edge/boundary sharpening aux)
    P_crown = torch.clamp(P_whole - dilate(P_trunk, k=5), min=0.0)
    return P_tree, P_crown


def train_one_epoch(train_loader, model, ema_model, ema_model_b, optimizer, epoch, log, scaler: GradScaler):
    global global_step, VIS_DIR

    cons_base = float(getattr(args, 'consistency', 0.0) or 0.0)
    warmup_epochs = int(getattr(args, 'warmup_epochs', 5))

    meters = AverageMeterSet()
    model.train()
    ema_model.eval()    # Teacher A: whole tree (frozen)
    ema_model_b.eval()  # Teacher B: trunk (frozen)

    end = time.time()
    global_batch_times = []
    for i, batch in enumerate(train_loader):
        meters.update('data_time', time.time() - end)

        #adjust_learning_rate(optimizer, epoch, i, len(train_loader))
        meters.update('lr', optimizer.param_groups[0]['lr'])

        # batch = ((weak,strong), (mask, is_labeled), paths)
        (x_w, x_s), (y_mask, is_labeled), paths = batch
        x_w = x_w.to(device, non_blocking=True)
        x_s = x_s.to(device, non_blocking=True)
        y_mask = y_mask.to(device, non_blocking=True).float()
        is_labeled = is_labeled.to(device, non_blocking=True).bool()
        H, W = x_s.shape[-2:]

        # -------------------------
        # Supervised (labeled-strong)
        # -------------------------
        loss_sup  = torch.zeros((), device=x_s.device)
        loss_cons = torch.zeros((), device=x_s.device)

        if is_labeled.any():
            idx_l = torch.nonzero(is_labeled, as_tuple=False).squeeze(1).tolist()
            imgs_l = [x_s[j] for j in idx_l]
            targs = []
            for j in idx_l:
                m = (y_mask[j] > 0.5).float()  # [1,H,W]
                boxes  = torch.tensor([[0.0, 0.0, float(W), float(H)]],
                                      device=x_s.device, dtype=torch.float32)
                labels = torch.tensor([1], device=x_s.device, dtype=torch.int64)
                targs.append({"boxes": boxes, "labels": labels, "masks": m})

            with autocast(device_type=AMP_DEVICE, enabled=False):
                loss_dict = model(imgs_l, targs)  # train path (needs targets)
                # Emphasize mask accuracy (+ optional hard mining)
                if "loss_mask" in loss_dict:
                    if loss_dict["loss_mask"].ndim == 0:
                        loss_dict["loss_mask"] *= 1.2  # emphasize segmentation
                    else:
                        k = max(1, int(0.25 * loss_dict["loss_mask"].numel()))  # top 25% hardest
                        hard_mask_loss, _ = torch.topk(loss_dict["loss_mask"].flatten(), k)
                        loss_dict["loss_mask"] = 1.2 * hard_mask_loss.mean()

                    print_once = getattr(args, "_mask_weight_logged", False)
                    if not print_once:
                        LOG.info("[Mask Loss] Weighted ×1.2 (+ OHEM top-25%)")
                        args._mask_weight_logged = True

                for k, v in loss_dict.items():
                    meters.update(k, float(v))
                loss_sup = sum(loss_dict.values())

        meters.update('seg_loss', float(loss_sup))

        # ----------------------------------------
        # Unlabeled (AFTER warm-up): granularity-aware KD
        # ----------------------------------------
        meters.update('cons_weight', 0.0); meters.update('cons_loss', 0.0)
        if (epoch >= warmup_epochs) and (~is_labeled).any() and (args.kd_weight > 0):
            idx_u = torch.nonzero(~is_labeled, as_tuple=False).squeeze(1).tolist()
            imgs_w_u = [x_w[j] for j in idx_u]   # teachers on WEAK
            imgs_s_u = [x_s[j] for j in idx_u]   # student on STRONG
            B_u = len(imgs_w_u)
            assert B_u == len(imgs_s_u), "weak/strong count mismatch for unlabeled"

            dev = imgs_w_u[0].device
            _, Hw, Ww = _check_imgs_list("imgs_w_u", imgs_w_u, device=dev)
            _ = _check_imgs_list("imgs_s_u", imgs_s_u, device=dev)
            H_t, W_t = Hw, Ww  # use WEAK resolution for teachers

            # (1) TEACHERS: inference only
            with torch.no_grad(), autocast(AMP_DEVICE):
                out_a = ema_model(imgs_w_u)    # whole tree teacher
                out_b = ema_model_b(imgs_w_u)  # trunk teacher
            assert isinstance(out_a, list) and isinstance(out_b, list), "teachers must return list[dict]"

            # (2) soft prob maps per teacher (instance→semantic)
            P_whole = _preds_to_semantic_prob(out_a, H_t, W_t)
            P_trunk = _preds_to_semantic_prob(out_b, H_t, W_t)
            _check_prob_map("P_whole", P_whole, B_u, H_t, W_t)
            _check_prob_map("P_trunk", P_trunk, B_u, H_t, W_t)

            # (3) granularity-aware fusion → P_tree (for KD) and P_crown
            P_tree, P_crown = fuse_granularity(
                P_whole, P_trunk, rule_tree=getattr(args, "fuse_rule", "hier")
            )
            _check_prob_map("P_tree", P_tree, B_u, H_t, W_t)
            _check_prob_map("P_crown", P_crown, B_u, H_t, W_t)

            # (4) Cross-view agreement (IoU between weak teacher and strong student augmentations)
            with torch.no_grad(), autocast(AMP_DEVICE):
                out_s_strong1 = _student_predict_no_targets(model, imgs_s_u, AMP_DEVICE)
                out_s_strong2 = _student_predict_no_targets(model, imgs_s_u, AMP_DEVICE)

            P_strong1 = _preds_to_semantic_prob(out_s_strong1, H_t, W_t)
            P_strong2 = _preds_to_semantic_prob(out_s_strong2, H_t, W_t)

            # Compute IoU agreement mask
            inter = (P_strong1 * P_strong2).sum(dim=[1,2,3])
            union = (P_strong1 + P_strong2 - P_strong1 * P_strong2).sum(dim=[1,2,3]) + 1e-6
            iou_agree = (inter / union).view(-1, 1, 1, 1)

            # Keep only pseudo-labels that agree across strong augmentations
            agree_mask = (iou_agree >= 0.5).float()

            # Refine teacher pseudo-probabilities using agreement mask
            P_tree = P_tree * agree_mask + (1 - agree_mask) * 0.0

            LOG.debug(f"[CrossView] epoch={epoch} IoU_agree={iou_agree.mean().item():.3f}")


            # ---- periodic visual dumps (keeps your original fused dump too) ----
            if ((epoch + 1) % 5 == 0) and (i == 0):
                # keep your original fused pseudo dump: expects logits
                eps_v = 1e-6
                logit_t_vis = torch.log(P_tree.clamp(eps_v,1-eps_v)) - torch.log((1-P_tree).clamp(eps_v,1-eps_v))
                dump_teacher_pseudomasks(
                    torch.stack(imgs_w_u, dim=0), logit_t_vis,
                    paths=[paths[j] for j in idx_u],
                    out_dir=VIS_DIR, step=global_step, max_samples=4,
                    thresh=getattr(args, 'conf_threshold', 0.9)
                )
                # optional: if you added the richer visualizer, dump all maps
                try:
                    dump_dual_teacher_pseudomasks(
                        torch.stack(imgs_w_u, dim=0),
                        maps={"whole": P_whole, "trunk": P_trunk, "fused_tree": P_tree, "crown": P_crown},
                        paths=[paths[j] for j in idx_u],
                        out_dir=os.path.join(VIS_DIR, "dual"),
                        step=global_step, max_samples=4,
                        thresh=getattr(args, 'conf_threshold', 0.5)
                    )
                except NameError:
                    pass

            # (4) confidence & disagreement masks
            if epoch < 25:
                tau = 0.6  # stricter early
            else:
                tau = 0.4  # relaxed later
            conf   = P_tree
            keep_f = (conf >= tau).float()
            num    = keep_f.sum().clamp(min=1.0)

            LOG.debug(f"[ConfFilter] epoch={epoch} τ={tau:.2f}, valid_px={num.item():.0f}")

            if int(num.item()) == 0:
                meters.update('cons_weight', 0.0)
                meters.update('cons_loss', 0.0)
            else:
                # (5) STUDENT predictions on STRONG view (no targets, keep grads)
                out_s = _student_predict_no_targets(model, imgs_s_u, AMP_DEVICE)
                P_s   = _preds_to_semantic_prob(out_s, H_t, W_t).clamp(1e-4, 1-1e-4)
                _check_prob_map("P_s", P_s, B_u, H_t, W_t)

                # Convert probabilities to logits safely
                def _safe_logit(p, eps=1e-6):
                    p = p.clamp(eps, 1 - eps)
                    return (p / (1 - p)).log()

                # (6) KD with temperature on LOGITS (not probs)
                T = float(getattr(args, "kd_temp", 2.0))

                # teacher: P_tree -> logit, soften, back to prob
                logit_t = _safe_logit(P_tree)
                Pt = torch.sigmoid(logit_t / T).clamp(1e-4, 1-1e-4)

                # student: P_s -> logit, soften, back to prob
                logit_s = _safe_logit(P_s)
                Ps = torch.sigmoid(logit_s / T).clamp(1e-4, 1-1e-4)

                # Bernoulli KL: KL(Pt || Ps)
                kl = Pt * (Pt.log() - Ps.log()) + (1 - Pt) * ((1 - Pt).log() - (1 - Ps).log())
                kd_loss = (kl * keep_f * conf).sum() / num

                # Check for numerical stability before applying weight
                if not torch.isfinite(kd_loss):
                    LOG.error("NaN/Inf in kd_loss at epoch %d, step %d", epoch, global_step)
                    LOG.error("P_tree min %.4f max %.4f | P_s min %.4f max %.4f",
                            float(P_tree.min()), float(P_tree.max()),
                            float(P_s.min()), float(P_s.max()))
                    raise ValueError("NaN in KD loss")

                rampup = max(1, int(getattr(args, "kd_rampup", 1)))
                ramp = min(1.0, epoch / float(rampup))
                kd_weight_now = float(args.kd_weight) * ramp
                loss_cons = kd_weight_now * kd_loss
                LOG.debug(f"[KD Ramp] epoch={epoch} ramp={ramp:.3f} kd_weight={kd_weight_now:.3f}")

                # (7) optional boundary sharpening from crown
                edge_w = float(getattr(args, "edge_weight", 0.1))
                if edge_w > 0:
                    edge_target = (P_crown > 0.2).float()
                    edge_keep   = (keep_f * (edge_target > 0)).float()
                    n_edge      = edge_keep.sum().clamp(min=1.0)
                    edge_loss   = ((1.0 - P_s) * edge_keep).sum() / n_edge
                    loss_cons  += edge_w * edge_loss
                    meters.update('edge_loss', float(edge_loss))

                meters.update('cons_weight', kd_weight_now)
                meters.update('cons_loss', float(loss_cons))

        # curriculum
        if epoch < warmup_epochs:
            loss = loss_sup
        else:
            loss = loss_sup + loss_cons
        meters.update('total_loss', float(loss))

        # ---- Guard: skip backward when neither term has a graph / or non-finite ----

        if (not (loss_sup.requires_grad or loss_cons.requires_grad)) or (not torch.isfinite(loss)):
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            meters.update('batch_time', time.time() - end)
            global_batch_times.append(meters['batch_time'].val if 'batch_time' in meters else (time.time() - end))


            end = time.time()
            if i % args.print_freq == 0:
                warmup_flag = 1 if (epoch < warmup_epochs) else 0
                eta_str = format_eta(epoch, i, train_loader, args.epochs, global_batch_times)

                LOG.info(
                'Epoch: [%d][%d/%d] (warmup=%d) Time %.3f Data %.3f Sup %.4f Cons %.4f (w=%.3f) '
                'Total %.4f IoU %.3f Dice %.3f ETA %s',
                epoch, i, len(train_loader), warmup_flag,
                meters['batch_time'].avg, meters['data_time'].avg,
                meters['seg_loss'].avg, meters['cons_loss'].avg, meters['cons_weight'].avg,
                meters['total_loss'].avg, meters['iou'].avg, meters['dice'].avg,
                eta_str
            )

                log.record(epoch + i / len(train_loader), {
                    'step': global_step, 'warmup': warmup_flag,
                    **meters.values(), **meters.averages(), **meters.sums()
                })
            continue


        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # unscale and clip
        scaler.unscale_(optimizer)
        clip_val = getattr(args.optimizer, "clip_grad_norm", 0.0) if hasattr(args, "optimizer") else 0.0
        if clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)

        # step
        scaler.step(optimizer)
        scaler.update()

        global_step += 1

        # quick labeled metrics (semanticized predictions on labeled STRONG)
        if is_labeled.any():
            idx_l = torch.nonzero(is_labeled, as_tuple=False).squeeze(1).tolist()
            imgs_l = [x_s[j] for j in idx_l]

            was_training = model.training
            model.eval()
            with torch.no_grad(), autocast(AMP_DEVICE):
                outs = model(imgs_l)  # inference without targets
            if was_training:
                model.train()

            probs = _preds_to_semantic_prob(outs, H, W)
            eps = 1e-6
            logits = torch.log(probs.clamp(eps, 1-eps)) - torch.log((1 - probs).clamp(eps, 1-eps))
            iou, dice = segmentation_metrics(logits, y_mask[is_labeled])
            meters.update('iou', iou, int(is_labeled.sum().item()))
            meters.update('dice', dice, int(is_labeled.sum().item()))

        meters.update('batch_time', time.time() - end)
        end = time.time()

        # --- your original warmup-flag logging block (kept) ---
        if i % args.print_freq == 0:
            warmup_flag = 1 if (epoch < warmup_epochs) else 0
            eta_str = format_eta(epoch, i, train_loader, args.epochs, global_batch_times)

            LOG.info(
                'Epoch: [%d][%d/%d] (warmup=%d) Time %.3f Data %.3f Sup %.4f Cons %.4f (w=%.3f) '
                'Total %.4f IoU %.3f Dice %.3f ETA %s',
                epoch, i, len(train_loader), warmup_flag,
                meters['batch_time'].avg, meters['data_time'].avg,
                meters['seg_loss'].avg, meters['cons_loss'].avg, meters['cons_weight'].avg,
                meters['total_loss'].avg, meters['iou'].avg, meters['dice'].avg,
                eta_str
            )

            log.record(epoch + i / len(train_loader), {
                'step': global_step, 'warmup': warmup_flag,
                **meters.values(), **meters.averages(), **meters.sums()
            })

def validate(eval_loader, model, log, global_step, epoch, scaler=None):
    """
    For Mask R-CNN, we run inference, convert to semantic logits, then
    compute IoU/Dice/AP for logging (supervised loss here is BCE+Dice on logits, only for reporting).
    """
    sup_criterion = losses.SegmentationLoss().to(device)
    meters = AverageMeterSet()
    model.eval()

    all_ious, all_scores = [], []
    end = time.time()
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            (x_w, _x_s), (y_mask, _is_lbl), _paths = batch
            x = x_w.to(device, non_blocking=True)
            y = y_mask.to(device, non_blocking=True).float()

            H, W = x.shape[-2:]

            # predictions list (eval mode, no targets)
            with autocast(AMP_DEVICE):
                outs = model([x[j] for j in range(x.size(0))])

            probs = _preds_to_semantic_prob(outs, H, W)  # [B,1,H,W]
            logits = torch.log(probs.clamp(1e-6, 1-1e-6)) - torch.log((1-probs).clamp(1e-6, 1-1e-6))

            with autocast(AMP_DEVICE):
                seg_loss = sup_criterion(logits, y)

            iou, dice = segmentation_metrics(logits, y)

            # confidence proxy: max prob
            probs_flat = probs.view(probs.size(0), -1)
            maxp = probs_flat.max(dim=1).values.detach().cpu().numpy().tolist()

            # per-sample IoU for AP calculation
            B = probs.size(0)
            preds = (probs > 0.5).float()
            inter = (preds * (y > 0.5)).view(B, -1).sum(dim=1)
            union = (preds.view(B, -1).sum(dim=1) + (y>0.5).view(B, -1).sum(dim=1) - inter).clamp(min=1e-6)
            ious = ((inter + 1e-6) / union).detach().cpu().numpy().tolist()
            all_ious.extend(ious)
            all_scores.extend(maxp)

            meters.update('data_time', time.time() - end)
            meters.update('seg_loss', float(seg_loss), x.size(0))
            meters.update('iou', iou, x.size(0))
            meters.update('dice', dice, x.size(0))
            meters.update('batch_time', time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                LOG.info('Test: [%d/%d] Time %.3f Data %.3f Seg %.4f IoU %.3f Dice %.3f',
                         i, len(eval_loader),
                         meters['batch_time'].avg, meters['data_time'].avg,
                         meters['seg_loss'].avg, meters['iou'].avg, meters['dice'].avg)

    ap_stats = compute_ap_set(all_ious, all_scores)
    LOG.info(" * IoU %.3f  Dice %.3f  mAP@50 %.3f  mAP@75 %.3f  mAP@[.5:.95] %.3f",
         meters['iou'].avg, meters['dice'].avg,
         ap_stats['mAP@50'], ap_stats['mAP@75'], ap_stats['mAP@[.5:.95]'])

    log.record(epoch, {
    'step': global_step,
    **meters.values(), **meters.averages(), **meters.sums(),
    'mAP@50': ap_stats['mAP@50'],
    'mAP@75': ap_stats['mAP@75'],   
    'mAP@[.5:.95]': ap_stats['mAP@[.5:.95]'],
})

    return meters['iou'].avg, meters['dice'].avg, ap_stats

def save_checkpoint(state, is_best, dirpath, epoch):
    filename = f'checkpoint.{epoch}.ckpt'
    path = osp.join(dirpath, filename)
    best_path = osp.join(dirpath, 'best.ckpt')
    torch.save(state, path)
    LOG.info("--- checkpoint saved to %s ---", path)
    if is_best:
        shutil.copyfile(path, best_path)
        LOG.info("--- checkpoint copied to %s ---", best_path)

if __name__ == '__main__':
    log_file = setup_logging(run_dir="logs")   # save under ./logs/
    print(f"Logging to {log_file}") 
    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()
    def print_config(args):
        """
        Pretty-print all configs after merging CLI, YAML, and defaults.
        """
        print("\n========== Loaded Configuration ==========")
        for k, v in sorted(vars(args).items()):
            print(f"{k:25}: {v}")
        print("==========================================\n")


    if getattr(args, "config", None):   # pass --config config.yaml
        cfg = load_config_file(args.config)
        for k, v in cfg.items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)
    visible_gpus = list(range(torch.cuda.device_count()))
    LOG.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')} "
             f"-> Using {len(visible_gpus)} GPUs: {visible_gpus}")

    # optional: save it into args for later
    args.visible_gpus = visible_gpus

    print_config(args)
    main(RunContext(__file__, 0))