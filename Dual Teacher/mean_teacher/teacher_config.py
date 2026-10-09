# Mask R-CNN with Swin-T backbone for Simulated Tree Trunk (4 folders)
import os.path as osp

# --- Paths ---
HOME = osp.expanduser("~")
DATA_ROOT = "/mnt/local_learning/data/pankaj/trees/"
WORK_DIR = osp.join(HOME, "implementations/mmdetection/work_dirs/mask_rcnn_simulated/whole_tree/Swin-T/1x_coco/")

# --- Model ---
model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=True,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
        )),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,  # only tree_trunk
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0, beta=0.1111111111111111)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    # --- Training and Testing Configs ---
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=4000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5))
)

# --- Image Normalization ---
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

# --- Pipelines ---
img_scale = (1280, 800)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=img_scale,
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img'])
         ])
]

# --- Dataset ---
classes = ('whole_tree',)
dataset_type = 'CocoDataset'

def coco_dataset(folder, split):
    return dict(
        type=dataset_type,
        ann_file=osp.join(DATA_ROOT, f"{folder}/whole_tree/{split}/coco_annotations_cleaned.json"),
        img_prefix=osp.join(DATA_ROOT, f"{folder}/whole_tree/{split}/images/"),
        classes=classes,
        pipeline=train_pipeline if split == 'train' else test_pipeline
    )

data = dict(
    samples_per_gpu=24,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
        type='ConcatDataset',
        separate_eval=True,
        datasets=[coco_dataset(i, 'train') for i in ['1', '2', '3', '4']]
    ),
    val=dict(
        type='ConcatDataset',
        separate_eval=True,
        datasets=[coco_dataset(i, 'val') for i in ['1', '2', '3', '4']]
    ),
    test=dict(
        type='ConcatDataset',
        separate_eval=True,
        datasets=[coco_dataset(i, 'test') for i in ['1', '2', '3', '4']]
    )
)

# --- Evaluation ---
evaluation = dict(
    metric=['bbox', 'segm'],
    interval=1,
    rule='greater'
)

# --- Optimizer & Schedule ---
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0)
        )
    )
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='step', warmup='linear', warmup_iters=1000, warmup_ratio=0.001, step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)

# --- Checkpoint ---
checkpoint_config = dict(
    interval=1,         # don't save every epoch
    max_keep_ckpts=3,    # only keep the best
    save_last=True       # also save last epoch
)

# --- Logging ---
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]

# --- Runtime Settings ---
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/pankaj/implementations/mmdetection/work_dirs/mask_rcnn_simulated/whole_tree/Swin-T/1x_coco/epoch_17.pth'
resume_from = '/home/pankaj/implementations/mmdetection/work_dirs/mask_rcnn_simulated/whole_tree/Swin-T/1x_coco/epoch_17.pth'
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
work_dir = WORK_DIR
auto_resume = True
gpu_ids = [0]

data['test'] = dict(pipeline=test_pipeline)