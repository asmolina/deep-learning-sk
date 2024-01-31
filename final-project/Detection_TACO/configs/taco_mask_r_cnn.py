_base_ = [
    './_base_/models/mask_rcnn_swin_fpn.py',
    './_base_/datasets/coco_instance.py',
    './_base_/schedules/schedule_1x.py', './_base_/default_runtime.py'
]
# 1. dataset settings
dataset_type = 'CocoDataset'
classes = ('Aluminium foil',
           'Battery',
           'Aluminium blister pack',
           'Carded blister pack',
           'Other plastic bottle',
           'Clear plastic bottle',
           'Glass bottle',
           'Plastic bottle cap',
           'Metal bottle cap',
           'Broken glass',
           'Food Can',
           'Aerosol',
           'Drink can',
           'Toilet tube',
           'Other carton',
           'Egg carton',
           'Drink carton',
           'Corrugated carton',
           'Meal carton',
           'Pizza box',
           'Paper cup',
           'Disposable plastic cup',
           'Foam cup',
           'Glass cup',
           'Other plastic cup',
           'Food waste',
           'Glass jar',
           'Plastic lid',
           'Metal lid',
           'Other plastic',
           'Magazine paper',
           'Tissues',
           'Wrapping paper',
           'Normal paper',
           'Paper bag',
           'Plastified paper bag',
           'Plastic film',
           'Six pack rings',
           'Garbage bag',
           'Other plastic wrapper',
           'Single-use carrier bag',
           'Polypropylene bag',
           'Crisp packet',
           'Spread tub',
           'Tupperware',
           'Disposable food container',
           'Foam food container',
           'Other plastic container',
           'Plastic glooves',
           'Plastic utensils',
           'Pop tab',
           'Rope & strings',
           'Scrap metal',
           'Shoe',
           'Squeezable tube',
           'Plastic straw',
           'Paper straw',
           'Styrofoam piece',
           'Unlabeled litter',
           'Cigarette')


model = dict(
    type='MaskRCNN',
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False),
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
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
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
            num_classes=60, ###################################################################
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
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
            num_classes=60, ###################################################################
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    # model training and testing settings
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
            nms_pre=2000,
            max_per_img=1000,
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
            mask_thr_binary=0.5)))



img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
######## added
    dict(type='Resize',
         img_scale=[(800, 1333)],
         multiscale_mode='value',
         keep_ratio=True),
######## added
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]



data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        pipeline=train_pipeline,
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/content/mmdetection/TACO-master/data/annotations_0_train.json',
        img_prefix='/content/mmdetection/TACO-master/data'),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/content/mmdetection/TACO-master/data/annotations_0_val.json',
        img_prefix='/content/mmdetection/TACO-master/data'),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/content/mmdetection/TACO-master/data/annotations_0_test.json',
        img_prefix='/content/mmdetection/TACO-master/data'))

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[8, 11])

runner = dict(type='EpochBasedRunner', max_epochs=12)


# runner = dict(type='EpochBasedRunnerAmp', max_epochs=12)

# # do not use mmdet version fp16
# fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )