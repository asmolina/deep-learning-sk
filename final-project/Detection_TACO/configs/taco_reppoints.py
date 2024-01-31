_base_ = [
    # './_base_/models/mask_rcnn_swin_fpn.py',
    './_base_/datasets/coco_detection.py',
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
    type='RepPointsDetector',#'MaskRCNN',
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
        num_outs=5,
        start_level=1,
        add_extra_convs='on_input',
    ),
    bbox_head=dict(
        type='RepPointsHead',
        num_classes=60,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.1,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.5),
        loss_bbox_refine=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
        transform_method='moment'),
    # training and testing settings
    train_cfg=dict(
        init=dict(
            assigner=dict(type='PointAssigner', scale=4, pos_num=1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        refine=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))



img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
######## added
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
######## added
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
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

optimizer = dict(_delete_=True, type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[8, 11])

runner = dict(type='EpochBasedRunner', max_epochs=12)
