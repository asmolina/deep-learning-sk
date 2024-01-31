_base_ = [
    # './_base_/models/cascade_mask_rcnn_swin_fpn.py'
    './_base_/models/cascade_mask_rcnn_r50_fpn.py',
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
    # backbone=dict(
    #     embed_dim=128,
    #     depths=[2, 2, 18, 2],
    #     num_heads=[4, 8, 16, 32],
    #     window_size=7,
    #     ape=False,
    #     drop_path_rate=0.3,
    #     patch_norm=True,
    #     use_checkpoint=False
    # ),
    # neck=dict(in_channels=[128, 256, 512, 1024]),

    ### configs from './_base_/models/cascade_mask_rcnn_r50_fpn.py'
    type='CascadeRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    ### configs from './_base_/models/cascade_mask_rcnn_r50_fpn.py'

    roi_head=dict(
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=60, #########################
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=60, #########################
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=60, #########################
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
            ],
        mask_head=dict(num_classes=60)
    ))
        
        


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                 (736, 1333), (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
data = dict(train=dict(pipeline=train_pipeline))



data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        pipeline=train_pipeline,
        type=dataset_type,
        classes=classes,
        ann_file='/content/mmdetection/TACO-master/data/annotations_0_train.json',
        img_prefix='/content/mmdetection/TACO-master/data'),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/content/mmdetection/TACO-master/data/annotations_0_val.json',
        img_prefix='/content/mmdetection/TACO-master/data'),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/content/mmdetection/TACO-master/data/annotations_0_test.json',
        img_prefix='/content/mmdetection/TACO-master/data'))


optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)

