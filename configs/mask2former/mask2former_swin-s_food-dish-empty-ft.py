_base_ = ['./mask2former_swin-s_8xb2-160k_ade20k-512x512.py']

custom_imports = dict(
    imports=['mmseg.datasets.food_dish_dataset'],
    allow_failed_imports=False
)

num_classes = 3

dataset_type = 'FoodDishDataset'
data_root = 'data/food_dish_correction'
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='annotations/train'),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/val',
            seg_map_path='annotations/val'),
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

model = dict(
    decode_head=dict(
        num_classes=num_classes,
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0, 1.0, 1.0, 0.1]
        )
    )
)

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-5,
        betas=(0.9, 0.999),
        weight_decay=0.05,
        eps=1e-8),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=dict({
            'absolute_pos_embed': dict(decay_mult=0.0, lr_mult=1.0),
            'backbone': dict(decay_mult=1.0, lr_mult=1.0),
            'backbone.norm': dict(decay_mult=0.0, lr_mult=1.0),
            'backbone.patch_embed.norm': dict(decay_mult=0.0, lr_mult=1.0),
            'relative_position_bias_table': dict(decay_mult=0.0, lr_mult=1.0),
            'query_embed': dict(decay_mult=0.0, lr_mult=1.0),
            'query_feat': dict(decay_mult=0.0, lr_mult=1.0),
            'level_embed': dict(decay_mult=0.0, lr_mult=1.0),
        }),
        norm_decay_mult=0.0
    ),
    type='OptimWrapper'
)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=1000,
        by_epoch=False)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=1000, val_interval=200)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=500,
        save_best='mIoU',
        max_keep_ckpts=10
    ),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False)
)

load_from = 'best_mIoU_iter_12000.pth'