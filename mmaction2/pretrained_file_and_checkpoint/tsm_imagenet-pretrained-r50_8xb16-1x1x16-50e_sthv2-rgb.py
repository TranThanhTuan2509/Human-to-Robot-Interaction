ann_file_train = 'data/sthv2/sthv2_train_list_videos.txt'
ann_file_val = 'data/sthv2/sthv2_val_list_videos.txt'
auto_scale_lr = dict(base_batch_size=128, enable=False)
data_root = 'data/sthv2/videos'
dataset_type = 'VideoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=3, max_keep_ckpts=3, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(io_backend='disk')
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    backbone=dict(
        depth=50,
        norm_eval=False,
        num_segments=16,
        pretrained='torchvision://resnet50',
        shift_div=8,
        type='ResNetTSM'),
    cls_head=dict(
        average_clips='prob',
        consensus=dict(dim=1, type='AvgConsensus'),
        dropout_ratio=0.5,
        in_channels=2048,
        init_std=0.001,
        is_shift=True,
        num_classes=174,
        num_segments=16,
        spatial_type='avg',
        type='TSMHead'),
    data_preprocessor=dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='ActionDataPreprocessor'),
    test_cfg=None,
    train_cfg=None,
    type='Recognizer2D')
optim_wrapper = dict(
    clip_grad=dict(max_norm=20, norm_type=2),
    constructor='TSMOptimWrapperConstructor',
    optimizer=dict(lr=0.02, momentum=0.9, type='SGD', weight_decay=0.0005),
    paramwise_cfg=dict(fc_lr5=True))
param_scheduler = [
    dict(begin=0, by_epoch=True, end=5, start_factor=0.1, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=50,
        gamma=0.1,
        milestones=[
            25,
            45,
        ],
        type='MultiStepLR'),
]
preprocess_cfg = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ], std=[
        58.395,
        57.12,
        57.375,
    ])
resume = False
sthv2_flip_label_map = dict({
    166: 167,
    167: 166,
    86: 87,
    87: 86,
    93: 94,
    94: 93
})
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='data/sthv2/sthv2_val_list_videos.txt',
        data_prefix=dict(video='data/sthv2/videos'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=1,
                frame_interval=1,
                num_clips=8,
                test_mode=True,
                twice_sample=True,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=256, type='ThreeCrop'),
            dict(input_format='NCHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    pipeline=[
        dict(io_backend='disk', type='DecordInit'),
        dict(
            clip_len=1,
            frame_interval=1,
            num_clips=16,
            test_mode=True,
            twice_sample=True,
            type='SampleFrames'),
        dict(type='DecordDecode'),
        dict(scale=(
            -1,
            256,
        ), type='Resize'),
        dict(crop_size=256, type='ThreeCrop'),
        dict(input_format='NCHW', type='FormatShape'),
        dict(type='PackActionInputs'),
    ],
    sampler=dict(shuffle=False, type='DefaultSampler'),
    test_mode=True)
test_evaluator = dict(type='AccMetric')
test_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(
        clip_len=1,
        frame_interval=1,
        num_clips=16,
        test_mode=True,
        twice_sample=True,
        type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(crop_size=256, type='ThreeCrop'),
    dict(input_format='NCHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
train_cfg = dict(
    max_epochs=50, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='data/sthv2/sthv2_train_list_videos.txt',
        data_prefix=dict(video='data/sthv2/videos'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=1,
                frame_interval=1,
                num_clips=16,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(
                input_size=224,
                max_wh_scale_gap=1,
                num_fixed_crops=13,
                random_crop=False,
                scales=(
                    1,
                    0.875,
                    0.75,
                    0.66,
                ),
                type='MultiScaleCrop'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(
                flip_label_map=dict({
                    166: 167,
                    167: 166,
                    86: 87,
                    87: 86,
                    93: 94,
                    94: 93
                }),
                flip_ratio=0.5,
                type='Flip'),
            dict(input_format='NCHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(clip_len=1, frame_interval=1, num_clips=16, type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(
        input_size=224,
        max_wh_scale_gap=1,
        num_fixed_crops=13,
        random_crop=False,
        scales=(
            1,
            0.875,
            0.75,
            0.66,
        ),
        type='MultiScaleCrop'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(
        flip_label_map=dict({
            166: 167,
            167: 166,
            86: 87,
            87: 86,
            93: 94,
            94: 93
        }),
        flip_ratio=0.5,
        type='Flip'),
    dict(input_format='NCHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='data/sthv2/sthv2_val_list_videos.txt',
        data_prefix=dict(video='data/sthv2/videos'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=1,
                frame_interval=1,
                num_clips=16,
                test_mode=True,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='AccMetric')
val_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(
        clip_len=1,
        frame_interval=1,
        num_clips=16,
        test_mode=True,
        type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(input_format='NCHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
