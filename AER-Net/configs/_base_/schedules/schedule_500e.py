# optimizer
optimizer = dict(
    type='AdamW',
    setting=dict(lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
)
optimizer_config = dict()
# learning policy
# TODO warmup only 'linear'
lr_config = dict(policy='PolyLR', warmup='linear', power=0.9, min_lr=1e-4, warmup_epochs=0)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=500)
checkpoint_config = dict(by_epoch=False, interval=1)
evaluation = dict(epochval=1, metric='mIoU', pre_eval=True)
