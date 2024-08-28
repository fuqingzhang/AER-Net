_base_ = [
    '../_base_/datasets/irstd1k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_500e.py',
    '../_base_/models/network.py'
]
model = dict(
    decode_head=dict(
        dim=16
    )
)
optimizer = dict(
    type='AdamW',
    setting=dict(lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
)
# optimizer = dict(
#     type='Adagrad',
#     setting=dict(lr=0.05, weight_decay=0.0001)
# )
runner = dict(type='EpochBasedRunner', max_epochs=1500)
data = dict(train_batch=16, test_batch=16)
