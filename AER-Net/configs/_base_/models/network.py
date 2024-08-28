model = dict(
    name='Network',
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type=None
    ),
    decode_head=dict(
        type='Network',
        in_ch=3,
        out_ch=1,
        dim=16,  # in dim
        deep_supervision=True
    ),
    loss=dict(type='SoftIoULoss')
)
