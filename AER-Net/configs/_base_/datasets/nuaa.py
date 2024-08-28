# dataset settings
data = dict(
    dataset_type='NUAA',
    data_root='./datasets/NUAA-SIRST',
    base_size=256,
    crop_size=256,
    data_aug=True,
    suffix='png',
    num_workers=8,
    train_batch=16,
    test_batch=16,
    train_dir='trainval',
    test_dir='test'
)
