# dataset settings
data = dict(
    dataset_type='NUDT',
    data_root='./datasets/NUDT-SIRST',
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
