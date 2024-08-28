# dataset settings
data = dict(
    dataset_type='BSIRST_v1',
    data_root='./datasets/BSIRST_v1',
    base_size=512,
    crop_size=512,
    data_aug=True,
    suffix='png',
    num_workers=8,
    train_batch=16,
    test_batch=16,
    train_dir='trainval',
    test_dir='test'
)
