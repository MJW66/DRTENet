# dataset settings
data = dict(
    dataset_type='NUAA',
    data_root='/home/b311/data3/majingwen/ABC-add/datasets/NUAA',
    base_size=512,
    crop_size=512,
    data_aug=True,
    suffix='png',
    num_workers=8,
    train_batch=16,
    test_batch=8,
    train_dir='trainval',
    test_dir='test'
)
