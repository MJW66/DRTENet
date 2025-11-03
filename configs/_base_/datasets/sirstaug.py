# dataset settings
data = dict(
    dataset_type='SIRSTAUG',
    data_root='/home/b311/data3/majingwen/ABC-add/datasets/SIRST_AUG',
    base_size=256,
    crop_size=256,
    data_aug=True,
    suffix='png',
    num_workers=8,
    train_batch=32,
    test_batch=32,
    train_dir='trainval',
    test_dir='test'
)
