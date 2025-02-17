from data_provider.data_loader import Dataset_net_abilene, Dataset_net_geant
from torch.utils.data import DataLoader

data_dict = {
    'net_traffic_abilene': Dataset_net_abilene,
    'net_traffic_geant': Dataset_net_geant

}

batch_size_gan = 256
def data_provider_gan(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        shuffle_flag = False
        # drop_last = True
        drop_last = False
        batch_size = batch_size_gan  # bsz for train and valid
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = batch_size_gan  # bsz for train and valid
        freq = args.freq
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        percent=percent,
        freq=freq,
        sample_num=args.sample_num,
        seasonal_patterns=args.seasonal_patterns
    )
    batch_size = batch_size_gan
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=False)  # drop_last=drop_last
    return data_set, data_loader
