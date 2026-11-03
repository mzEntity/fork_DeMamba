import torch.utils.data as data
from torch.utils.data import Dataset
import pandas as pd
import torch
import random

from .video_processing import crop_center_by_percentage, prepare_clip


class Ours_Dataset_train(Dataset):
    def __init__(self, index_list=None, df=None):
        self.index_list = index_list
        self.df = df
        self.positive_indices = df[df['label'] == 1].index.tolist()
        self.negative_indices = df[df['label'] == 0].index.tolist()
        self.balanced_indices = []
        self.resample()

    def resample(self):
        # Ensure each epoch uses a balanced dataset
        min_samples = min(len(self.positive_indices), len(self.negative_indices))
        self.balanced_indices.clear()
        self.balanced_indices.extend(random.sample(self.positive_indices, min_samples))
        self.balanced_indices.extend(random.sample(self.negative_indices, min_samples))
        random.shuffle(self.balanced_indices)  # Shuffle to mix positive and negative samples

    def __getitem__(self, idx):
        real_idx = self.balanced_indices[idx]
        row = self.df.iloc[real_idx]
        video_id = row['content_path']
        label = row['label']
        frame_list = eval(row['frame_seq'])
        label_onehot = [0]*2
        frames = prepare_clip(
            len(frame_list),
            lambda i: download_oss_file('GenVideo/' + video_id + '/' + str(frame_list[i]) + '.jpg'),
            training=True, video_id=video_id,
        ).unsqueeze(0)
        label_onehot[int(label)] = 1
        label_onehot = torch.FloatTensor(label_onehot)
        binary_label = torch.FloatTensor([int(label)])

        return self.index_list[idx], frames, label_onehot, binary_label

    def __len__(self):
        return len(self.balanced_indices)


class Ours_Dataset_val(data.Dataset):
    def __init__(self, cfg, index_list=None, df=None):
        self.index_list = index_list
        self.cfg = cfg
        self.df = df
        self.frame_dir = df['image_path'].tolist()

    def __getitem__(self, idx):
        df_v = self.df.loc[self.index_list[idx]]
        video_id = df_v['content_path']
        label = df_v['label']
        label_onehot = [0]*2
        frame_list = eval(df_v['frame_seq'])

        frames = prepare_clip(
            len(frame_list),
            lambda i: download_oss_file('GenVideo/' + video_id + '/' + str(frame_list[i]) + '.jpg'),
            training=False, video_id=video_id, task=self.cfg['task'],
        ).unsqueeze(0)
        label_onehot[int(label)] = 1
        label_onehot = torch.FloatTensor(label_onehot)
        binary_label = torch.FloatTensor([int(label)])
        return self.index_list[idx], frames, label_onehot, binary_label, video_id

    
    def __len__(self):
        return len(self.index_list)



def generate_dataset_loader(cfg):
    df_train = pd.read_csv('GenVideo/datasets/train.csv')

    if cfg['task'] == 'normal':
        df_val = pd.read_csv('GenVideo/datasets/val_id.csv')
    elif cfg['task'] == 'robust_compress':
        df_val = pd.read_csv('GenVideo/datasets/com_28.csv')
    elif cfg['task'] == 'Image_Water_Attack':
        df_val = pd.read_csv('GenVideo/datasets/imgwater.csv')
    elif cfg['task'] == 'Text_Water_Attack':
        df_val = pd.read_csv('GenVideo/datasets/textwater.csv')
    elif cfg['task'] == 'one2many':
        df_val = pd.read_csv('GenVideo/datasets/val_ood.csv')
        if cfg['train_sub_set'] == 'pika':
            prefixes = ["fake/pika", "real"]
            video_condition = df_train['content_path'].str.startswith(prefixes[0])
            for prefix in prefixes[1:]:
                video_condition |= df_train['content_path'].str.startswith(prefix)
            df_train = df_train[video_condition]
        elif cfg['train_sub_set'] == 'SEINE':
            prefixes = ["fake/SEINE", "real"]
            video_condition = df_train['content_path'].str.startswith(prefixes[0])
            for prefix in prefixes[1:]:
                video_condition |= df_train['content_path'].str.startswith(prefix)
            df_train = df_train[video_condition]
        elif cfg['train_sub_set'] == 'OpenSora':
            prefixes = ["fake/OpenSora", "real"]
            video_condition = df_train['content_path'].str.startswith(prefixes[0])
            for prefix in prefixes[1:]:
                video_condition |= df_train['content_path'].str.startswith(prefix)
            df_train = df_train[video_condition]
        elif cfg['train_sub_set'] == 'Latte':
            prefixes = ["fake/Latte", "real"]
            video_condition = df_train['content_path'].str.startswith(prefixes[0])
            for prefix in prefixes[1:]:
                video_condition |= df_train['content_path'].str.startswith(prefix)
            df_train = df_train[video_condition]
    else:
        df_val = pd.read_csv('GenVideo/datasets/val_ood.csv')

    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)
    
    index_val = df_val.index.tolist()
    index_val = index_val[:]

    val_dataset = Ours_Dataset_val(cfg, index_val, df_val)
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg['val_batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=True, drop_last=False
        )

    index_train = df_train.index.tolist()
    index_train = index_train[:]
    train_dataset = Ours_Dataset_train(index_train, df_train)
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg['train_batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=True, drop_last=True
        )

    print("******* Training Video IDs", str(len(index_train))," Training Batch size ", str(cfg['train_batch_size'])," *******")
    print("******* Testing Video IDs", str(len(index_val)), " Testing Batch size ", str(cfg['val_batch_size'])," *******")

    return train_loader, val_loader


