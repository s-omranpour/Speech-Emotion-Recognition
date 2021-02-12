import librosa
from librosa import display
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence

class AudioDataset(Dataset):
    def __init__(self, data_path='data/', split_gender=True):
        actors = os.listdir(data_path)
        self.files = [data_path + actor + '/' + file  for actor in actors for file in os.listdir(data_path + actor)]
        self.feel_labels = [int(file.split('/')[-1][:-4].split('-')[2][1])-1 for file in self.files]
        self.gender_labels = [int(file.split('/')[-1][:-4].split('-')[-1][1]) % 2 for file in self.files]
        self.labels = [f + 8*g if split_gender else f for f,g in zip(self.feel_labels, self.gender_labels)]
        self.num_features = 12 + 48 + 64
        self.num_classes = 16 if split_gender else 8
        ## labels are from 0 to 15, 0~7 for female and 8~15 for male

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x, _ = librosa.load(self.files[idx], sr=16000)
        ## MFCC
        mfcc = get_mfcc(x)
        ## chroma
        chroma = get_chroma(x)
        ## mel Spectrogram
        mel = get_mel(x)

        # if self.reduce:
        mfcc = np.mean(mfcc, axis=1)
        chroma = np.mean(chroma, axis=1)
        mel = np.mean(mel, axis=1)
        feat = np.hstack([mfcc, chroma, mel])
        # else:
        #     feat = np.concatenate([mfcc, chroma, mel], axis=0).astype(float)
        #     if feat.shape[1] > self.max_len:
        #         feat = feat[:, :self.max_len]
        #     elif feat.shape[1] < self.max_len:
        #         feat = np.pad(feat, ((0,0), (0,self.max_len - feat.shape[1])))

        return torch.tensor([feat]).float(), self.labels[idx]

def get_mfcc(x):
    return librosa.feature.mfcc(y=x, sr=16000, n_mfcc=48)

def get_chroma(x):
    S = np.abs(librosa.stft(x))
    return librosa.feature.chroma_stft(S=S, sr=16000)

def get_mel(x):
    return librosa.feature.melspectrogram(x, sr=16000, n_mels=64)

def fn(batch):
    X = pad_sequence([s[0] for s in batch], batch_first=True)
    Y = torch.tensor([s[1] for s in batch])
    return X, Y

def get_loaders(dataset, bs=32, val_frac=0.1):
    n = len(dataset)
    t = int(n*val_frac)
    print('train samples:', n-t, 'val samples:', t)
    train, val = random_split(dataset, [n-t, t])
    tl = DataLoader(train, shuffle=True, batch_size=bs)
    vl = DataLoader(val, shuffle=False, batch_size=bs)
    return tl, vl

if __name__ == '__main__':
    dataset = AudioDataset('../data/')
    file = dataset.files[0]
    print(file)
    x, _ = librosa.load(file, sr=16000)
    mfcc = get_mfcc(x)
    print('mfcc:', mfcc.shape)
    chroma = get_chroma(x)
    print('chroma:', chroma.shape)
    mel = librosa.amplitude_to_db(get_mel(x))
    print('mel:', mel.shape)
    plt.title('MFCC')
    display.specshow(mfcc, x_axis='time', y_axis='log', sr=16000)
    plt.show()

    plt.title('chroma')
    display.specshow(chroma, x_axis='time', y_axis='chroma', sr=16000)
    plt.show()

    plt.title('Mel Spectrogram')
    display.specshow(mel, x_axis='time', y_axis='log', sr=16000)
    plt.show()






    
