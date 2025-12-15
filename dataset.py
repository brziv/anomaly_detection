import torch.utils.data as data
import numpy as np
import torch
import random
torch.set_float32_matmul_precision('medium')
import option
args=option.parse_args()

class Dataset(data.Dataset):
    def __init__(self, args, test_mode=False):
        if test_mode:
            self.rgb_list_file = args.test_rgb_list
        else:
            self.rgb_list_file = args.rgb_list

        self.test_mode = test_mode
        self.list = [line.strip().split() for line in open(self.rgb_list_file)]
        self.paths = [parts[0] for parts in self.list]
        self.labels = [0.0 if parts[1] == "normal" else 1.0 for parts in self.list]
        self.n_len = sum(1 for l in self.labels if l == 0.0)
        self.a_len = sum(1 for l in self.labels if l == 1.0)


    def __getitem__(self, index):
        if not self.test_mode:
            if index == 0:
                self.n_ind = [i for i, l in enumerate(self.labels) if l == 0.0]
                self.a_ind = [i for i, l in enumerate(self.labels) if l == 1.0]
                random.shuffle(self.n_ind)
                random.shuffle(self.a_ind)

            nindex = self.n_ind.pop()
            aindex = self.a_ind.pop()

            nfeatures = np.load(self.paths[nindex], allow_pickle=True)
            nfeatures = np.array(nfeatures, dtype=np.float32)
            nlabel = self.labels[nindex]

            afeatures = np.load(self.paths[aindex], allow_pickle=True)
            afeatures = np.array(afeatures, dtype=np.float32)
            alabel = self.labels[aindex]

            return nfeatures, nlabel, afeatures, alabel
    
        else:
            features = np.load(self.paths[index], allow_pickle=True)
            features = np.array(features, dtype=np.float32)
            label = self.labels[index]
            return features, label

    def __len__(self):

        if self.test_mode:
            return len(self.list)
        else:
            return min(self.a_len, self.n_len)
