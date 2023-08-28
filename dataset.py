from torch.utils.data import Dataset
import torch
from constant import START_TOKEN, END_TOKEN
from torch.nn.utils.rnn import pad_sequence


class SenCLSDataset(Dataset):
    def __init__(self, datas, vocab):
        self.preprocess(datas, vocab)

    def preprocess(self, datas, vocab):
        self.x, self.y = [], []
        for data in datas:
            sent, label = data

            
            if isinstance(sent, str):  # Check if 'sent' is a string
                words = sent.split()
                self.x.append([vocab[w] for w in words if w in vocab])  # Only include words in vocab
            else:
                self.x.append([])  # Add empty list for non-string sentences
            self.y.append(label)

        
        self.x = [torch.tensor(seq) for seq in self.x]
        self.x = pad_sequence(self.x, batch_first=True, padding_value=0)
        # Convert self.y to a tensor
        self.y = torch.LongTensor(self.y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


