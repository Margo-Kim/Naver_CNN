import torch.nn as nn
import torch
from dataset import SenCLSDataset
from torch.utils.data import DataLoader
from model import SenCNN
from vocab import get_vocab
from ds import *


dataset = ds()
x_train, y_train = dataset.get_train()
data = list(zip(x_train, y_train))

vocab = get_vocab()
dataset = SenCLSDataset(data, vocab)
loader = DataLoader(dataset, batch_size=10)
model = SenCNN(len(vocab),100)


criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 0
for epoch in range(nb_epochs+1):
    avg_cost = 0
    total_batch = len(loader)

    for i, batch in enumerate(loader):
        x, y = batch
        optimizer.zero_grad()
        hat_y = model(x)  # [bsz, vocab_len]
        
        cost = criterion(hat_y, y.view(-1))
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch
        print('idx {:4d} / cost: {:.6f}'.format(i, cost))
    if epoch % 10 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))

print('Learning finished')
