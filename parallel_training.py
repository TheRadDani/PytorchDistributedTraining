import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
# import model module
from model import LSTM

#import dataloaders
from dataloader import train_data_loader, valid_data_loader, test_data_loader, vocab, train_data, pad_index, vectors

vocab_size = len(vocab)
embedding_dim = 300
hidden_dim = 300
output_dim = len(train_data.unique("label"))
n_layers = 2
bidirectional = True
dropout_rate = 0.5

#hyperparameters
lr = 5e-4
n_epochs = 5

# Initialize weights depending on the type of nn
def initialize_weights(m):
  if isinstance(m, nn.Linear):
    nn.init.xavier_normal_(m.weight)
    nn.init.zeros_(m.bias)
  elif isinstance(m, nn.LSTM):
    for name, param in m.named_parameters():
      if "bias" in name:
        nn.init.zeros_(param)
      elif "weight" in name:
        nn.init.orthogonal_(param)


def parallel_training(local_world_size, local_rank):
  
  n = torch.cuda.device_count() // local_world_size
  device_ids = list(range(local_rank * n, (local_rank + 1) * n))
  print(
      f"[{os.getpid()}] rank = {dist.get_rank()}, "
      + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids} \n", end=""
  )

  model = LSTM(
    vocab_size,
    embedding_dim,
    hidden_dim,
    output_dim,
    n_layers,
    bidirectional,
    dropout_rate,
    pad_index,).cuda(device_ids[0])
  model.apply(initialize_weights)
  
  pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())
  model.embedding.weight.data = pretrained_embedding

  ddp_model = DDP(model, device_ids=device_ids, output_device=device_ids[0])

  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.Adam(ddp_model.parameters(), lr=lr)

  for _ in n_epochs:
    ddp_model.train()
    for batch in train_data_loader:
      ids = batch["ids"]
      length = batch["length"]
      label = batch["label"]
      optimizer.zero_grad()
      prediction = ddp_model(ids, length)
      loss = loss_fn(prediction, label)
      loss.backwards()
      optimizer.step()
