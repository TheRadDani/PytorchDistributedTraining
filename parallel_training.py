import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
# import model module
from model import LSTM

def parallel_training(local_world_size, local_rank):
  
  n = torch.cuda.device_count() // local_world_size
  device_ids = list(range(local_rank * n, (local_rank + 1) * n))
  print(
      f"[{os.getpid()}] rank = {dist.get_rank()}, "
      + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids} \n", end=""
  )

  model = LSTM().cuda(device_ids[0])
  ddp_model = DDP(model, device_ids=device_ids, output_device=device_ids[0])

  #loss_fn = 