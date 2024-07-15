import argparse
import os
import sys
import tempfile
from urllib.parse import urlparse

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

# import parallel training function
from parallel_training import parallel_training

def spmd_main(local_world_size, local_rank):
  # These are the parameters used to initialize the process group
  env_dict = {
    key: os.environ[key]
    for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
  }
  if sys.platform == "win32":
    if "INIT_METHOD" in os.environ.keys():
      print(f"init_method is {os.environ['INIT_METHOD']}")
      url_obj = urlparse(os.environ["INIT_METHOD"])
      if url_obj.scheme.lower() == "file":
        raise ValueError("Windows only supports FileStore")
      else:
        init_method = os.environ["INIT_METHOD"]
    else:
      temp_dir = tempfile.gettempdir()
      init_method = f"file:////{os.path.join(temp_dir, 'ddp_example')}"
    dist.init_process_group(backend="gloo", init_method=init_method, rank=int(env_dict["RANK"]), world_size=int(env_dict["WORLD_SIZE"]))
  else:
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
  print(
      f"[{os.getpid()}]:world_size = {dist.get_world_size()}, "
      + f"rank = {dist.get_rank()}, backened={dist.get_backened()} \n", end=""
  )
  parallel_training(local_world_size, local_rank)
  # Tear down the process group
  dist.destroy_process_group()

if __name__ == "__main__":
  parser =   argparse.ArgumentParser(description="Distributed Data Parallel")
  parser.add_argument("--local_rank", type=int, default=0)
  parser.add_argument("--local_world_size", type=int, default=1)
  args = parser.parse_args()
  spmd_main(args.local_world_size, args.local_rank)