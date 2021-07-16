""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""

import os
import functools
import math
import numpy as np
from tqdm import tqdm, trange


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision
import horovod.torch as hvd

# Import my stuff
import inception_utils
import utils
import losses
import train_fns
from sync_batchnorm import patch_replication_callback

import random
import sys
import signal
from datetime import datetime
from time import time


# The main training file. Config is a dictionary specifying the configuration
# of this training run.
def run(config, n_iter=10):
  # Update the config dict as necessary
  # This is for convenience, to add settings derived from the user-specified
  # configuration into the config-dict (e.g. inferring the number of classes
  # and size of the images from the dataset, passing in a pytorch object
  # for the activation specified as a string)
  config['resolution'] = utils.imsize_dict[config['dataset']]
  config['n_classes'] = utils.nclass_dict[config['dataset']]
  config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['D_activation'] = utils.activation_dict[config['D_nl']]

  device = 'cuda'
  
  # Seed RNG
  utils.seed_rng(config['seed'] + hvd.rank())

  # Setup cudnn.benchmark for free speed
  torch.backends.cudnn.benchmark = True

  # Import the model--this line allows us to dynamically select different files.
  model = __import__(config['model'])
  experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
  if hvd.rank() == 0:
    print('Experiment name is %s' % experiment_name)

  # Next, build the model
  G = model.Generator(**config).to(device)
  D = model.Discriminator(**config).to(device)

  GD = model.G_D(G, D)
  if hvd.rank() == 0:
    print('Number of params in G: {} D: {}'.format(
      *[sum([p.data.nelement() for p in net.parameters()]) for net in [G,D]]))
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}

  hvd.broadcast_parameters(G.state_dict(), root_rank=0)
  hvd.broadcast_parameters(D.state_dict(), root_rank=0)
  hvd.broadcast_optimizer_state(G.optim, root_rank=0)
  hvd.broadcast_optimizer_state(D.optim, root_rank=0)

  # Prepare data; the Discriminator's batch size is all that needs to be passed
  # to the dataloader, as G doesn't require dataloading.
  # Note that at every loader iteration we pass in enough data to complete
  # a full D iteration (regardless of number of D steps and accumulations)
  D_batch_size = (config['batch_size'] * config['num_D_steps']
                  * config['num_D_accumulations']) // hvd.size()
  loaders = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size,
                                      'start_itr': state_dict['itr']})
  loader = loaders[0]
  n_epochs = config["num_epochs"]

  # Prepare noise and randomly sampled label arrays
  # Allow for different batch sizes in G
  G_batch_size = max(config['G_batch_size'], config['batch_size']) // hvd.size()
  z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                             device=device, fp16=config['G_fp16'])
  # Loaders are loaded, prepare the training function
  if config['which_train_fn'] == 'GAN':
    train = train_fns.GAN_training_function(G, D, GD, z_, y_, 
                                            None, state_dict, config)
  # Else, assume debugging and use the dummy train fn
  else:
    train = train_fns.dummy_training_function()

  if hvd.rank() == 0:
    print('Using dataloder %s' % loader)
    if hasattr(loader, 'root'): print("root: %s"%loader.root)
    if hasattr(loader, 'dataset'):
      if hasattr(loader.dataset, 'root'): print("root: %s"%loader.dataset.root)
    if hasattr(loader, 'num_workers'): print("num_workers: %d"%loader.num_workers)
    print('Beginning training at epoch %d...' % state_dict['epoch'])
    print("n_iter\telapsed\timgs/sec", flush=True)
  # Train for specified number of epochs, although we mostly track G iterations.
  for epoch in range(n_epochs):
    # Horovod: set epoch to sampler for shuffling.
    if hasattr(loader, "sampler"): loader.sampler.set_epoch(epoch)
    start = time()
    for i, (x, y) in enumerate(loader):
      # Increment the iteration counter
      state_dict['itr'] += 1
      # Make sure G and D are in training mode, just in case they got set to eval
      # For D, which typically doesn't have BN, this shouldn't matter much.
      G.train()
      D.train()
      x, y = x.to(device), y.to(device)
      metrics = train(x, y)

      # If using my progbar, print metrics.
      if i + 1 >= n_iter:
        break

    elapsed = time() - start
    tot_imgs = n_iter * hvd.size() * D_batch_size
    if hvd.rank() == 0:
      print("%d\t%.2f\t%.2f"%(n_iter, elapsed, tot_imgs/elapsed), flush=True)
    # Increment epoch counter at end of epoch
    state_dict['epoch'] += 1


def main():
  # parse command line and run
  parser = utils.prepare_parser()
  config = vars(parser.parse_args())
  hvd.init()

  if torch.cuda.is_available():
    torch.cuda.set_device(hvd.local_rank())

  now = datetime.now()
  dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

  print('%s - Hello, rank = %d, local_rank = %d, size = %d, local_size = %d\n \
         device: %s\tcuda_device: %d' % (dt_string, hvd.rank(), hvd.local_rank(),
         hvd.size(), hvd.local_size(), hvd.local_rank(), torch.cuda.current_device()))
  sys.stdout.flush()
  sys.stderr.flush()


  start = time()
  if config["copy_in_mem"]:
    utils.copy_data_in_mem(config)
  if hvd.rank() == 0:
    print(config)
  run(config, n_iter=10)
  if config["copy_in_mem"]:
    utils.rm_data_in_mem(**config)
  if hvd.rank() == 0:
    print("Run over in %.2f sec\n\n\n"%(time()-start))
  sys.stdout.flush()
  sys.stderr.flush()

if __name__ == '__main__':
  main()
