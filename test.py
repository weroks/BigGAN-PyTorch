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

interrupted = torch.tensor(0)
def signal_handler(signum, frame):
  if hvd.rank() == 0:
    print('Rank %d got signal %d.'%(hvd.rank(), signum), flush=True)
    global interrupted
    interrupted = torch.tensor(signum)

# The main training file. Config is a dictionary specifying the configuration
# of this training run.
def run(config):
  # Update the config dict as necessary
  # This is for convenience, to add settings derived from the user-specified
  # configuration into the config-dict (e.g. inferring the number of classes
  # and size of the images from the dataset, passing in a pytorch object
  # for the activation specified as a string)
  config['resolution'] = utils.imsize_dict[config['dataset']]
  config['n_classes'] = utils.nclass_dict[config['dataset']]
  config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['D_activation'] = utils.activation_dict[config['D_nl']]
  # By default, skip init if resuming training.
  if config['resume']:
    if hvd.rank() == 0:
      print('Skipping initialization for training resumption...')
    config['skip_init'] = True
  config = utils.update_config_roots(config)
  device = 'cuda'

  # Seed RNG
  utils.seed_rng(config['seed'] + hvd.rank())

  # Prepare root folders if necessary
  if hvd.rank() == 0:
    utils.prepare_root(config)

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

   # If using EMA, prepare it
  if config['ema']:
    if hvd.rank() == 0:
      print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
    G_ema = model.Generator(**{**config, 'skip_init':True,
                               'no_optim': True}).to(device)
    ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
  else:
    G_ema, ema = None, None

  # FP16?
  if config['G_fp16']:
    print('Casting G to float16...')
    G = G.half()
    if config['ema']:
      G_ema = G_ema.half()
  if config['D_fp16']:
    print('Casting D to fp16...')
    D = D.half()
    # Consider automatically reducing SN_eps?
  GD = model.G_D(G, D)
  if hvd.rank() == 0:
    # print(G)
    # print(D)
    print('Number of params in G: {} D: {}'.format(
      *[sum([p.data.nelement() for p in net.parameters()]) for net in [G,D]]))
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}

  # If loading from a pre-trained model, load weights
  if config['resume']:
    if config['load_from']:
      root = config['load_from']
    else:
      root = '/'.join([config['weights_root'], experiment_name])

    if hvd.rank() == 0:
      print('Loading weights...')
      utils.load_weights(G, D, state_dict, root,
                        config['load_weights'] if config['load_weights'] else None,
                        G_ema if config['ema'] else None)
    else:
      name_suffix = config['load_weights'] if config['load_weights'] else None
      for item in state_dict:
        state_dict[item] = torch.load('%s/%s.pth' % (root, utils.join_strings('_', ['state_dict', name_suffix])))[item]

  hvd.broadcast_parameters(G.state_dict(), root_rank=0)
  hvd.broadcast_parameters(D.state_dict(), root_rank=0)
  hvd.broadcast_optimizer_state(G.optim, root_rank=0)
  hvd.broadcast_optimizer_state(D.optim, root_rank=0)
  if config['ema']:
    hvd.broadcast_parameters(G_ema.state_dict(), root_rank=0)

  # Prepare loggers for stats; metrics holds test metrics,
  # lmetrics holds any desired training metrics.
  if hvd.rank() == 0:
    test_metrics_fname = '%s/%s_log.jsonl' % (config['logs_root'],
                                              experiment_name)
    train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
    test_log = utils.MetricsLogger(test_metrics_fname,
                                  reinitialize=(not config['resume']))
    train_log = utils.MyLogger(train_metrics_fname,
                              reinitialize=(not config['resume']),
                              logstyle=config['logstyle'])
    print('Inception Metrics will be saved to {}'.format(test_metrics_fname))
    print('Training Metrics will be saved to {}'.format(train_metrics_fname))
    # Write metadata
    utils.write_metadata(config['logs_root'], experiment_name, config, state_dict)
  else:
    test_log = None
  # Prepare data; the Discriminator's batch size is all that needs to be passed
  # to the dataloader, as G doesn't require dataloading.
  # Note that at every loader iteration we pass in enough data to complete
  # a full D iteration (regardless of number of D steps and accumulations)
  D_batch_size = (config['batch_size'] * config['num_D_steps']
                  * config['num_D_accumulations']) // hvd.size()
  loaders = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size,
                                      'start_itr': state_dict['itr']})
  # Bug when use_multiepoch_sampler, epochs squared!
  n_epochs = config["num_epochs"]
  if config["use_multiepoch_sampler"]:
    # TODO:
    raise "No use_multiepoch_sampler"
    n_epochs = 1

  # Prepare inception metrics: FID and IS
  get_inception_metrics = inception_utils.prepare_inception_metrics(dataset=config['dataset'],
                                                                    parallel=False,
                                                                    no_fid=config['no_fid'])

  # Prepare noise and randomly sampled label arrays
  # Allow for different batch sizes in G
  G_batch_size = max(config['G_batch_size'], config['batch_size']) // hvd.size()
  z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                             device=device, fp16=config['G_fp16'])
  # Prepare a fixed z & y to see individual sample evolution throghout training
  fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.dim_z,
                                       config['n_classes'], device=device,
                                       fp16=config['G_fp16'])
  fixed_z.sample_()
  fixed_y.sample_()
  # Loaders are loaded, prepare the training function
  if config['which_train_fn'] == 'GAN':
    train = train_fns.GAN_training_function(G, D, GD, z_, y_,
                                            ema, state_dict, config)
  # Else, assume debugging and use the dummy train fn
  else:
    train = train_fns.dummy_training_function()
  # Prepare Sample function for use with inception metrics
  sample = functools.partial(utils.sample,
                              G=(G_ema if config['ema'] and config['use_ema']
                                 else G),
                              z_=z_, y_=y_, config=config)
  if hvd.rank() == 0:
    print('Beginning training at epoch %d...' % state_dict['epoch'])
  # Train for specified number of epochs, although we mostly track G iterations.
  for epoch in range(state_dict['epoch'], n_epochs):

    # Horovod: set epoch to sampler for shuffling.
    loaders[0].sampler.set_epoch(epoch)

    # Which progressbar to use? TQDM or my own?
    if config['pbar'] == 'mine':
      pbar = utils.progress(loaders[0],displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
    else:
      # TODO:
      raise "No tqdm bar"
      pbar = tqdm(loaders[0])

    if config['G_eval_mode']:
      if hvd.rank() == 0:
        print('Switchin G to eval mode...')
      G.eval()
      if config['ema']:
        G_ema.eval()
    train_fns.test(G, D, G_ema, z_, y_, state_dict, config, sample,
                    get_inception_metrics, experiment_name, test_log)
    train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y,
                              state_dict, config, experiment_name)
    break


def main():
  # parse command line and run
  parser = utils.prepare_parser()
  config = vars(parser.parse_args())

  hvd.init()

  if torch.cuda.is_available():
    torch.cuda.set_device(hvd.local_rank())

  now = datetime.now()
  dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

  # print('%s - Hello, rank = %d, local_rank = %d, size = %d, local_size = %d\n \
  #        device: %s\tcuda_device: %d' % (dt_string, hvd.rank(), hvd.local_rank(),
  #        hvd.size(), hvd.local_size(), hvd.local_rank(), torch.cuda.current_device()))

  if hvd.rank() == 0:
    print(config)

  if config["copy_in_mem"]:
    utils.copy_data_in_mem(config)

  # Register the signal handler
  signal.signal(signal.SIGUSR1, signal_handler)
  signal.signal(signal.SIGTERM, signal_handler)
  signal.signal(signal.SIGINT, signal_handler)
  sys.stdout.flush()
  sys.stderr.flush()
  run(config)

if __name__ == '__main__':
  main()
