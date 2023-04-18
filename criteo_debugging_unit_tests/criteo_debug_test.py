import jax
import tensorflow as tf
import psutil
from typing import Optional
import os
import functools
import matplotlib.pyplot as plt
import numpy as np


def get_criteo1tb_dataset(split: str,
                          shuffle_rng,
                          data_dir: str,
                          global_batch_size: int,
                          shuffle: bool = True,
                          interleave: bool = True,
                          num_parallel_calls_interleave: int = 16,
                          cycle_length: int = 128,
                          block_length_fraction: int = 8,
                          num_files: int = 849,
                          prefetch: int = 10,):
  """Get the Criteo 1TB dataset for a given split."""
  files = os.listdir(data_dir)
  train_files = [os.path.join(data_dir, f) for f in files if int(f.split("_")[1]) in range(0,23)]
  train_files = train_files[:num_files]

  ds = tf.data.Dataset.list_files(train_files, shuffle=shuffle, seed=shuffle_rng[0])

  ds = ds.interleave(
      tf.data.TextLineDataset,
      cycle_length=128,
      block_length=65536,
      num_parallel_calls=16,
      deterministic=False)
  
  ds = ds.shuffle(buffer_size=524_288 * 100, seed=shuffle_rng[1])
  ds = ds.batch(524288, drop_remainder=True)

  ds = ds.repeat()
  ds = ds.prefetch(10)

  return ds


def get_ram_usage():
    return psutil.virtual_memory()[3]/1000000000


def make_dataset_and_iter(save_dir,
              global_batch_size: int = int(65536*8),
              shuffle: bool = True,
              interleave: bool = True,
              num_parallel_calls_interleave: int = 16,
              cycle_length: int = 128,
              block_length_fraction: int = 8,
              num_files: int = 849,
              prefetch: int = 10,):
    test_name = (f'_cycle_length_{cycle_length}_block_size_fraction_{block_length_fraction}')
    print(f'Running test {test_name}')
    mem_usage = []
    ds = iter(get_criteo1tb_dataset(split='train', 
                                shuffle_rng=jax.random.PRNGKey(0), 
                                data_dir='/home/kasimbeg/data/criteo1tb',
                                global_batch_size=global_batch_size,
                                shuffle=shuffle,
                                cycle_length=cycle_length,
                                block_length_fraction=block_length_fraction,
                                num_files=num_files,
                                prefetch=prefetch))

    # Iterate and log memory usage  
    m = get_ram_usage()                          
    mem_usage.append(m)
    print(f"Batch: {0}: RAM USED (GB) {m}")
    for i in range(1, 2001):
        next_batch = next(ds)
        if (i % 100 == 0) or (i == 1):
            m = get_ram_usage()
            mem_usage.append(m)
            print(f"Batch: {i}. RAM USED (GB) {m}")
     
    np.save(os.path.join(save_dir, test_name), np.array(mem_usage))
    
def run_test(save_dir):
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  for block_length_fraction in [8]:
    for cycle_length in [128]:
      make_dataset_and_iter(save_dir,
      cycle_length=cycle_length,
      block_length_fraction=block_length_fraction,)

run_test('criteo_debugging_ram_usage_test_extended')
