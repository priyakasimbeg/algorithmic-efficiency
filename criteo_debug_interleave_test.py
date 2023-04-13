import jax
import tensorflow as tf
import psutil
from typing import Optional
import os
import functools
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_CYCLE_LENGTH = 128
DEFAULT_BLOCK_LENGTH = 865536
BATCH_SIZE = 524288

def generate_dataset_fn():
  """Parser function for pre-processed Criteo TSV records."""
  fields = [1.0] * 40  
  dataset = tf.data.Dataset.from_tensor_slices(fields).repeat(5_000_000)
  while True:
    yield dataset

def get_fake_dataset(cycle_length=DEFAULT_CYCLE_LENGTH, block_length=DEFAULT_BLOCK_LENGTH,):
  ds = tf.data.Dataset.from_tensor_slices([1] * 847)
  ds = ds.interleave(lambda x: tf.data.Dataset.from_generator(generate_dataset_fn,
                                                              output_signature=tf.data.DatasetSpec(tf.TensorSpec(shape=(), dtype=tf.float32, name=None), tf.TensorShape([]))),
                                  cycle_length=cycle_length,
                                  block_length=block_length)
  ds = ds.batch(BATCH_SIZE)
  return ds

def get_ram_usage():
    return psutil.virtual_memory()[3]/1000000000


def make_dataset_and_iter(save_dir,
                          cycle_length: int = DEFAULT_CYCLE_LENGTH,
                          block_length: int = DEFAULT_BLOCK_LENGTH,):
    test_name = (f'_cycle_length_{cycle_length}_block_length_{block_length}')
    print(f'Running test {test_name}')
    mem_usage = []
    ds = iter(get_fake_dataset(cycle_length=cycle_length,
                               block_length=block_length,))

    # Iterate and log memory usage  
    m = get_ram_usage()                          
    mem_usage.append(m)
    print(f"Batch: {0}: RAM USED (GB) {m}")
    for i in range(1, 1001):
        next_batch = next(ds)
        if (i % 100 == 0) or (i == 1):
            m = get_ram_usage()
            mem_usage.append(m)
            print(f"Batch: {i}. RAM USED (GB) {m}")
     
    np.save(os.path.join(save_dir, test_name), np.array(mem_usage))
    
def run_test(save_dir):
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  for cycle_length in [128, 64]:
    for block_length in [DEFAULT_BLOCK_LENGTH, int(DEFAULT_BLOCK_LENGTH/2)]:
          make_dataset_and_iter(save_dir,
          cycle_length=cycle_length,
          block_length=block_length)

run_test('criteo_debugging_ram_usage_unit_test')
# for i in generate_example_fn():
#   print(i)
#   break