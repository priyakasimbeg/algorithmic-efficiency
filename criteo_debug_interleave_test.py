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
NUM_FILES = 849
TEST_DATA_DIR = 'test_data'

def generate_test_data_file(save_path, num_lines=5000_000):
  if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))
  
  if not os.path.exists(save_path):
    with open(save_path, 'w') as f:
      for i in range(num_lines):
        f.write('1       5       110             16              1       0       14      7       1               306             62770d79        e21f5d58        afea442f        945c7fcf        38b02748       6fcd6dcb        3580aa21        28808903        46dedfa6        2e027dc1        0c7c4231        95981d1f        00c5ffb7        be4ee537        8a0b74cc        4cdc3efa      d20856aa b8170bba        9512c20b        c38e2f28        14f65a5d        25b1b089        d7c1fc0b        7caf609c        30436bfc        ed10571d\n')
  return 

def generate_dataset_fn():
  """Parser function for pre-processed Criteo TSV records."""
  fields = [1.0] * 40  
  dataset = tf.data.Dataset.from_tensor_slices(fields).repeat(5_000_000)
  while True:
    yield dataset

def get_fake_dataset(cycle_length=DEFAULT_CYCLE_LENGTH, 
                    block_length=DEFAULT_BLOCK_LENGTH,
                    source=None,):
  if source == None:
    print("Getting dataset from generator")
    ds = tf.data.Dataset.from_tensor_slices([1] * 847)
    ds = ds.interleave(lambda x: tf.data.Dataset.from_generator(generate_dataset_fn,
                                                              output_signature=tf.data.DatasetSpec(tf.TensorSpec(shape=(), dtype=tf.float32, name=None), tf.TensorShape([]))),
                                  cycle_length=cycle_length,
                                  block_length=block_length)
  else:
    print("Getting dataset from test_data")
    data_dir = '/home/kasimbeg/data/criteo1tb'
    files = os.listdir(data_dir)
    train_files = [os.path.join(data_dir, f) for f in files if int(f.split("_")[1]) in range(0,23)]
    train_files = train_files[:NUM_FILES]
    ds = tf.data.Dataset.list_files(train_files)
    # ds = tf.data.Dataset.list_files([source] * NUM_FILES)
    ds = ds.interleave(tf.data.TextLineDataset, 
                       cycle_length=cycle_length,
                       block_length=block_length)
  ds = ds.batch(BATCH_SIZE)
  return ds

def get_ram_usage():
    return psutil.virtual_memory()[3]/1000000000


def make_dataset_and_iter(save_dir,
                          cycle_length: int = DEFAULT_CYCLE_LENGTH,
                          block_length: int = DEFAULT_BLOCK_LENGTH,
                          source: str = None):
    test_name = (f'_cycle_length_{cycle_length}_block_length_{block_length}')
    print(f'Running test {test_name}')
    mem_usage = []
    ds = iter(get_fake_dataset(cycle_length=cycle_length,
                               block_length=block_length,
                               source=source,))

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
     
    np.savetxt(os.path.join(save_dir, f'{test_name}.csv'), np.array(mem_usage), delimiter=',')
    
def run_test(save_dir, source=None):
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  for cycle_length in [128, 64]:
    for block_length in [DEFAULT_BLOCK_LENGTH, int(DEFAULT_BLOCK_LENGTH/2)]:
          make_dataset_and_iter(save_dir,
          cycle_length=cycle_length,
          block_length=block_length,
          source=source)

print('Generating data')
source_path = os.path.join(TEST_DATA_DIR, 'criteo_dummy_data')
generate_test_data_file(save_path=source_path)
print('Running test')
run_test(save_dir='criteo_debugging_ram_usage_test_fake_data',
         source=source_path)
# for i in generate_example_fn():
#   print(i)
#   break