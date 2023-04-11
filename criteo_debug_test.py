import jax
import tensorflow as tf
import psutil
from typing import Optional
import os
import functools
import matplotlib.pyplot as plt

_NUM_DAY_23_FILES = 36


_VOCAB_SIZES = [
    39884406,
    39043,
    17289,
    7420,
    20263,
    3,
    7120,
    1543,
    63,
    38532951,
    2953546,
    403346,
    10,
    2208,
    11938,
    155,
    4,
    976,
    14,
    39979771,
    25641295,
    39664984,
    585935,
    12972,
    108,
    36,
]

@tf.function
def _parse_example_fn(num_dense_features, example):
  """Parser function for pre-processed Criteo TSV records."""
  label_defaults = [[0.0]]
  int_defaults = [[0.0] for _ in range(num_dense_features)]
  categorical_defaults = [['00000000'] for _ in range(len(_VOCAB_SIZES))]
  record_defaults = label_defaults + int_defaults + categorical_defaults
  # fields = tf.io.decode_csv(
  #     example, record_defaults, field_delim='\t', na_value='-1')
  batch_size = 524_288
  fields = [tf.constant([1.0] * batch_size)] * 14 + [tf.constant(['1'] * batch_size)] * 26
  num_labels = 1
  features = {}
  features['targets'] = tf.reshape(fields[0], (-1,))

  int_features = []
  for idx in range(num_dense_features):
    positive_val = tf.nn.relu(fields[idx + num_labels])
    int_features.append(tf.math.log(positive_val + 1))
  int_features = tf.stack(int_features, axis=1)

  cat_features = []
  for idx in range(len(_VOCAB_SIZES)):
    field = fields[idx + num_dense_features + num_labels]
    # We append the column index to the string to make the same id in different
    # columns unique.
    cat_features.append(
        tf.strings.to_hash_bucket_fast(field + str(idx), _VOCAB_SIZES[idx]))
  cat_features = tf.cast(
      tf.stack(cat_features, axis=1), dtype=int_features.dtype)
  features['inputs'] = tf.concat([int_features, cat_features], axis=1)
  return features


def get_criteo1tb_dataset(split: str,
                          shuffle_rng,
                          data_dir: str,
                          num_dense_features: int,
                          global_batch_size: int,
                          num_batches: Optional[int] = None,
                          repeat_final_dataset: bool = False,
                          shuffle: bool = True,
                          interleave: bool = True,
                          num_parallel_calls_interleave: int = 16,
                          cycle_length: int = 128,
                          block_length_fraction: int = 8,
                          text_line_ds_buffer_size: int = 256 * 1024,
                          num_files: int = 849,
                          prefetch: int = 10,):
  """Get the Criteo 1TB dataset for a given split."""
  files = os.listdir(data_dir)
  train_files = [os.path.join(data_dir, f) for f in files if int(f.split("_")[1]) in range(0,23)]
  train_files = train_files[:num_files]

  ds = tf.data.Dataset.list_files(
      train_files, shuffle=shuffle, seed=shuffle_rng[0])

  if interleave: 
    ds = ds.interleave(
        tf.data.TextLineDataset,
        cycle_length=cycle_length,
        block_length=global_batch_size // block_length_fraction,
        num_parallel_calls=num_parallel_calls_interleave,
        deterministic=False)

  else:
     ds = tf.data.TextLineDataset(train_files, buffer_size= test_line_ds_buffer_size)

  if shuffle:
    ds = ds.shuffle(buffer_size=524_288 * 100, seed=shuffle_rng[1])

  ds = ds.batch(global_batch_size, drop_remainder=True)

  parse_fn = functools.partial(_parse_example_fn, num_dense_features)

  ds = ds.repeat()
  ds = ds.prefetch(prefetch)

  if num_batches is not None:
    ds = ds.take(num_batches)

  if repeat_final_dataset:
    ds = ds.repeat()

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
              text_line_ds_buffer_size: int = 256 * 1024,
              num_files: int = 849,
              prefetch: int = 10,):
    test_name = (f's_{shuffle}_i_{interleave}_p_{num_parallel_calls_interleave}'
                f'_c_{cycle_length}_block_{block_length_fraction}'
                f'_buff_{text_line_ds_buffer_size}_num_files_{num_files}'
                f'_prefetch_{prefetch}')
    print(f'Running test {test_name}')
    mem_usage = []
    ds = iter(get_criteo1tb_dataset(split='train', 
                                shuffle_rng=jax.random.PRNGKey(0), 
                                data_dir='/home/kasimbeg/data/criteo1tb',
                                num_dense_features=13,
                                global_batch_size=global_batch_size,
                                shuffle=shuffle,
                                interleave=interleave,
                                cycle_length=cycle_length,
                                block_length_fraction=block_length_fraction,
                                text_line_ds_buffer_size=text_line_ds_buffer_size,
                                num_files=num_files,
                                prefetch=prefetch))

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
    
    np.save(os.path.join(save_dir, test_name, mem_usage))
    
def run_test(save_dir):
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  for global_batch_size in [65536*8, 65536*4]:
    for interleave in [True, False]:
      for cycle_length in [128, 64]:
        for block_length_fraction in [8, 16]:
          for text_line_ds_buffer_size in [256 * 1024, 128 * 1024]:
            for num_files in [849, 400]:
              for prefetch in [10, 5]:
                make_dataset_and_iter(save_dir,
                global_batch_size=global_batch_size,
                interleave=interleave,
                cycle_length=cycle_length,
                block_length_fraction=block_length_fraction,
                text_line_ds_buffer_size=text_line_ds_buffer_size,
                num_files=num_files,
                prefetch=prefetch,)

run_test('criteo_debugging_ram_usage_test')