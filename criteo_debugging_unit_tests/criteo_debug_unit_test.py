import jax
import tensorflow as tf
import psutil
from typing import Optional


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

def generate_example_fn():
  """Parser function for pre-processed Criteo TSV records."""
  num_dense_features=13
  batch_size = 524_288
  label_defaults = [[0.0]]
  int_defaults = [[0.0] for _ in range(num_dense_features)]
  categorical_defaults = [['00000000'] for _ in range(len(_VOCAB_SIZES))]
  record_defaults = label_defaults + int_defaults + categorical_defaults
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

  while True:
    yield features


def get_criteo1tb_dataset_from_generator(split: str,
                          shuffle_rng,
                          data_dir: str,
                          num_dense_features: int,
                          global_batch_size: int,
                          num_batches: Optional[int] = None,
                          repeat_final_dataset: bool = False):
  """Get the Criteo 1TB dataset for a given split."""
  ds = tf.data.Dataset.from_generator(generate_example_fn,
                                      output_types={'inputs':tf.float32, 'targets':tf.float32},)
  # ds = ds.map(parse_fn, num_parallel_calls=16)
  is_training=True
  if is_training:
    ds = ds.repeat()
  ds = ds.prefetch(10)

  if num_batches is not None:
    ds = ds.take(num_batches)

  # We do not use ds.cache() because the dataset is so large that it would OOM.
  if repeat_final_dataset:
    ds = ds.repeat()


  return ds


print("Getting dataset")
ds = get_criteo1tb_dataset_from_generator(split='train', 
                                        shuffle_rng=jax.random.PRNGKey(0), 
                                        data_dir='/home/kasimbeg/data/criteo1tb',
                                        num_dense_features=13,
                                        global_batch_size=int(65536*8))
ds = iter(ds)
print("Iterating dataset")
print(f"Batch: {0}. RAM USED (GB) {psutil.virtual_memory()[3]/1000000000}")
for i in range(1000):
    next_batch = next(ds)
    if (i % 100 == 0):
        print(f"Batch: {i}. RAM USED (GB) {psutil.virtual_memory()[3]/1000000000}")