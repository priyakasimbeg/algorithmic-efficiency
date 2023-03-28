import algorithmic_efficiency.workloads.criteo1tb.input_pipeline as input_pipeline
import time
import jax
import psutil
import tracemalloc
import linecache
import os 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorboard

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((tracemalloc.Filter(False, "<frozen importlib._bootstrap>"), tracemalloc.Filter(False, "<unknown>"),))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB" % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

        other = top_stats[limit:]
        if other:
            size = sum(stat.size for stat in other)
            print("%s other: %.1f KiB" % (len(other), size / 1024))
            total = sum(stat.size for stat in top_stats)
            print("Total allocated size: %.1f KiB" % (total / 1024))

logdir = '/home/kasimbeg/logdir'
writer = tf.summary.create_file_writer(logdir)

model = keras.Sequential(
    [
        layers.Dense(200, activation="relu", name="layer1"),
        layers.Dense(1, activation="relu", name="layer2"),
    ])

model.compile(
    loss='sigmoid_binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

print("getting dataset")
ds = input_pipeline.get_criteo1tb_dataset(split='train', 
                                        shuffle_rng=jax.random.PRNGKey(0), 
                                        data_dir='/home/kasimbeg/data/criteo1tb',
                                        num_dense_features=13,
                                        global_batch_size=int(65537)
)

tf.profiler.experimental.start(logdir)

ds = iter(ds)
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir,
                                                 histogram_freq = 1,
                                                 profile_batch = '500,520')

ds_x = ds.map(lambda batch: batch['inputs'])
ds_y = ds.map(lambda batch: batch['targets'])

model.fit(x=ds_x,
          y=ds_y,
          steps_per_epoch=5,
          epochs=1,
          callbacks = [tboard_callback])

tf.profiler.experimental.stop()

# print("iterating dataset")
# print(f"Batch: {0}. RAM USED (GB) {psutil.virtual_memory()[3]/1000000000}")
# batch = next(ds)

# tf.profiler.experimental.start(logdir)

# for i in range(11):
#     next_batch = next(ds)
#     # y = model(next_batch['inputs'])
#     if (i % 1 == 0):
#         print(f"Batch: {i}. RAM USED (GB) {psutil.virtual_memory()[3]/1000000000}")

# tf.profiler.experimental.stop()
