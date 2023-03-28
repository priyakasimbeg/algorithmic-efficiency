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

logdir = '/home/kasimbeg/logdir_keras_prefetch'
writer = tf.summary.create_file_writer(logdir)

model = keras.Sequential(
    [
        layers.Dense(200, activation="relu", name="layer1"),
        layers.Dense(1, activation="relu", name="layer2"),
    ])

model.compile(
    loss='binary_crossentropy',
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

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir,
                                                 histogram_freq = 1,
                                                 profile_batch = '100,150')

ds = ds.map(lambda batch: (batch['inputs'], batch['targets']))

model.fit(x=ds,
          steps_per_epoch=100,
          epochs=3,
          callbacks = [tboard_callback])

