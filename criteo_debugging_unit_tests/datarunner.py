import algorithmic_efficiency.workloads.criteo1tb.input_pipeline as input_pipeline
import time
import jax
import psutil
import os 
import tensorflow as tf

print("Getting dataset")
ds = input_pipeline.get_criteo1tb_dataset(split='train', 
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
ta