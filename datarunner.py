import algorithmic_efficiency.workloads.criteo1tb.input_pipeline as input_pipeline
import time
import jax
import psutil
import tracemalloc
import linecache
import os 

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

import tensorflow as tf

tf.profiler.experimental.start('logdir')
print("getting dataset")
ds = input_pipeline.get_criteo1tb_dataset(split='train', 
                                        shuffle_rng=jax.random.PRNGKey(0), 
                                        data_dir='/home/kasimbeg/data/criteo1tb',
                                        num_dense_features=13,
                                        global_batch_size=int(65536/16)
)
print("iterating dataset")
print(f"Batch: {0}. RAM USED (GB) {psutil.virtual_memory()[3]/1000000000}")
for i in range(1000):
    next_batch = next(ds)
    tf.profiler.experimental.stop()
    if (i % 10 == 0):
        print(f"Batch: {i}. RAM USED (GB) {psutil.virtual_memory()[3]/1000000000}")
