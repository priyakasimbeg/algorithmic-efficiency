import psutil
from absl import logging

DEBUG_FILE = '/home/kasimbeg/debug_log.txt'
def log_mem_usage(event):
    message = f"{event}: RAM USED (GB) {psutil.virtual_memory()[3]/1000000000}\n"
    logging.info(message)
    with open (DEBUG_FILE, 'a') as f:
        f.write(message)

