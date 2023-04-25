import psutil
from absl import logging
import datetime

now = datetime.datetime.now().strftime('-%Y-%m-%d-%H-%M-%S')
DEBUG_FILE = f'/logs/memory_log{now}.txt'
def log_mem_usage(event):
    message = f"{event}: RAM USED (GB) {psutil.virtual_memory()[3]/1000000000}\n"
    logging.info(message)
    with open (DEBUG_FILE, 'a') as f:
        f.write(message)

