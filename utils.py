import psutil
import time

def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024 * 1024)  # Convert bytes to MB

def monitor_memory_usage(interval, stop_event, max_memory_usage):
    current_max = 0
    while not stop_event.is_set():
        memory_usage = get_memory_usage()
        if memory_usage > current_max:
            current_max = memory_usage
        time.sleep(interval)
    max_memory_usage.value = current_max