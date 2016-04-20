import os
import psutil


def memory_usage(format='bytes'):
    """Returns the memory usage in the given format."""
    process = psutil.Process(os.getpid())
    usage_bytes = process.memory_full_info().uss

    for child_process in process.children(recursive=True):
        usage_bytes += child_process.memory_full_info().uss

    if format == 'bytes':
        return usage_bytes
    elif format == 'kb' or format == 'kilobytes':
        return usage_bytes / 1000
    elif format == 'mb' or format == 'megabytes':
        return usage_bytes / 1000000
    elif format == 'gb' or format == 'gigabytes':
        return usage_bytes / 1000000000

    raise ValueError('Invalid format')
