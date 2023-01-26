from botocore.exceptions import ClientError
import time
import pandas as pd
import urllib.request
import subprocess
import sys
import os
from langdetect import detect
import random

def exponential_backoff(func, *args, **kwargs):
    """Exponential backoff to deal with request limits"""
    delay = 1  # initial delay
    delay_incr = 1  # additional delay in each loop
    max_delay = 10  # max delay of one loop. Total delay is (max_delay**2)/2 plus random jitter

    while delay < max_delay:
        try:
            return func(*args, **kwargs)
        except ClientError:
            # add random delay to avoid hitting the limit again
            time.sleep(delay + random.random())
            delay += delay_incr
    else:
        raise

def install_import(package):
    def install_package(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    try:
        __import__(package)
    except ImportError as e:
        print(e)
        install_package(package)
        __import__(package)



