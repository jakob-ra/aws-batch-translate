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

def detect_lang(text: str) -> str:
    try:
        return detect(text)
    except:
        return None

# import argostranslate.package
# import argostranslate.translate
#
# def download_argos_model(from_code, to_code):
#     argos_models = pd.read_json('https://github.com/argosopentech/argospm-index/raw/main/index.json')
#     argos_models = argos_models[argos_models.to_code == to_code]
#     argos_models = argos_models[argos_models.from_code == from_code]
#     argos_link = argos_models.iloc[0].links[0]
#     argos_model_name = argos_link.split('/')[-1]
#     urllib.request.urlretrieve(argos_link, argos_model_name)
#     model_path = os.path.join(os.getcwd(), argos_model_name)
#
#     return model_path
#
# def install_argos_model(model_path):
#     argostranslate.package.install_from_path(model_path)
#
# def load_argos_model(from_code, to_code):
#     installed_languages = argostranslate.translate.get_installed_languages()
#     from_lang = list(filter(lambda x: x.code == from_code, installed_languages))[0]
#     to_lang = list(filter(lambda x: x.code == to_code, installed_languages))[0]
#     model = from_lang.get_translation(to_lang)
#
#     return model
#
# def argos_translate(model, text):
#     try:
#         return model.translate(text)
#     except:
#         return None




