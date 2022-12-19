import pandas as pd
import urllib.request
import argostranslate.package

def download_install_argos_model(from_code, to_code):
    argos_models = pd.read_json('https://github.com/argosopentech/argospm-index/raw/main/index.json')
    argos_models = argos_models[argos_models.to_code == to_code]
    argos_models = argos_models[argos_models.from_code == from_code]
    argos_link = argos_models.iloc[0].links[0]
    argos_model_name = argos_link.split('/')[-1]
    urllib.request.urlretrieve(argos_link, argos_model_name)
    argostranslate.package.install_from_path(argos_model_name)

if __name__ == '__main__':
    for lang in ['de', 'es', 'nl', 'fr', 'pt', 'it', 'ja', 'ru', 'id', 'sv', 'pl']:
        from_code = lang
        to_code = 'en'
        download_install_argos_model(from_code, to_code)