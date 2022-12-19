#FROM bitnami/pytorch:1.13.0

FROM python:3.10.7

USER root

COPY cc-downloader .
COPY requirements.txt .
#COPY download_install_argos_models.py .

RUN pip install --no-cache-dir -r requirements.txt

# download textblob and nltk data
RUN python -m textblob.download_corpora lite
RUN python -c "import nltk; nltk.download('omw-1.4')"

# download and install argos models
#RUN python download_install_argos_models.py

#CMD python cc-download.py --batch_size=100 --output_bucket=cc-extract --result_output_path=cc-download-test --keywords='https://github.com/jakob-ra/cc-downloader/raw/main/keywords.csv' --topic_keywords='https://github.com/jakob-ra/cc-download-translate/raw/main/topic_keywords.json'

## the command is submitted directly to AWS batch as a job definition:
# python cc-downloader/cc-download.py --batch_size=100 --output_bucket=cc-extract --result_output_path=cc-download-test --keywords='https://github.com/jakob-ra/cc-downloader/raw/main/keywords.csv' --topic_keywords='https://github.com/jakob-ra/cc-download-translate/raw/main/topic_keywords.json'


## to push the dockerfile to Amazon Elastic Container Registry:
# start docker daemon
# docker build -t public.ecr.aws/r9v1u7o6/cc-download-translate:latest .
# docker tag cc-download:latest public.ecr.aws/r9v1u7o6/cc-download:latest
# docker tag cc-download:latest public.ecr.aws/r9v1u7o6/cc-download-translate:latest
# aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
# docker push public.ecr.aws/r9v1u7o6/cc-download:latest
# docker push public.ecr.aws/r9v1u7o6/cc-download-translate:latest

