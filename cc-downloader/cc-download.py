import pandas as pd
import boto3
import time
import argparse
import awswrangler as wr
import os
import json
from textblob import TextBlob
from urllib.request import urlopen
from utils import *

if __name__ == "__main__":
    # print('Installing packages...')
    # for package in ['argostranslate', 'langdetect']:
    #     install_import(package)
    # print('Packages installed.')


    parser = argparse.ArgumentParser()
    parser.add_argument("--output_bucket", type=str, required=True)
    parser.add_argument("--result_output_path", type=str, required=True)
    args = parser.parse_args()

    if "AWS_BATCH_JOB_ARRAY_INDEX" in os.environ:
        batch_n = os.environ['AWS_BATCH_JOB_ARRAY_INDEX']
        batch_n = int(batch_n)
        print(f'Processing batch {batch_n}.')
    else:
        batch_n = 0
        print('Processing first batch (no array index found).')

    session = boto3.Session(region_name='us-east-1')

    sts = session.client("sts")
    print(sts.get_caller_identity())

    # read cc-index table with warc filenames and byte positions
    partition_n = batch_n
    query = f'SELECT * FROM non_english_unique_paragraphs WHERE partition={partition_n} ORDER BY LENGTH(paragraph) OFFSET {0} LIMIT {1000}'

    df = exponential_backoff(wr.athena.read_sql_query, sql=query, database='ccindex', boto3_session=session)

    query = f'SELECT * FROM non_english_unique_paragraphs WHERE partition={1} ORDER BY LENGTH(paragraph) OFFSET {0} LIMIT {1000}'

    df = wr.athena.read_sql_query(sql=query, database='ccindex', boto3_session=session)


    # df = pd.read_parquet('s3://cc-extract/cc-download-problems-sentiment/non_english/2020-05_0.parquet')

    assert len(df) > 1, "Empty input table!"

    # initialize s3
    s3client = boto3.client('s3', region_name='us-east-1', use_ssl=False)

   # download paragraphs and fill into new column
    print('Starting download...')
    start = time.process_time()
    df['paragraphs'] = df.apply(lambda row: fetch_process_warc_records(row, s3client, keywords), axis=1)
    print(f'Success! Finished downloading in {time.process_time() - start} seconds.')
    print(f'Share of domains mentioning at least one keyword: {df.groupby("url_host_registered_domain").paragraphs.apply(lambda x: len(x[x.str.len()>0]) > 0).mean()}')
    print(f'Share of subpages mentioning at least one keyword: {len(df.paragraphs[df.paragraphs.str.len()>0])/len(df)}')

    # drop offsets
    df.drop(columns=['warc_filename', 'warc_record_offset', 'warc_record_end'], inplace=True)

    # save domains without any mentions of keywords
    domains_without_mentions = df[df.paragraphs.str.len() == 0][['url_host_registered_domain', 'url_host_tld', 'crawl', 'fetch_time']].drop_duplicates(subset=['url_host_registered_domain', 'crawl'])
    s3path = f's3://{args.output_bucket}/{args.result_output_path}/domains_without_mentions/{crawls_name}_{batch_n}.parquet'
    wr.s3.to_parquet(df=domains_without_mentions, path=s3path, index=False, compression='gzip')

    # continue with non-empty domains
    df = df[df.paragraphs.str.len() > 0].copy(deep=True)

    # detect language on first characters of first paragraph
    print('Starting language detection...')
    start = time.process_time()
    df['lang'] = df.paragraphs.str[0].str.strip().str[:50].apply(detect_lang)
    print(f'Success! Finished language detection in {time.process_time() - start} seconds.')

    # explode so we have one paragraph per row
    df = df.explode('paragraphs')
    df.reset_index(drop=True, inplace=True)
    df.rename(columns={'paragraphs': 'paragraph'}, inplace=True)
    df['paragraph'] = df.paragraph.str.strip()
    df = df[df.paragraph.str.len() > 20].copy(deep=True) # drop very short paragraphs

    # save non-english pages to S3
    non_english = df[df.lang != 'en']
    if len(non_english) > 0:
        s3path = f's3://{args.output_bucket}/{args.result_output_path}/non_english/{crawls_name}_{batch_n}.parquet'
        wr.s3.to_parquet(df=non_english, path=s3path, index=False, compression='gzip')

    # continue with english pages
    df = df[df.lang == 'en'].copy(deep=True)

    # translation
    # print('Starting translation...')
    # start = time.process_time()
    # df['translated_paragraphs'] = np.nan
    # to_code = 'en'
    # langs = ['de', 'es', 'nl', 'fr', 'pt', 'it', 'ja', 'ru', 'id', 'sv', 'pl']
    # lang_counts = df.lang.value_counts(normalize=True)
    # nonrare_langs = lang_counts[lang_counts > 0.01].index.tolist()
    # langs = [l for l in nonrare_langs if l in langs]
    # for from_code in langs:
    #     print(f'Downloading model {from_code}-{to_code}...')
    #     model_path = download_argos_model(from_code, to_code)
    #     install_argos_model(model_path)
    #     print(f'Loading model {from_code}-{to_code}...')
    #     model = load_argos_model(from_code, to_code)
    #     print(f'Translating {len(df[df.lang == from_code])} paragraphs from {from_code} to {to_code}...')
    #     df.loc[df.lang == from_code, 'translated_paragraphs'] = df[df.lang == from_code].paragraphs.apply(lambda text: argos_translate(model, text))
    # df['translated_paragraphs'] = df.translated_paragraphs.astype(str).str.strip()
    # print(f'Success! Finished translation in {time.process_time() - start} seconds.')

    if len(df) > 0:
        # problem classification
        print('Starting problem classification...')
        start = time.process_time()
        with urlopen(args.topic_keywords_path) as url:
            topic_keywords = json.load(url)
        problem_classifier = ProblemClassifier(topic_keywords)
        df = pd.concat([df, df['paragraph'].apply(problem_classifier.classify).apply(pd.Series)], axis=1)
        print(f'Success! Finished problem classification in {time.process_time() - start} seconds.')

        # sentiment analysis
        print('Starting sentiment analysis...')
        start = time.process_time()
        df['sentiment'] = df.paragraph.apply(str).apply(lambda x: TextBlob(x).sentiment.polarity)
        print(f'Success! Finished sentiment analysis in {time.process_time() - start} seconds.')

        s3path = f's3://{args.output_bucket}/{args.result_output_path}/english/{crawls_name}_{batch_n}.parquet'
        wr.s3.to_parquet(df=df, path=s3path, index=False, compression='gzip')



