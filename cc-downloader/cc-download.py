import pandas as pd
import boto3
from warcio.archiveiterator import ArchiveIterator
import time
from io import BytesIO
from bs4 import BeautifulSoup, SoupStrainer
import argparse
import awswrangler as wr
import os
import json
from textblob import TextBlob
from urllib.request import urlopen

from passage_extraction import PassageExtractor
from problem_classification import ProblemClassifier
from utils import *

def replace_newlines(text):
    """ Replace newlines with periods to split sentence on newlines """
    return text.replace('\n', '.')

def replace_carriage_returns(text):
    """ Replace carriage returns with spaces """
    return text.replace('\r', ' ')

def replace_xa0(text):
    """ Replace non-breaking space with space """
    return text.replace(u'\xa0', u' ')

def replace_multiple_spaces(text):
    """ Replace multiple spaces with single space """
    return ' '.join(text.split())

def replace_all(text):
    text = replace_newlines(text)
    text = replace_carriage_returns(text)
    text = replace_xa0(text)
    text = replace_multiple_spaces(text)

    return text

def fetch_process_warc_records(row, s3client, keywords):
    """Fetch all WARC records defined by filenames and offsets in batch,
    parse the records and the contained HTML, return all paragraphs containing at least one of the
    keywords.csv"""

    only_paragraphs = SoupStrainer('p')

    warc_path, offset, end = row.warc_filename, str(row.warc_record_offset), str(row.warc_record_end)

    rangereq = 'bytes={}-{}'.format(offset, end)

    response = exponential_backoff(s3client.get_object, Bucket='commoncrawl', Key=warc_path, Range=rangereq)

    record_stream = BytesIO(response["Body"].read())

    extracts = []
    for record in ArchiveIterator(record_stream):
        page = record.content_stream().read()

        soup = BeautifulSoup(page, 'lxml', parse_only=only_paragraphs)

        text = soup.get_text()

        text = replace_all(text)

        extractor = PassageExtractor(text, keywords)
        extracts += extractor.extract_relevant_passages()

    return list(set(extracts)) # remove duplicates
    # return list(set([replace_newlines(extract) for extract in extracts])) # remove duplicates and replace newlines



if __name__ == "__main__":
    # print('Installing packages...')
    # for package in ['argostranslate', 'langdetect']:
    #     install_import(package)
    # print('Packages installed.')
    #
    # # download textblob corpora
    # print('Downloading textblob corpora...')
    # os.system('python -m textblob.download_corpora lite')
    # print('Corpora downloaded.')
    #
    # # download nltk corpora
    # print('Downloading nltk corpora...')
    # nltk.download('omw-1.4')
    # print('nltk corpora downloaded.')


    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--batches_per_partition", type=int, required=True)
    parser.add_argument("--output_bucket", type=str, required=True)
    parser.add_argument("--result_output_path", type=str, required=True)
    parser.add_argument("--keywords_path", type=str, required=True) # default='https://github.com/jakob-ra/cc-download/raw/main/cc-download/keywords.csv')
    parser.add_argument("--topic_keywords_path", type=str, required=True)
    args = parser.parse_args()
    # args = parser.parse_args(['--batch_size', '10000', '--batches_per_partition', '1', '--output_bucket',
    #                           'cc-extract', '--result_output_path', 'test', '--keywords_path',
    #                           'https://github.com/jakob-ra/cc-download-translate/raw/main/keywords.csv',
    #                           '--topic_keywords_path', 'https://github.com/jakob-ra/cc-download-translate/raw/main/topic_keywords.json'])

    keywords = pd.read_csv(args.keywords_path).squeeze().tolist()

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
    partition_n = batch_n//args.batches_per_partition + 1
    batch_n_within_partition = batch_n%args.batches_per_partition
    query = f'SELECT * FROM urls_merged_cc_to_download WHERE partition={partition_n} ORDER BY crawl, url_host_tld, fetch_time OFFSET {batch_n_within_partition*args.batch_size} LIMIT {args.batch_size}'
    # query = f'SELECT * FROM urls_merged_cc_to_download WHERE partition={batch_n+1}' # +1 because partitions start at 1 not 0

    df = exponential_backoff(wr.athena.read_sql_query, sql=query, database='ccindex', boto3_session=session)
    # df = wr.athena.read_sql_query(sql=query, database='ccindex', boto3_session=session)
    assert len(df) > 1, "Empty input table!"

    # get unique crawl ids
    crawl_ids = df.crawl.unique()
    crawl_dates = ['-'.join(crawl.split('-')[-2:]) for crawl in crawl_ids]
    crawls_name = '_'.join(crawl_dates)

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



