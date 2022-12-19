import pandas as pd
import os

## Credentials
# make IAM role with AWSBatchFullAccess, AmazonEC2FullAccess, AmazonAthenaFullAccess, AmazonS3FullAccess, AmazonElasticContainerRegistryPublicFullAccess
def aws_config_credentials(credentials_csv_filepath, region, profile_name='default'):
    os.path.split(credentials_csv_filepath)
    credentials = pd.read_csv(credentials_csv_filepath)
    # user_name = os.path.split(credentials_csv_filepath)[-1].split('_accessKeys')[0] # do not rename credentials csv downloaded from IAM as the user name is inferred from the filename
    credentials['User Name'] = [profile_name]
    credentials.to_csv(credentials_csv_filepath, index=False)
    os.system(f'aws configure import --csv file://{credentials_csv_filepath}')
    os.system(f'aws configure set region {region}')
    os.system(f'aws configure set output json')

    # credentials = credentials.T.squeeze().to_dict()