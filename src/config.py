import boto3
from botocore.exceptions import ClientError
import snowflake.snowpark as snowpark
from snowflake.snowpark import Session
import json

def get_secret():

    secret_name = "snowpark_config"
    region_name = "us-west-2"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Decrypts secret using the associated KMS key.
    secret = get_secret_value_response['SecretString']
    print(secret)
    return secret
def get_session():
    conn_params = None
    with open('./SNOWPARK_CREDS.json', 'r') as f:
        conn_params = json.load(f)
        
    connection_parameters = {
        "account": conn_params['account'],
        "user": conn_params['user'], 
        "password": conn_params['password'],
        "role": conn_params['role'],
        "warehouse": conn_params['warehouse'], 
        "database": "DATA_SCIENCE",
        "schema":"PUBLIC"
    }
    return Session.builder.configs(connection_parameters).create()
if __name__ == "__main__" :
    try:
        session = get_session()
        session.sql("select 1/2")
        print('success')
    except Exception as e:
        print('failed')
        raise e
    
    