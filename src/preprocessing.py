from __future__ import annotations
from typing import Any
import snowflake.snowpark as snowpark
from snowflake.snowpark import Session
import snowflake.snowpark.functions as F
import snowflake.snowpark.window as W
import snowflake.snowpark.types as T
from config import get_session
from transformers import MinMaxScaler,BinEncoder,OneHotEncoder

session = get_session()

df = session.sql('SELECT * FROM DATA_SCIENCE.EXPERIMENT.EXTENDED_ATTRIBUTES_CONSOLIDATED').cache_result()

#avg nulls
df = df.withColumn('AGE',F.coalesce(F.col('AGE'),F.mean(F.col('AGE')).over()))
df = df.withColumn('LENGTH_OF_RESIDENCE',F.coalesce(F.col('LENGTH_OF_RESIDENCE'),F.mean(F.col('LENGTH_OF_RESIDENCE')).over()))
df =df.cache_result()

columns_onehot_catagories = ['EDUCATION_LEVEL','HOUSEHOLD_COMPOSITION','OCCUPATION','DWELLING_TYPE','HOME_OWNERSHIP','MARITAL_STATUS']
columns_encode_catagories = ['AGE','LENGTH_OF_RESIDENCE','ESTIMATED_HOUSEHOLD_INCOME','HOME_MKT_VAL','NUMBER_OF_CHILDREN','CHILD_PRESENT','CHILD_UNDER_6_PRESENT','CHILD_6_10_PRESENT','CHILD_11_15_PRESENT','CHILD_16_17_PRESENT','EDUCATION_LEVEL','HOME_POOL','SENIOR_ADULT_IN_HH','SINGLE_PARENT','SPANISH_SPEAKING','USES_CREDIT_CARD','HOME_OFFICE']+list(filter(lambda c: "_INTEREST" in c,df.columns))
encode_dict = {
    'ESTIMATED_HOUSEHOLD_INCOME' : ['<$20,000','$50,000-$99,999','UNKNOWN','$100,000-249,999','$250,000+'],
    'HOME_MKT_VAL' : ['$1K - $24,999','$25K - $49,999','$50K - $74,999','$75K - $99,999','$100K - $124,999','$125K - $149,999','$150K - $174,999','$175K - $199,999','$200K - $224,999','UNKNOWN','$225K - $249,999','$250K - $274,999','$275K - $299,999','$300K - $349,999','$350K - $399,999','$400K - $449,999','$450K - $499,999','$500K - $749,999','$750K - $999,999','$1M+'],
    'NUMBER_OF_CHILDREN' : ['0','1-3','UNKNOWN','3-5','6+'],
    'CHILD_PRESENT' : ['NO','UNKNOWN','YES'],
    'CHILD_UNDER_6_PRESENT' : ['UNKNOWN','YES'],
    'CHILD_6_10_PRESENT' : ['UNKNOWN','YES'],
    'CHILD_11_15_PRESENT' : ['UNKNOWN','YES'],
    'CHILD_16_17_PRESENT' : ['UNKNOWN','YES'],
    'EDUCATION_LEVEL' : ['HIGH SCHOOL','VOCATIONAL/TECHNICAL SCHOOL','UNKNOWN','COLLEGE','GRADUATE SCHOOL'], # maybe onehot
    'HOME_POOL' : ['UNKNOWN','YES'],
    'SENIOR_ADULT_IN_HH' : ['UNKNOWN','YES'],
    'SINGLE_PARENT' : ['NO','UNKNOWN','YES'],
    'SPANISH_SPEAKING' : ['NO','UNKNOWN','YES'],
    'USES_CREDIT_CARD' : ['UNKNOWN','YES'],
    'HOME_OFFICE' : ['UNKNOWN','YES'],
}
columns_to_scale = []

def scale_column(df, col):
    """Scale the given column using MinMaxScaler."""
    print(f"{col}: scale begin")
    # Replace null values with 0
    df = df.withColumn(col, F.when(F.col(col).is_not_null(), F.col(col)).otherwise(0))
    df, col_scaled = MinMaxScaler(df, col, f"{col}_scale")
    df = df.cache_result()
    print(f"{col}: scale end")
    return df, col_scaled

def encode_column(df, col, encode_dict):
    """Encode the given column using BinEncoder and then scale the encoded values using MinMaxScaler."""
    print(f"{col}: encode begin")
    df, col_encoded = BinEncoder(df, col, f"{col}_encode", valueOrder=encode_dict.get(col))
    df, col_scaled = MinMaxScaler(df, col_encoded, col_encoded)
    df = df.cache_result()
    print(f"{col}: encode end")
    return df, col_scaled

def onehot_column(df, col):
    """One-hot encode the given column."""
    print(f"{col}: onehot begin")
    df, cols_onehot = OneHotEncoder(df, col, f"{col}_onehot")
    df = df.cache_result()
    print(f"{col}: onehot end")
    return df, cols_onehot

# Initialize list of columns to keep in the final dataframe
cols = ['UNIVERSAL_ID']

# Scale each column in columns_to_scale
for c in columns_to_scale:
    df, col = scale_column(df, c)
    cols.append(col)

# Encode and scale each column in columns_encode_catagories
for c in columns_encode_catagories:
    df, col = encode_column(df, c, encode_dict)
    cols.append(col)

# One-hot encode each column in columns_onehot_catagories
for c in columns_onehot_catagories:
    df, cols_onehot = onehot_column(df, c)
    cols.extend(cols_onehot)

# Keep only the specified columns in the final dataframe
df = df[cols].cache_result()


#df.write.mode("overwrite").save_as_table('EXTENDED_ATTRIBUTES_CONSOLIDATED_SAMPLE')
df.write.mode("overwrite").save_as_table('DATA_SCIENCE.PUBLIC.EXTENDED_ATTRIBUTES_CONSOLIDATED_FULL')