from __future__ import annotations
from typing import Any
import snowflake.snowpark as snowpark
from snowflake.snowpark import Session
import snowflake.snowpark.functions as F
import snowflake.snowpark.window as W
import snowflake.snowpark.types as T

def BinEncoder(df: snowpark.DataFrame ,inputCol: str ,outputCol: str,valueOrder : list=None):
    types = None
    if valueOrder == None:
        #types = df.groupBy(F.col(inputCol)).agg(F.row_number().over(W.Window.orderBy(F.col(inputCol).asc_nulls_first())).alias('r'))
        df = df.select(F.col('*'),F.dense_rank().over(W.Window.orderBy(F.col(inputCol).asc_nulls_first())).alias(outputCol))
    else:
        #types = session.createDataFrame(list(enumerate(valueOrder)),['r',inputCol])
        df = df.withColumn(outputCol,F.when(F.col(inputCol).isin(valueOrder), F.array_position(variant=F.to_variant(inputCol), array=F.lit(valueOrder))).otherwise(0))
    return df,outputCol

def OneHotEncoder(df: snowpark.DataFrame ,inputCol: str ,outputCol: str):
    types = df.select(F.col(inputCol)).distinct().toPandas().values[:,0].tolist()
    outputColType = []
    for t in types:
        outputColType.append(outputCol+'_'+str(t))
        df = df.withColumn(outputColType[-1],F.when(F.col(inputCol) == F.lit(t),1).otherwise(0)) 
    return df,outputColType

def MinMaxScaler(df: snowpark.DataFrame ,inputCol: str ,outputCol: str):
    df = df.withColumn(outputCol,F.when((F.max(F.col(inputCol).cast('float')).over()-F.min(F.col(inputCol).cast('float')).over() > F.lit(0)),(F.col(inputCol)-F.min(F.col(inputCol)).over()).cast('float')/(F.max(F.col(inputCol).cast('float')).over()-F.min(F.col(inputCol).cast('float')).over())).otherwise(F.lit(0)))
    return df,outputCol


if __name__ == "__main__":
    rows = [
        ('value1'),
        ('value3'),
        ('value2')
    ]

    # Create a list of column names
    columns = ['input_col']

    # Create a PySpark DataFrame
    df = snowpark.Session.createDataFrame(data=rows, schema=columns)
    # Call BinEncoder() with appropriate arguments
    df, output_col = BinEncoder(df, 'input_col', 'output_col', ['value1', 'value2', 'value3'])

    # Print resulting DataFrame
    df.show()