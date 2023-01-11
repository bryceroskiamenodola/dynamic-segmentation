from __future__ import annotations
from typing import Any
import snowflake.snowpark as snowpark
from snowflake.snowpark import Session
import snowflake.snowpark.functions as F
import snowflake.snowpark.window as W
import snowflake.snowpark.types as T

def silhouette(model,df:snowpark.DataFrame,id: str,inputCols: str)-> snowpark.DataFrame:
    '''
    this is the simplify version that O(NK) instead of O(N^2)
    https://arrow.tudublin.ie/cgi/viewcontent.cgi?article=1214&context=scschcomcon
    '''
    data = model.distance(df,id,inputCols).cache_result()
    a = data.where(F.col('r') == 1)#.groupBy(F.col(model.cluster_id)).agg(F.mean(F.col('distance')).alias('distance')).cache_result()
    b = data.where(F.col('r') == 2)#.join(data.where(F.col('r') != 1),on=[id],lsuffix='_L',rsuffix='_R').groupBy(F.col(model.cluster_id+'_L'),F.col(model.cluster_id+'_R')).agg(F.mean(F.col('distance'+'_R')).alias('distance')).groupBy(F.col(model.cluster_id+'_L')).agg(F.min(F.col('distance')).alias('distance')).cache_result()#.select(F.col(model.cluster_id+'_L').alias(model.cluster_id),F.col('distance')).cache_result()
    return a,b,float(b.join(a,[id],lsuffix='_L',rsuffix='_R').select(((F.col('distance'+'_L') - F.col('distance'+'_R')) / F.when(F.col('distance'+'_L') >= F.col('distance'+'_R'),F.col('distance'+'_L')).otherwise(F.col('distance'+'_R'))).alias('distance')).agg(F.mean(F.col('distance')).alias('distance')).toPandas().values[:,0])
    #a = data.where(F.col('r') == 1).groupBy(F.col(model.cluster_id)).agg(F.mean(F.col('distance')).alias('distance')).cache_result()
    #b = data.where(F.col('r') == 1).join(data.where(F.col('r') != 1),on=[id],lsuffix='_L',rsuffix='_R').groupBy(F.col(model.cluster_id+'_L'),F.col(model.cluster_id+'_R')).agg(F.mean(F.col('distance'+'_R')).alias('distance')).groupBy(F.col(model.cluster_id+'_L')).agg(F.min(F.col('distance')).alias('distance')).cache_result()#.select(F.col(model.cluster_id+'_L').alias(model.cluster_id),F.col('distance')).cache_result()
    #return a,b,float(b.join(a,F.col(model.cluster_id+'_L')==F.col(model.cluster_id),lsuffix='_L',rsuffix='_R').select(((F.col('distance'+'_L') - F.col('distance'+'_R')) / F.when(F.col('distance'+'_L') >= F.col('distance'+'_R'),F.col('distance'+'_L')).otherwise(F.col('distance'+'_R'))).alias('distance')).agg(F.mean(F.col('distance')).alias('distance')).toPandas().values[:,0])
#a,b,v = silhouette(model,df,'UNIVERSAL_ID',[c for c in df.columns if c != 'UNIVERSAL_ID'])
#print(v)