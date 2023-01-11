from __future__ import annotations
from typing import Any
import snowflake.snowpark as snowpark
from snowflake.snowpark import Session
import snowflake.snowpark.functions as F
import snowflake.snowpark.window as W
import snowflake.snowpark.types as T
import math
import uuid
#from config import get_session
def join(s1: str, s2: str):
    if s1[-1] == '"':
        return s1[:-1]+s2+'"'
    else:
        return s1+s2
def decay(iteration:int,time_constant:int) -> float:
    return math.exp(-iteration/time_constant)
def step(iteration:int,time_constant:int) -> float:
    return (1.0 - (float(iteration)/time_constant))
def constant(iteration:int,time_constant:int) -> float:
    return float(1.0)

class Kmeans(object):
    def __init__(self, k: int,run_id : str=None,*args):
        super(Kmeans, self).__init__(*args)
        self.k = int(k)
        self.cluster_id = 'CLUSTER_ID'
        self.cluster = None #get_session().createDataFrame(data=[ v for v in range(self.k)],schema=T.StructType([T.StructField(self.cluster_id, T.IntegerType())])).cache_result()
        self.prediction = None
        self.run_id = run_id if run_id else  str(uuid.uuid4())
    def initializeCluster(self,df: snowpark.DataFrame,inputCols: str) -> snowpark.DataFrame:
        return df.crossJoin(self.cluster).groupBy(F.col(self.cluster_id)) \
            .agg([F.min(F.col(c)).alias(c+'_min') for c in inputCols]+[F.max(F.col(c)).alias(c+'_max') for c in inputCols]) \
            .select([self.cluster_id]+[(F.uniform(F.lit(0.0),F.lit(1.0), F.random())*(F.col(c+'_max')-F.col(c+'_min'))+F.col(c+'_min')).alias(c) for c in inputCols])
    def initializeCluster2(self,df: snowpark.DataFrame, inputCols: str):
        return df.sample(n=self.k).select([F.col(c) for c in inputCols]+[F.row_number().over(W.Window.orderBy(F.lit(None))).alias(self.cluster_id)])
    def distance(self,df: snowpark.DataFrame,id: str,inputCols: str) -> snowpark.DataFrame:
        return df.crossJoin(self.cluster,lsuffix='_L',rsuffix='_R') \
            .select(F.col(id) if id != self.cluster_id else F.col(join(id,'_L')),F.col(self.cluster_id) if id != self.cluster_id else F.col(join(self.cluster_id,'_R')),F.sqrt(sum([F.pow(F.col(join(c,'_L'))-F.col(join(c,'_R')),F.lit(2)) for c in inputCols])).alias('distance'),F.row_number().over(W.Window.partitionBy(F.col(id) if id != self.cluster_id else F.col(join(id,'_L'))).orderBy(F.col('distance').asc())).alias('r'))\
            .select(F.col(id) if id != self.cluster_id else F.col(join(id,'_L')),F.col(self.cluster_id) if id != self.cluster_id else F.col(join(self.cluster_id,'_R')),F.col('distance'),F.col('r'))
    def distance2(self,df: snowpark.DataFrame,id: str,inputCols: str) -> snowpark.DataFrame:
        return df.crossJoin(self.cluster,lsuffix='_L',rsuffix='_R') \
            .select(F.col(id) if id != self.cluster_id else F.col(join(id,'_L')),F.col(self.cluster_id) if id != self.cluster_id else F.col(join(self.cluster_id,'_R')),sum([F.pow(F.col(join(c,'_L'))-F.col(join(c,'_R')),F.lit(2)) for c in inputCols]).alias('distance'),F.row_number().over(W.Window.partitionBy(F.col(id) if id != self.cluster_id else F.col(join(id,'_L'))).orderBy(F.col('distance').asc())).alias('r'))\
            .select(F.col(id) if id != self.cluster_id else F.col(join(id,'_L')),F.col(self.cluster_id) if id != self.cluster_id else F.col(join(self.cluster_id,'_R')),F.col('distance'),F.col('r'))
    def fit(self,df: snowpark.DataFrame,id: str,inputCols: str,maxIter=None,initialize=True) -> Kmeans:
        if initialize:
            self.cluster = self.initializeCluster2(df,inputCols).cache_result()
        prediction = self.transform(df,id,inputCols).cache_result()
        iteration = 0
        while (self.prediction == None or (self.prediction.except_(prediction).count() > 0)) and (maxIter==None or iteration < maxIter):
            print('run_id:',self.run_id,',iteration:',iteration) 
            self.prediction = prediction
            print(self.prediction.groupBy(F.col(self.cluster_id)).agg(F.count('*')).orderBy(F.col(self.cluster_id)).toPandas())
            self.cluster = df.join(self.prediction,[id]).groupBy(F.col(self.cluster_id)).agg([ F.mean(c).alias(c) for c in inputCols]) \
                .union(self.cluster.where(F.col(self.cluster_id).isin(self.prediction.select(F.col(self.cluster_id))) == F.lit(False))).cache_result()
            prediction = self.transform(df,id,inputCols).cache_result()
            iteration += 1
            self.cluster.select(F.col('*'),F.lit(iteration).alias('t'),F.lit(self.run_id).alias('run_id')).write.mode('append').save_as_table('kmeans_t2')
        return self
    
    def transform(self,df: snowpark.DataFrame,id: str,inputCols: str) ->snowpark.DataFrame:
        return self.distance2(df,id,inputCols).where(F.col('r') == F.lit(1)).select(F.col(id),F.col(self.cluster_id))
    
    
class SOM(object):
    # https://www.osti.gov/servlets/purl/1566795
    def __init__(self,k : int=None,h : int=None,w : int=None,sigma : float=1.0 ,lr : float=1.0,topology='rectangular',activation_distance='euclidean',lr_func=step,sigma_func=step, run_id : str=None,*args):
        super(SOM, self).__init__(*args)
        if w is None and h is None:
            self.h = int(k)
            self.w = int(k)
            self.k = int(k**2)
        else:
            self.h = int(h)
            self.w = int(w)
            self.k = int(self.w)*int(self.h)
        self.lr = lr
        self.sigma = sigma
        self.lr_func= lr_func
        self.sigma_func= sigma_func
        self.topology = topology
        self.distance_function = {'euclidean': self._euclidean_distance,
                              'cosine': self._cosine_distance,
                              'manhattan': self._manhattan_distance,
                              'chebyshev': self._chebyshev_distance}[activation_distance]
        self.cluster_id = 'CLUSTER_ID'
        self.cluster = None #get_session().createDataFrame(data=[ v for v in range(self.w*self.h)],schema=T.StructType([T.StructField(self.cluster_id, T.IntegerType())])).cache_result()
        self.neighbor = None #self.convertToCoord(self.cluster).crossJoin(self.convertToCoord(self.cluster),lsuffix='_L',rsuffix='_R').select(F.col(join(self.cluster_id,'_L')).alias('BMI_ID'),F.col(join(self.cluster_id,'_R')).alias(self.cluster_id),(F.pow(F.col(join('X','_L'))-F.col(join('X','_R')),F.lit(2))+F.pow(F.col(join('Y','_L'))-F.col(join('Y','_R')),F.lit(2))).alias('DISTANCE')).cache_result()
        self.run_id = run_id if run_id else  str(uuid.uuid4())
    def initializeCluster(self,df: snowpark.DataFrame, inputCols: str) -> snowpark.DataFrame:
        return df.crossJoin(self.cluster).groupBy(F.col(self.cluster_id)) \
            .agg([F.min(F.col(c)).alias(c+'_min') for c in inputCols]+[F.max(F.col(c)).alias(c+'_max') for c in inputCols]) \
            .select([self.cluster_id]+[(F.uniform(F.lit(0.0),F.lit(1.0), F.random())*(F.col(c+'_max')-F.col(c+'_min'))+F.col(c+'_min')).alias(c) for c in inputCols])
    def initializeCluster2(self,df: snowpark.DataFrame, inputCols: str):
        return df.sample(n=self.w*self.h).select([F.col(c) for c in inputCols]+[(F.row_number().over(W.Window.orderBy(F.lit(None)))-F.lit(1)).alias(self.cluster_id)]).cache_result()
    
    def _cosine_distance(self, inputCols):
        num = sum([F.col(join(c,'_L'))*F.col(join(c,'_R')) for c in inputCols])
        denum = F.sqrt(sum([F.pow(F.col(join(c,'_L')),F.lit(2)) for c in inputCols]))*F.sqrt(sum([F.pow(F.col(join(c,'_R')),F.lit(2)) for c in inputCols]))
        return F.lit(1 - num / (denum+1e-8))

    def _euclidean_distance(self, inputCols):
        return F.sqrt(sum([F.pow(F.col(join(c,'_L'))-F.col(join(c,'_R')),F.lit(2)) for c in inputCols]))

    def _manhattan_distance(self, inputCols):
        return sum([F.abs(F.col(join(c,'_L'))-F.col(join(c,'_R'))) for c in inputCols])

    def _chebyshev_distance(self, inputCols):
        return F.greatest(*[F.abs(F.col(join(c,'_L'))-F.col(join(c,'_R'))) for c in inputCols])

    def distance(self,df: snowpark.DataFrame,id: str,inputCols: str) -> snowpark.DataFrame:
        print(self.distance_function.__name__)
        return df.crossJoin(self.cluster,lsuffix='_L',rsuffix='_R') \
            .select(F.col(id) if id != self.cluster_id else F.col(join(id,'_L')),F.col(self.cluster_id) if id != self.cluster_id else F.col(join(self.cluster_id,'_R')),self.distance_function(inputCols).alias('distance'),F.row_number().over(W.Window.partitionBy(F.col(id) if id != self.cluster_id else F.col(join(id,'_L'))).orderBy(F.col('distance').asc())).alias('r'))\
            .select(F.col(id) if id != self.cluster_id else F.col(join(id,'_L')),F.col(self.cluster_id) if id != self.cluster_id else F.col(join(self.cluster_id,'_R')),F.col('distance'),F.col('r'))
    def getNeighborHood(self,inputCols: str, sigma: float) -> snowpark.DataFrame:
        return self.cluster.join(self.neighbor,[self.cluster_id],lsuffix='_L',rsuffix='_R')\
            .select([F.col('BMI_ID')]+[F.col(self.cluster_id).alias(self.cluster_id)]+[F.col(c) for c in self.cluster.columns if c != self.cluster_id]+[F.when(F.lit(sigma)>F.lit(0.0),F.exp(-F.col('X_D')/(F.lit(2)*F.pow(sigma,2)))*F.exp(-F.col('Y_D')/(F.lit(2)*F.pow(sigma,2)))).otherwise(F.when((F.col('DISTANCE')==F.lit(0.0)) & (F.col('Y_D')==F.lit(0.0)),F.lit(1.0)).otherwise(F.lit(0.0))).alias('influence_rate')])
                    
    def getNeighborHood2(self, sigma: float) -> snowpark.DataFrame:
        return self.neighbor.select(F.col('BMI_ID'),F.col(self.cluster_id).alias(self.cluster_id),F.when(F.lit(sigma)>F.lit(0.0),F.exp(-(F.col('DISTANCE'))/(F.lit(2)*F.pow(sigma,2)))).otherwise(F.when((F.col('DISTANCE')==F.lit(0.0)),F.lit(1.0)).otherwise(F.lit(0.0))).alias('influence_rate'))
    def convertToCoord(self,df: snowpark.DataFrame):
        print(self.topology)
        if self.topology == 'hexagonal':
            return self.cluster.select(F.col(self.cluster_id),(F.col(self.cluster_id)%F.lit(self.w)).alias('x'),F.floor(F.col(self.cluster_id)/F.lit(self.w)).alias('y')).join(df,self.cluster_id).select(self.cluster_id,'x',(F.col('y')-F.when((F.max('x').over()-F.col('x'))%2==0,F.lit(0.5)).otherwise(F.lit(0.0))).alias('y'))
        else:
            return self.cluster.select(F.col(self.cluster_id),(F.col(self.cluster_id)%F.lit(self.w)).alias('x'),F.floor(F.col(self.cluster_id)/F.lit(self.w)).alias('y')).join(df,self.cluster_id)
    def fit(self,df: snowpark.DataFrame,id: str,inputCols: str,maxIter=5, batchSize=None,initialize=True) -> SOM:
        if initialize == True:
            print('initializing')
            self.cluster = self.initializeCluster2(df,inputCols).cache_result()
            self.neighbor = self.convertToCoord(self.cluster).crossJoin(self.convertToCoord(self.cluster),lsuffix='_L',rsuffix='_R').select(F.col(join(self.cluster_id,'_L')).alias('BMI_ID'),F.col(join(self.cluster_id,'_R')).alias(self.cluster_id),(F.pow(F.col(join('X','_L'))-F.col(join('X','_R')),F.lit(2))+F.pow(F.col(join('Y','_L'))-F.col(join('Y','_R')),F.lit(2))).alias('DISTANCE')).cache_result()
        else:
            print('skipped initializing')
        iteration = 0
        lr=self.lr
        sigma=self.sigma
        maxIter = maxIter if batchSize==None else maxIter*(df.count()/batchSize)
        time_constant = maxIter
        while iteration < maxIter:
            print('run_id:',self.run_id,',iteration:',iteration,',lr:',lr,',sigma:',sigma) 
            sample = df if batchSize==None else df.sample(n=batchSize).cache_result()
            bmu = self.distance(sample,id,inputCols).where(F.col('r') == F.lit(1)).drop(F.col('r')).rename(self.cluster_id,'BMI_ID').cache_result()
            self.bmu = bmu
            #print(bmu.groupBy(F.col('BMI_ID')).agg(F.count('*')).orderBy(F.col('BMI_ID').toPandas())
            neighborhood = self.getNeighborHood2(sigma).cache_result()
            self.cluster = sample.join(bmu,id).join(neighborhood.join(self.cluster,on=self.cluster_id),on='BMI_ID',lsuffix='_L',rsuffix='_R').select([F.col(self.cluster_id)]+[(F.lit(lr)*(F.col('influence_rate')*(F.col(join(c,'_L'))-F.col(join(c,'_R'))))).alias(c) for c in inputCols]) \
                .groupBy(F.col(self.cluster_id)).agg([F.mean(c).alias(c) for c in inputCols])\
                .join(self.cluster,on=self.cluster_id,lsuffix='_L',rsuffix='_R').select([F.col(join(self.cluster_id,'')).alias(self.cluster_id)]+[(F.col(join(c,'_L'))+F.col(join(c,'_R'))).alias(c) for c in inputCols]).cache_result()
                # .union(self.cluster.where(F.col(self.cluster_id).isin(neighborhood.select(F.col(self.cluster_id))) == F.lit(False))) is not needed as the 'influence_rate' is zero where the neighbors are out of range 
            iteration += 1
            lr=self.lr*self.lr_func(iteration,time_constant)
            sigma=self.sigma*self.sigma_func(iteration,time_constant)
            #self.cluster.select(F.col('*'),F.lit(iteration).alias('t'),F.lit(self.run_id).alias('run_id')).write.mode('append').save_as_table('som_t3')
        return self
    
    def transform(self,df: snowpark.DataFrame,id: str,inputCols: str):
        return self.distance(df,id,inputCols).where(F.col('r') == F.lit(1)).select(F.col(id),F.col(self.cluster_id))