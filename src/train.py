from __future__ import annotations
from typing import Any
import snowflake.snowpark as snowpark
import snowflake.snowpark.functions as F
import snowflake.snowpark.window as W
import snowflake.snowpark.types as T
import datetime
import json
import mlflow
import optuna
from optuna.samplers import TPESampler
from ml_models import Kmeans,SOM
from ml_steps import constant,step,decay
from ml_scores import silhouette
from config import get_session

def train(client:mlflow.tracking.MlflowClient,experiment_id:int,parent_run_id:int,sample:snowpark.DataFrame,id_:str,inputCols:list,func_param:dict):
    func = func_param
    #experiment_id,parent_run_id = func['experiment_id'],func['parent_run_id']
    m,p= func['algo'],func['param']
    p['model_name'] = func['model_name']
    lr_func,sigma_func = p['lr_func'],p['sigma_func']
    p['lr_func'] = p.get('lr_func').__name__
    p['sigma_func'] = p.get('sigma_func').__name__
    print('parameters:',json.dumps(p))
    score = None
    try:
        child_run = client.create_run(
            experiment_id=experiment_id,
            tags={
                mlflow.utils.mlflow_tags.MLFLOW_PARENT_RUN_ID: parent_run_id
            }
        )
        run_id=child_run.info.run_id
        start = datetime.datetime.now()
        client.set_tag(run_id,'start',start)
        print('run_id:',run_id)
        print('param:',p)
        for k,v in p.items():
            client.log_param(run_id,k,v)
        m = Kmeans(k=p['k'],run_id=run_id) if p['model_name'] == 'kmeans' else SOM(k=p.get('k'),w=p.get('w'),h=p.get('h'),sigma=p['sigma'],lr=p['lr'],topology=p['topology'],activation_distance=p['activation_distance'],lr_func=lr_func,sigma_func=sigma_func,run_id=run_id)
        m=m.fit(sample,id_,inputCols,maxIter=p.get('maxIter'))
        _,_,score = silhouette(m,sample,id_,inputCols)
        end = datetime.datetime.now()
        runtime =str(end-start)[:-7]
        ## Write csv from stats dataframe
        model_file_name = './models/{run_id}_cluster.csv'.format_map({'run_id':run_id})
        m.cluster.toPandas().to_csv(model_file_name)
        ## Log CSV to MLflow
        client.log_artifact(run_id,model_file_name)
        client.log_metric(run_id,'score',score)
        client.set_tag(run_id,'end',end)
        client.set_tag(run_id,'runtime',runtime)
        client.set_terminated(run_id)
        #mlflow.log_dict({'score':score,'start':start,'end':end,'runtime':runtime},)
        # to tables
        m.cluster.select(F.lit(experiment_id).alias('experiment_id'),F.lit(run_id).alias('run_id'),F.col('*')).write.mode("append").save_as_table('cluster')
        session = get_session()
        session.sql('select 6 as version').select(F.lit(experiment_id).alias('experiment_id'),F.lit(run_id).alias('run_id'),F.lit(score).alias('score'),F.lit(json.dumps(p)).alias('param'),F.lit(start).alias('START_TIMESTAMP'),F.lit(end).alias('END_TIMESTAMP'),F.lit(runtime).alias('RUNTIME_TIMESTAMP'),F.col('*')).write.mode("append").save_as_table('cluster_meta')
        print('parameters:',json.dumps(p),', score:',score)
    except Exception as e:
        print(e)
    return score

if __name__ == "__main__":
    session = get_session()
    df = session.read.table('EXTENDED_ATTRIBUTES_CONSOLIDATED_FULL').cache_result()
    sample = df#.sample(n=5).cache_result()
    id_ = 'UNIVERSAL_ID'
    inputCols = [c for c in sample.columns if c != id_]

    def get_objective(client,experiment,parent_run_id,sample,id_,inputCols):
        
        def objective(trails: optuna.Trial):
            search_space = {'model_name':'som','algo':SOM,'param':{'w':trails.suggest_int('w',2, 6),'h':trails.suggest_int('h',2, 6),'maxIter':trails.suggest_int('maxIter',15, 30), 'sigma':trails.suggest_float('sigma',0, 0.75), 'lr':trails.suggest_float('lr',1.0, 1.0),'topology':trails.suggest_categorical('topology',['hexagonal', 'rectangular']),'activation_distance':trails.suggest_categorical('activation_distance',['euclidean', 'cosine', 'manhattan', 'chebyshev']),'lr_func':trails.suggest_categorical('lr_func',[constant,step,decay]), 'sigma_func':trails.suggest_categorical('sigma_func',[constant,step,decay])}}
            
            return -1*train(client,experiment,parent_run_id,sample,id_,inputCols,search_space)
        return objective
    best_hyperparameters = None
    client = mlflow.tracking.MlflowClient()
    experiment_name = 'SOM'
    try:
        experiment = client.create_experiment(experiment_name)
    except:
        experiment = client.get_experiment_by_name(experiment_name).experiment_id

    study_run = client.create_run(experiment_id=experiment)
    parent_run_id = study_run.info.run_id
    study = optuna.create_study(direction="minimize",sampler=TPESampler())
    best_hyperparameters = study.optimize(get_objective(client,experiment,parent_run_id,sample,id_,inputCols), n_trials=30, n_jobs=4)
    best_hyperparameters