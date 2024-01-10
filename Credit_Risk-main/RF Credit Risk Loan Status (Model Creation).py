# Databricks notebook source
#importing cleaned dataset:
df = spark.table('credit_risk_cleaned')
display(df)

# COMMAND ----------

#checking the number of rows:
df.count()

# COMMAND ----------

# DBTITLE 1,Creating the ML Pipeline
#importing the libraries
from pyspark.ml.feature import StandardScaler,RFormula
from pyspark.ml import Pipeline

#separatng numeric and categorical columns columns:
num_cols = [i for (i,j) in  df.dtypes if i != 'loan_status' and j != 'string']
print(num_cols)
cat_cols = [i for (i,j) in  df.dtypes if i != 'loan_status' and j == 'string']
print(cat_cols)


# COMMAND ----------

#data spliting:
train,val,test = df.randomSplit([0.6,0.2,0.2])

# COMMAND ----------

#creating standard scaler and rformula object
rform = RFormula(formula="loan_status ~ .",
                 handleInvalid='skip',
                 featuresCol='enc_features',
                 labelCol='loan_status')

sc = StandardScaler(inputCol='enc_features',
                    outputCol='sc_features')
                     

# COMMAND ----------

# DBTITLE 1,Random Forest
#creating randomforest pipeline
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(featuresCol='sc_features',
                            labelCol='loan_status',
                            )
stages = [rform,sc,rf]
pipeline = Pipeline(stages=stages)
pipe_model = pipeline.fit(train)
preds = pipe_model.transform(test)

# COMMAND ----------

#evaluating the base model
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(metricName='f1',labelCol='loan_status')
score = evaluator.evaluate(preds)
print(score)

# COMMAND ----------

#creating randomforest pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler

num_cols = [i for (i,j) in train.dtypes if i != 'loan_status' and j!='string']
cat_cols = [i for (i,j) in train.dtypes if i != 'loan_status' and j=='string']
si_cols = [x + '_si' for x in cat_cols]
si = StringIndexer(inputCols=cat_cols,outputCols=si_cols,handleInvalid='skip')

inp_cols = num_cols + si_cols
vc = VectorAssembler(inputCols=inp_cols,outputCol='features')
rf = RandomForestClassifier(featuresCol='features',
                            labelCol='loan_status',
                            maxBins=60,numTrees=50)

stages = [si,vc,rf]
pipeline = Pipeline(stages=stages)
pipe_model = pipeline.fit(train)
preds = pipe_model.transform(test)
score = evaluator.evaluate(preds)
print(score)

# COMMAND ----------

#checking the feature importances:
import pandas as pd
features = pipe_model.stages[-1].featureImportances
features_df = pd.DataFrame(list(zip(vc.getInputCols(),features)),columns=["feature","importance"])
features_df

# COMMAND ----------

# DBTITLE 1,Hyper Parameter Tuning for LR Model
#defining the objective function:
import mlflow

def objective_fn(params):
        #setting up the hyperparameters:
        max_iter = params["max_iter"]
        threshold = params["threshold"]
        
        with mlflow.start_run(run_name='LR_Model_hyp') as run:
            estimator = pipeline_lr.copy({algo.maxIter:max_iter, algo.threshold:threshold})
            model = estimator.fit(train)
            preds = model.transform(test)
            score = evaluator.evaluate(preds)
            mlflow.log_metric('f1_score',score)

        return -score

# COMMAND ----------

#defining the search space:
from hyperopt import hp
search_space = {
    'max_iter' : hp.quniform("max_iter",100,500,1),
    'threshold' : hp.uniform("threshold",0.1,1)
}


# COMMAND ----------

#creating the best_params
from hyperopt import hp,tpe,Trials,fmin
best_params = fmin(fn=objective_fn,
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=16,
                    trials=Trials())
best_params

# COMMAND ----------


