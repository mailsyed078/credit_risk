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

#applying rformula on train data:
rform_model = rform.fit(train)
enc_df_train = rform_model.transform(train)
enc_df_train.display()


# COMMAND ----------

#applying rform on test data
enc_df_test = rform_model.transform(test)
enc_df_test.display()

# COMMAND ----------

#applying standard scaler:
sc_model = sc.fit(enc_df_train)
sc_train = sc_model.transform(enc_df_train)
sc_train.display()

# COMMAND ----------

#applying standard scaler on test data:
sc_test = sc_model.transform(enc_df_test)
sc_test.display()

# COMMAND ----------

# DBTITLE 1,Logistic Regression
#creating the pipeline
from pyspark.ml.classification import LogisticRegression
algo = LogisticRegression(featuresCol='sc_features',
                            labelCol='loan_status',
                            )

stages = [rform,sc,algo]
pipeline_lr = Pipeline(stages=stages)


# COMMAND ----------

#training the model with default hyperparameters:

pipe_model = pipeline_lr.fit(train)
preds = pipe_model.transform(val)
display(preds)

# COMMAND ----------

#evaluating the base model
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(metricName='f1',labelCol='loan_status')
score = evaluator.evaluate(preds)
print(score)

# COMMAND ----------

# DBTITLE 1,Hyper Parameter Tuning for LR Model
#defining the objective function:
import mlflow

def objective_fn(params):
        #setting up the hyperparameters:
        max_iter = params["max_iter"]
        threshold = params["threshold"]
        
        with mlflow.start_run(run_name='LR_Model_hyp') as run:
            estimator = pipeline_lr.copy({algo.maxIter:max_iter, 
                                        algo.threshold:threshold})
            model = estimator.fit(train)
            preds = model.transform(val)
            score = evaluator.evaluate(preds)
            mlflow.log_metric('f1_score',score)

        return -score

# COMMAND ----------

#defining the search space:
from hyperopt import hp
search_space = {
    'max_iter' : hp.quniform("max_iter",100,500,10),
    'threshold' : hp.quniform("threshold",0.1,1,0.1),
}


# COMMAND ----------

#creating the best_params
from hyperopt import hp,tpe,Trials,fmin
best_params = fmin(fn=objective_fn,
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=8,
                    trials=Trials())
best_params

# COMMAND ----------

#retraining the model with the best params:
with mlflow.start_run(run_name='LR_Final_CreditRisk') as run:
    mlflow.autolog()
    best_max_iter = best_params["max_iter"]
    best_threshold = best_params["threshold"]
    
    #creating the pipeline
    estimator = pipeline_lr.copy({algo.maxIter:best_max_iter, 
                                        algo.threshold:best_threshold})
    model = estimator.fit(train)
    preds = model.transform(val)
    score = evaluator.evaluate(preds)

    #logging the parameters and metrics:
    mlflow.log_metric('f1_score',score)
    mlflow.log_param('best_max_iter',best_max_iter)
    mlflow.log_param('best_threshold',best_threshold)
    

# COMMAND ----------

# DBTITLE 1,Predictions on test data
import mlflow
run_id = run.info.run_id
logged_model = f'runs:/{run_id}/model'

# Load model
loaded_model = mlflow.spark.load_model(logged_model)

# Perform inference via model.transform()
predictions = loaded_model.transform(test)
display(predictions)

# COMMAND ----------

#accuracy on test data:
score = evaluator.evaluate(predictions)
print(score)

# COMMAND ----------


