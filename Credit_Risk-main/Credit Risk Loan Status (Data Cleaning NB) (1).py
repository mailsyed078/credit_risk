# Databricks notebook source
#importing the datasets:
df = spark.table('credit_risk_data')
display(df)

# COMMAND ----------

#summarizing the dataset:
dbutils.data.summarize(df)

# COMMAND ----------

#checking the datatypes of columns:
df.printSchema()

# COMMAND ----------

#checking the number of rows:
df.count()

# COMMAND ----------

df.na.drop().display()

# COMMAND ----------

# DBTITLE 1,Cleaning the data:
#removing not required columns:
# list of columns to be removed: ('member_id','funded_amnt_inv','batch_enrolled','pymnt_plan','desc','title','zip_code','mths_since_last_delinq','mths_since_last_record','mths_since_last_major_derog','verification_status_joint')

new_df = df.drop('member_id','funded_amnt_inv','batch_enrolled','pymnt_plan','desc','title','zip_code','mths_since_last_delinq','mths_since_last_record','mths_since_last_major_derog','emp_title','verification_status_joint')
display(new_df)

# COMMAND ----------

len(new_df.columns)

# COMMAND ----------

#checking for the duplicates:
new_df.dropDuplicates().count()

# COMMAND ----------

#checking for missing values:
new_df.na.drop().count()

# COMMAND ----------

#dropping the missing values:
new_df = new_df.na.drop()
display(new_df)

# COMMAND ----------

#checking the values in emp_length column:
display(new_df.groupBy('emp_length').count().orderBy('count',ascending=False))

# COMMAND ----------

#checking the values in purpose column:
display(new_df.groupBy('purpose').count().orderBy('count',ascending=False))

# COMMAND ----------

#checking the values in addr_state column:
display(new_df.groupBy('addr_state').count().orderBy('count',ascending=False))

# COMMAND ----------

#cleaning dti:
print(new_df.select('dti'))
display(new_df.select('dti').distinct())

# COMMAND ----------

#cleaning delinq_2yrs:
print(new_df.select('delinq_2yrs'))
display(new_df.select('delinq_2yrs').distinct())

# COMMAND ----------

#cleaning inq_last_6mths:
print(new_df.select('inq_last_6mths'))
display(new_df.select('inq_last_6mths').distinct())

# COMMAND ----------

#cleaning open_acc:
print(new_df.select('open_acc'))
display(new_df.select('open_acc').distinct())

# COMMAND ----------

#cleaning pub_rec:
print(new_df.select('pub_rec'))
display(new_df.select('pub_rec').distinct())

# COMMAND ----------

#cleaning revol_bal:
print(new_df.select('revol_bal'))
display(new_df.select('revol_bal').distinct())

# COMMAND ----------

#cleaning revol_util:
print(new_df.select('revol_util'))
display(new_df.select('revol_util').distinct())

# COMMAND ----------

#cleaning total_acc:
print(new_df.select('total_acc'))
display(new_df.select('total_acc').distinct())

# COMMAND ----------

#cleaning initial_list_status:
print(new_df.select('initial_list_status'))
display(new_df.select('initial_list_status').distinct())

# COMMAND ----------

#cleaning total_rec_int:
print(new_df.select('total_rec_int'))
display(new_df.select('total_rec_int').distinct())

# COMMAND ----------

#cleaning total_rec_late_fee:
print(new_df.select('total_rec_late_fee'))
display(new_df.select('total_rec_late_fee').distinct())

# COMMAND ----------

#cleaning recoveries:
print(new_df.select('recoveries'))
display(new_df.select('recoveries').distinct())

# COMMAND ----------

#cleaning collection_recovery_fee:
print(new_df.select('collection_recovery_fee'))
display(new_df.select('collection_recovery_fee').distinct())

# COMMAND ----------

#cleaning collections_12_mths_ex_med:
print(new_df.select('collections_12_mths_ex_med'))
display(new_df.select('collections_12_mths_ex_med').distinct())

# COMMAND ----------

#cleaning last_week_pay:
print(new_df.select('last_week_pay'))
display(new_df.select('last_week_pay').distinct())

# COMMAND ----------

#cleaning acc_now_delinq:
print(new_df.select('acc_now_delinq'))
display(new_df.select('acc_now_delinq').distinct())

# COMMAND ----------

#cleaning tot_coll_amt:
print(new_df.select('tot_coll_amt'))
display(new_df.select('tot_coll_amt').distinct())

# COMMAND ----------

#cleaning tot_cur_bal:
print(new_df.select('tot_cur_bal'))
display(new_df.select('tot_cur_bal').distinct())

# COMMAND ----------

#cleaning total_rev_hi_lim:
print(new_df.select('total_rev_hi_lim'))
display(new_df.select('total_rev_hi_lim').distinct())

# COMMAND ----------

#cleaning loan_status:
print(new_df.select('loan_status'))
display(new_df.groupBy('loan_status').count())

# COMMAND ----------

#saving the cleaned data
new_df.write.mode('overwrite').saveAsTable('credit_risk_cleaned')

# COMMAND ----------


