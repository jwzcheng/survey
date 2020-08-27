#!/usr/bin/env python
# coding: utf-8

# # Note
# Build the Spark environment (Run Standalone) with jupyter notebook:   
# 1. Install Anaconda.
# 2. Install JAVA 8.   
# 3. (MAC) brew install Apache-spark.  
# 4. pip install findspark.  
# 

# DATA: 82 features, 8921483 instances.  

# In[2]:


import findspark
findspark.init()
import os
import pyspark
import pandas as pd
from pyspark.sql import SparkSession
import pandas as pd
# spark = SparkSession.builder.getOrCreate()
os.chdir('/Users/jaycheng/Documents/ms_comp/')
# os.chdir('D:/ms_comp/')

# Load DATA
spark = SparkSession.builder.getOrCreate()
# data_train = spark.read.csv('./data/train_sample.csv', header=True)
# data_train = data_train.select(col)
# data = data_train
data_train = spark.read.csv('./data/train.csv', header=True)
# data_test = spark.read.csv('./data/test.csv', header=True)
# data_train = spark.read.csv('/Users/jaycheng/Documents/ms_comp/data/train.csv', header=True, inferSchema=True)
# data_train.show()

# data_train.count()
# data_train.dtypes
# len(data_train.columns)
# data_train.printSchema()
# data_test.printSchema()
# data_train.select('ProductName').show()
# %time data_train.select('ProductName').distinct().show()
# %time data_train.describe().toPandas()
# df = data_train.drop('MachineIdentifier').collect()


# In[239]:


# The features have too many categories.
col_many_category = [
 'CountryIdentifier',
 'Census_InternalPrimaryDisplayResolutionHorizontal',
 'Census_OSBuildRevision',
 'Census_TotalPhysicalRAM',
 'LocaleEnglishNameIdentifier',
 'GeoNameIdentifier',
 'OrganizationIdentifier',
 'AVProductStatesIdentifier',
 'Census_ProcessorCoreCount',
 'Census_PrimaryDiskTotalCapacity',
 'CityIdentifier',
 'Census_SystemVolumeTotalCapacity',
 'Census_OSBuildNumber',
 'Census_InternalPrimaryDisplayResolutionVertical']


col = pd.read_csv('col_all_nona.txt', header=None)
col = col.iloc[:,0].tolist()
col_a = pd.read_csv('col_a.txt', header=None)
col_a = col_a.iloc[:,0].tolist()
col_string = pd.read_csv('StringFeature.txt', header=None)
col_string = col_string.iloc[:,0].tolist()
col = set(col) - set(col_a) -set(col_many_category)
col = list(col)
data_train = data_train.select(col)

col_si = []
for i in col:
    for j in col_string:
        if i==j:
            col_si.append(i)
            
col_num = list(set(col) - set(col_si))


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql.functions import desc

for i in data_train.columns:
    if i == 'MachineIdentifier':
        continue
    else:    
        x = data_train.groupby(i).count().sort(desc('count')).toPandas()
        x.index = x.iloc[:,0]
        x['prob'] = x['count'] / sum(x['count'])
        threshold = 0.02
        mask = x['prob'] > threshold
        tail_prob = x['prob'].loc[~mask].sum()
        prob = x.prob.loc[mask]
        prob['other'] = tail_prob
        prob.plot(kind='barh')
        plt.title(i ,size=20)
#         plt.xticks(rotation=80)
        plt.xticks(size=8)
        plt.yticks(size=8)
#         plt.show()
        figname = "./plot/Bar_" + i + ".png"
        plt.savefig(figname, dpi=600)
        plt.close()  # must close, or ERROE occurs.


# In[237]:


# If a category > 0.8, ignore the feature.
# SAVE the features in "col_a"
# Add features with too many categories.
col_a = ['feature']
for i in data_train.columns:
    if i == 'MachineIdentifier':
        continue
    else:    
        x = data_train.groupby(i).count().toPandas()
        x['prob'] = x['count'] / sum(x['count'])
        if x['prob'].max() > 0.8 :
            print(i)
            col_a.append(i)            

col_many_category = [
 'CountryIdentifier',
 'Census_InternalPrimaryDisplayResolutionHorizontal',
 'Census_OSBuildRevision',
 'Census_TotalPhysicalRAM',
 'LocaleEnglishNameIdentifier',
 'GeoNameIdentifier',
 'OrganizationIdentifier',
 'AVProductStatesIdentifier',
 'Census_ProcessorCoreCount',
 'Census_PrimaryDiskTotalCapacity',
 'CityIdentifier',
 'Census_SystemVolumeTotalCapacity',
 'Census_OSBuildNumber',
 'Census_InternalPrimaryDisplayResolutionVertical']

for i in col_many_category:
    col_a.append(i)

col_a = pd.Series(col_a)
with open('col_a.txt', 'w') as f:
    f.write(col_a.to_string())


# In[ ]:


from pyspark.sql.functions import desc

for i in data_train.columns:
    if i == 'MachineIdentifier':
        continue
    else:    
        x = data_train.groupby(i).count().sort(desc('count')).toPandas()
    
    with open('value_count_spark_sample.txt', 'a', encoding='utf-8') as f:
        f.write('###=== ' + i + ' === \n')
        f.write(x.to_string() + '\n')
        f.write('\n')


# In[ ]:


from pyspark.sql.functions import desc

for i in data_train.columns:
    if i == 'MachineIdentifier':
        continue
    else:    
        x = data_train.filter(data_train['HasDetections']==0).groupby(i).count().sort(desc('count')).toPandas()

    with open('value_count_HasDetections_0.txt', 'a', encoding='utf-8') as f:
        f.write('###=== ' + i + ' === \n')
        f.write(x.to_string() + '\n')
        f.write('\n')

from pyspark.sql.functions import desc

for i in data_train.columns:
    if i == 'MachineIdentifier':
        continue
    else:    
        x = data_train.filter(data_train['HasDetections']==1).groupby(i).count().sort(desc('count')).toPandas()

    with open('value_count_HasDetections_1.txt', 'a', encoding='utf-8') as f:
        f.write('###=== ' + i + ' === \n')
        f.write(x.to_string() + '\n')
        f.write('\n')


# # Spark Pipeline

# In[3]:


from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier


# In[82]:


col_a = pd.read_csv('col_a.txt')
col_a = col_a['feature'].tolist()
col_select = set(data_train.columns) - set(col_a) -set(['MachineIdentifier','HasDetections'])
col_select 


# In[39]:


# Without Pipeline
import findspark
findspark.init()
import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.feature import StringIndexer, VectorIndexer, OneHotEncoder, VectorAssembler, IndexToString
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import *
import pandas as pd
os.chdir('/Users/jaycheng/Documents/ms_comp/')
# os.chdir('D:/ms_comp/')

# Load DATA
spark = SparkSession.builder.config('spark.debug.maxToStringFields', '100').getOrCreate()
data = spark.read.csv('./data/train_sample.csv', header=True)
# test = spark.read.csv('./data/test.csv', header=True)
test = spark.read.csv('./data/test_sample.csv', header=True)
# data = spark.read.csv('/Users/jaycheng/Documents/ms_comp/data/train.csv', header=True, inferSchema=True)
# data.show()

print('==PREPROCESSING== \n')
col = pd.read_csv('col_all_nona.txt', header=None)
col = col.iloc[:,0].tolist()
col.remove('SmartScreen')
col_a = pd.read_csv('col_a.txt', header=None)
col_a = col_a.iloc[:,0].tolist()
col_string = pd.read_csv('StringFeature.txt', header=None)
col_string = col_string.iloc[:,0].tolist()
col = set(col) - set(col_a)
col_test = set(col) - set(['HasDetections'])
col = list(col)
col_test = list(col_test)
col_test.append('MachineIdentifier')

col_si = []
for i in col:
    for j in col_string:
        if i==j:
            col_si.append(i)
            
col_num = list(set(col) - set(col_si))

data = data.select(col)
test = test.select(col_test)

# drop samples with missing value
data = data.dropna('any')

# # Spliting in train and test set. Beware : It sorts the dataset
# (data, data_test) = data.randomSplit([0.7,0.3])

# StringIndexer all features.
# stringindexer = [StringIndexer(inputCol=i, outputCol=i+"_index").fit(data) for i in col_si]
# pipeline = Pipeline(stages=stringindexer)
# data = pipeline.fit(data).transform(data)

for i in col:
    data = StringIndexer(inputCol=i, outputCol=i+"_index").fit(data).transform(data)

for i in col_test:
    if i == 'MachineIdentifier':
        continue
    else:  
        test = StringIndexer(inputCol=i, outputCol=i+"_index").fit(test).transform(test)

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
data = StringIndexer(inputCol="HasDetections", outputCol="indexedLabel").fit(data).transform(data)

# encoder_input_col = []
# for i in col:
#     encoder_input_col.append(i + '_index')
# for i in col_num:
#     encoder_input_col.append(i)

encoder_input_col = [
 'Census_MDC2FormFactor_index',
 'Census_ActivationChannel_index',
 'Census_ChassisTypeName_index',
 'OsSuite_index',
 'SkuEdition_index',
 'Census_PowerPlatformRoleName_index',
 'Census_OSBranch_index',
 'Census_OSSkuName_index',
 'Census_OSWUAutoUpdateOptionsName_index',
 'Census_InternalPrimaryDiagonalDisplaySizeInInches_index',
 'Census_OSInstallTypeName_index',
 'Census_OSEdition_index',
 'EngineVersion_index',
 'AppVersion_index',
 'Census_PrimaryDiskTypeName_index',
 'Census_IsSecureBootEnabled_index',
 'OsBuildLab_index',
 'AvSigVersion_index',
 'OsPlatformSubRelease_index',
 'Wdft_IsGamer_index',
 'Census_OSVersion_index',
 'OsBuild_index' 
]

# OneHotEncoder
for i in encoder_input_col:
    data = OneHotEncoder(dropLast = False, inputCol = i, outputCol = i+"_Vec").transform(data)
    test = OneHotEncoder(dropLast = False, inputCol = i, outputCol = i+"_Vec").transform(test)

assembler_input_col = []
for i in encoder_input_col:
    assembler_input_col.append(i + '_Vec')
    
# # Assembel all features into 'features'
data = VectorAssembler(inputCols=assembler_input_col, outputCol = 'features').transform(data)
test = VectorAssembler(inputCols=assembler_input_col, outputCol = 'features').transform(test)

# # Split the data into training and test sets (30% held out for testing)
(data_train, data_test) = data.randomSplit([0.7, 0.3])

print('==TRAINING MODEL== \n')

rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features", numTrees=10)

model = rf.fit(data_train)
 
predictions = model.transform(test)

test_predictions = model.transform(test)

# Select example rows to display.
predictions.select(["prediction", "probability"]).show(5)

# Select (prediction, true label) and compute test error
predictions = predictions.select(["prediction", "indexedLabel"])
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

# rfModel = model.stages[6]
# print(rfModel)  # summary only
 
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g" % accuracy)
 
evaluatorf1 = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="f1")
f1 = evaluatorf1.evaluate(predictions)
print("f1 = %g" % f1)
 
evaluatorwp = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="weightedPrecision")
wp = evaluatorwp.evaluate(predictions)
print("weightedPrecision = %g" % wp)
 
evaluatorwr = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="weightedRecall")
wr = evaluatorwr.evaluate(predictions)
print("weightedRecall = %g" % wr)

print('SAVE MODEL')
model.save("./rf_model")


# In[93]:


# Prediction
import findspark
findspark.init()
import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.feature import StringIndexer, VectorIndexer, OneHotEncoder, VectorAssembler, IndexToString
from pyspark.sql.functions import *
import pandas as pd
os.chdir('/Users/jaycheng/Documents/ms_comp/')

print('==PREDICTION== \n')
# Load DATA
spark = SparkSession.builder.config('spark.debug.maxToStringFields', '100').getOrCreate()
# test = spark.read.csv('./data/test.csv', header=True)
test = spark.read.csv('./data/test_sample.csv', header=True)

print('==PREPROCESSING== \n')
col = pd.read_csv('col_all_nona.txt', header=None)
col = col.iloc[:,0].tolist()
col.remove('SmartScreen')
col_a = pd.read_csv('col_a.txt', header=None)
col_a = col_a.iloc[:,0].tolist()
col_string = pd.read_csv('StringFeature.txt', header=None)
col_string = col_string.iloc[:,0].tolist()
col = set(col) - set(col_a)
col_test = set(col) - set(['HasDetections'])
col = list(col)
col_test = list(col_test)
col_test.append('MachineIdentifier')

col_si = []
for i in col:
    for j in col_string:
        if i==j:
            col_si.append(i)
            
col_num = list(set(col) - set(col_si))

test = test.select(col_test)

for i in col_test:
    if i == 'MachineIdentifier':
        continue
    else:  
        test = StringIndexer(inputCol=i, outputCol=i+"_index").fit(test).transform(test)

# encoder_input_col = []
# for i in col:
#     encoder_input_col.append(i + '_index')
# for i in col_num:
#     encoder_input_col.append(i)        
        
encoder_input_col = [
 'Census_MDC2FormFactor_index',
 'Census_ActivationChannel_index',
 'Census_ChassisTypeName_index',
 'OsSuite_index',
 'SkuEdition_index',
 'Census_PowerPlatformRoleName_index',
 'Census_OSBranch_index',
 'Census_OSSkuName_index',
 'Census_OSWUAutoUpdateOptionsName_index',
 'Census_InternalPrimaryDiagonalDisplaySizeInInches_index',
 'Census_OSInstallTypeName_index',
 'Census_OSEdition_index',
 'EngineVersion_index',
 'AppVersion_index',
 'Census_PrimaryDiskTypeName_index',
 'Census_IsSecureBootEnabled_index',
 'OsBuildLab_index',
 'AvSigVersion_index',
 'OsPlatformSubRelease_index',
 'Wdft_IsGamer_index',
 'Census_OSVersion_index',
 'OsBuild_index' 
]

# OneHotEncoder
for i in encoder_input_col:
    test = OneHotEncoder(dropLast = False, inputCol = i, outputCol = i+"_Vec").transform(test)

assembler_input_col = []
for i in encoder_input_col:
    assembler_input_col.append(i + '_Vec')
    
# Assembel all features into 'features'
test = VectorAssembler(inputCols=assembler_input_col, outputCol = 'features').transform(test)

#Load RF model
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
model = RandomForestClassificationModel.load("./rf_model")
prediction = model.transform(test)
print('DONE!!!')


# # Gradient Boosting

# In[29]:


# Without Pipeline
import findspark
findspark.init()
import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, OneHotEncoder, VectorAssembler, IndexToString
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import *
import pandas as pd
os.chdir('/Users/jaycheng/Documents/ms_comp/')
# os.chdir('D:/ms_comp/')

# Load DATA
spark = SparkSession.builder.config('spark.debug.maxToStringFields', '100').getOrCreate()
data = spark.read.csv('./data/train_sample2.csv', header=True)
# test = spark.read.csv('./data/test.csv', header=True)
test = spark.read.csv('./data/test_sample.csv', header=True)
# data = spark.read.csv('/Users/jaycheng/Documents/ms_comp/data/train.csv', header=True, inferSchema=True)

print('==PREPROCESSING== \n')
# Input the selected features
col = pd.read_csv('/Users/jaycheng/Dropbox/python/ms_comp/feature1.csv', header=None, index_col=0)
col = col.iloc[:,0].tolist()
col.append('MachineIdentifier')
col.append('HasDetections')
data = data.select(col)

# drop samples with missing value
data = data.dropna('any')

print('==StringIndexer== \n')
ignore = ['MachineIdentifier', 'HasDetections']
# StringIndexer all features.
stringindexer = [StringIndexer(inputCol=i, outputCol=i+"_index") for i in data.columns if i not in ignore]
pipeline = Pipeline(stages=stringindexer)
data = pipeline.fit(data).transform(data)

# Fit on whole dataset to include all labels in index.
labelindex = StringIndexer(inputCol="HasDetections", outputCol="indexedLabel")
data = labelindex.fit(data).transform(data)

# OneHotEncoder
print('==OneHotEncoder== \n')
encoder_input_col = []
for i in col: 
    if i not in ignore:
        encoder_input_col.append(i + '_index')

onehotencoder = [OneHotEncoder(dropLast = False, inputCol = i, outputCol = i+"_Vec") for i in encoder_input_col]
pipeline = Pipeline(stages=onehotencoder)
data = pipeline.fit(data).transform(data)
  
# Assembel all features into 'features'
print('==VectorAssembler== \n')
assembler_input_col = []
for i in encoder_input_col:
    assembler_input_col.append(i + '_Vec')
assembler = VectorAssembler(inputCols=assembler_input_col, outputCol = 'features')
# data = (assembler.transform(data).select("indexedLabel", "features"))
data = assembler.transform(data)


# # Split the data into training and test sets (30% held out for testing)
(data_train, data_test) = data.randomSplit([0.7, 0.3])
print('==TRAINING MODEL== \n')
#Gradient Boosting
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="features", maxIter=10)

model = gbt.fit(data_train)
 
predictions = model.transform(data_test)

# test_predictions = model.transform(test)

# Select example rows to display.
# predictions.select(["prediction", "probability"]).show(5)

# Select (prediction, true label) and compute test error
predictions = predictions.select(["prediction", "indexedLabel"])
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

# rfModel = model.stages[6]
# print(rfModel)  # summary only
 
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g" % accuracy)
 
evaluatorf1 = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="f1")
f1 = evaluatorf1.evaluate(predictions)
print("f1 = %g" % f1)
 
evaluatorwp = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="weightedPrecision")
wp = evaluatorwp.evaluate(predictions)
print("weightedPrecision = %g" % wp)
 
evaluatorwr = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="weightedRecall")
wr = evaluatorwr.evaluate(predictions)
print("weightedRecall = %g" % wr)


# In[67]:


# prepare submission
from pyspark.sql.types import *
predictions = model.transform(test)

print("predicted testing data")

# extract ids from test data
ids = test.select("MachineIdentifier").rdd.map(lambda x: str(x[0]))

# we should provide probability of 2nd class
targets = predictions.select("probability").rdd.map(lambda x: float(x[0][1]))

# create data frame consists of id and probabilities
submission = spark.createDataFrame(ids.zip(targets), StructType([StructField(
    "MachineIdentifier", IntegerType(), True), StructField("targets", FloatType(), True)]))

# store results after coalescing
# submission.coalesce(1).write.csv('%d-%g-%g.csv' %
#                                  (iteration, auc_roc, gini), header="true")
submission.coalesce(1).write.csv('./result_test.csv', header="true")


# In[65]:


submission.show(5)


# In[53]:


result = predictions.select(['MachineIdentifier', 'predition'])
result.coalesce(1).write.csv('result.csv', header="true")


# In[51]:


result


# In[41]:


# TESTING 
# GBoosting

# Without Pipeline
import findspark
findspark.init()
import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, OneHotEncoder, VectorAssembler, IndexToString
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import *
import pandas as pd
os.chdir('/Users/jaycheng/Documents/ms_comp/')
# os.chdir('D:/ms_comp/')

# Load test
spark = SparkSession.builder.config('spark.debug.maxToStringFields', '100').getOrCreate()
# test = spark.read.csv('./test/test.csv', header=True)
test = spark.read.csv('./data/test_sample.csv', header=True)

print('==PREPROCESSING== \n')
# Input the selected features
col = pd.read_csv('/Users/jaycheng/Dropbox/python/ms_comp/feature1.csv', header=None, index_col=0)
col = col.iloc[:,0].tolist()
col.append('MachineIdentifier')
test = test.select(col)

# drop samples with missing value
test = test.dropna('any')

print('==StringIndexer== \n')
ignore = ['MachineIdentifier']
# StringIndexer all features.
stringindexer = [StringIndexer(inputCol=i, outputCol=i+"_index") for i in test.columns if i not in ignore]
pipeline = Pipeline(stages=stringindexer)
test = pipeline.fit(test).transform(test)

# OneHotEncoder
print('==OneHotEncoder== \n')
encoder_input_col = []
for i in col: 
    if i not in ignore:
        encoder_input_col.append(i + '_index')

onehotencoder = [OneHotEncoder(dropLast = False, inputCol = i, outputCol = i+"_Vec") for i in encoder_input_col]
pipeline = Pipeline(stages=onehotencoder)
test = pipeline.fit(test).transform(test)
  
# Assembel all features into 'features'
print('==VectorAssembler== \n')
assembler_input_col = []
for i in encoder_input_col:
    assembler_input_col.append(i + '_Vec')
assembler = VectorAssembler(inputCols=assembler_input_col, outputCol = 'features')
# test = (assembler.transform(test).select("indexedLabel", "features"))
test = assembler.transform(test)

# test_predictions = model.transform(test)


# In[42]:


test_predictions = model.transform(test)


# In[39]:


test_predictions.columns


# In[1]:


# Random Forest
# Without Pipeline
import findspark
findspark.init()
import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, OneHotEncoder, VectorAssembler, IndexToString
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import *
import pandas as pd
os.chdir('/Users/jaycheng/Documents/ms_comp/')

# Load DATA
spark = SparkSession.builder.getOrCreate()
data = spark.read.csv('./data/train_sample100w.csv', header=True)

print('==PREPROCESSING== \n')
# Input the selected features
col = pd.read_csv('/Users/jaycheng/Dropbox/python/ms_comp/feature1.csv', index_col=0)
col = col.iloc[:,0].tolist()
col.append('MachineIdentifier')
col.append('HasDetections')
data = data.select(col)

# drop samples with missing value
# data = data.dropna('any')

# fill missing value -1
data = data.fillna('-1')

print('==StringIndexer== \n')
ignore = ['MachineIdentifier', 'HasDetections']
# StringIndexer all features.
stringindexer = [StringIndexer(inputCol=i, outputCol=i+"_index") for i in data.columns if i not in ignore]
pipeline = Pipeline(stages=stringindexer)
data = pipeline.fit(data).transform(data)

# Fit on whole dataset to include all labels in index.
labelindex = StringIndexer(inputCol="HasDetections", outputCol="indexedLabel")
data = labelindex.fit(data).transform(data)

# OneHotEncoder
print('==OneHotEncoder== \n')
encoder_input_col = []
for i in col: 
    if i not in ignore:
        encoder_input_col.append(i + '_index')

onehotencoder = [OneHotEncoder(dropLast = False, inputCol = i, outputCol = i+"_Vec") for i in encoder_input_col]
pipeline = Pipeline(stages=onehotencoder)
data = pipeline.fit(data).transform(data)

    
# Assembel all features into 'features'
print('==VectorAssembler== \n')
assembler_input_col = []
for i in encoder_input_col:
    assembler_input_col.append(i + '_Vec')
assembler = VectorAssembler(inputCols=assembler_input_col, outputCol = 'features')
# data = (assembler.transform(data).select("indexedLabel", "features"))
data = assembler.transform(data)

# Split the data into training and test sets (30% held out for testing)
(data_train, data_test) = data.randomSplit([0.7, 0.3])

print('==Training Random Forest== \n')
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features")
model = rf.fit(data_train)
predictions = model.transform(data_test)
 
# Select example rows to display.
predictions.select(["prediction", "probability"]).show(5)

# Select (prediction, true label) and compute test error
predictions = predictions.select(["prediction", "indexedLabel"])
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

# rfModel = model.stages[6]
# print(rfModel)  # summary only
 
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g" % accuracy)
 
evaluatorf1 = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="f1")
f1 = evaluatorf1.evaluate(predictions)
print("f1 = %g" % f1)
 
evaluatorwp = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="weightedPrecision")
wp = evaluatorwp.evaluate(predictions)
print("weightedPrecision = %g" % wp)
 
evaluatorwr = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="weightedRecall")
wr = evaluatorwr.evaluate(predictions)
print("weightedRecall = %g" % wr)

# rf_model.featureImportances


# ### RESULTS
# sample 100w
# 

# In[20]:


predictions.select(["prediction", "indexedLabel"]).show()


# In[ ]:


from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation

# r1 = Correlation.corr(data_train, data_train.columns).head()
pearsonCorr = Correlation.corr(data_train, 'ProductName', 'pearson').collect()[0][0]

print("Pearson correlation matrix:\n" + str(pearsonCorr[0]))


# In[22]:


from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import ChiSquareTest
# dataset = spark.createDataFrame(dataset, ["label", "features"])
chiSqResult = ChiSquareTest.test(data, 'features', 'indexedLabel')
chiSqResult.select("degreesOfFreedom").collect()[0]
chiSqResult.toPandas()


# In[ ]:


from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import ChiSquareTest
dataset = [[0, Vectors.dense([0, 0, 1])],
            [0, Vectors.dense([1, 0, 1])],
            [1, Vectors.dense([2, 1, 1])],
            [1, Vectors.dense([3, 1, 1])]]
dataset = spark.createDataFrame(dataset, ["label", "features"])
chiSqResult = ChiSquareTest.test(dataset, 'features', 'label')
chiSqResult.select("degreesOfFreedom").collect()[0]
# chiSqResult.toPandas()


# In[150]:


from pyspark.sql.types import DoubleType

output = output.withColumn("label", output["HasDetections"].cast(DoubleType()))


# In[151]:


from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import ChiSquareTest
chiSqResult = ChiSquareTest.test(output, 'features', 'label')
chiSqResult.select("degreesOfFreedom").collect()[0]


# In[ ]:


from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation

data = [(Vectors.sparse(4, [(0, 1.0), (3, -2.0)]),),
        (Vectors.dense([4.0, 5.0, 0.0, 3.0]),),
        (Vectors.dense([6.0, 7.0, 0.0, 8.0]),),
        (Vectors.sparse(4, [(0, 9.0), (3, 1.0)]),)]
df = spark.createDataFrame(data, ["features"])

r1 = Correlation.corr(df, "features").head()
print("Pearson correlation matrix:\n" + str(r1[0]))

r2 = Correlation.corr(df, "features", "spearman").head()
print("Spearman correlation matrix:\n" + str(r2[0]))


# In[ ]:


# 從多數資料先過濾

# 不對！！

df = data_train.filter(data_train['ProductName']=='win8defender')
df = df.filter((df['EngineVersion']=='1.1.15200.1') | (df['EngineVersion']=='1.1.15100.1'))
df = df.filter(df['AppVersion']=='4.18.1807.18075')
df = df.filter(df['IsBeta']=='0')
df = df.filter(df['RtpStateBitfield']=='7')
# df = df.filter(df['AppVersion']=='4.18.1807.18075')

print(df.count())


# In[ ]:


# https://github.com/Bergvca/pyspark_dist_explore/blob/master/README.md

from pyspark_dist_explore import hist
import matplotlib.pyplot as plt
plt.style.use('seaborn')
get_ipython().run_line_magic('matplotlib', 'inline')

tmp = data_train.select('CityIdentifier')

fig, ax = plt.subplots()
hist(ax, tmp, bins = 20, color=['red'])


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

s2 = pd.Series([1,2,3,4,5,2,3,333,2,123,434,1,2,3,1,11,11,432,3,2,4,3,3,3,54,34,24,2,223,2535334,3,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,30000, 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])
prob = s2.value_counts(normalize=True)
threshold = 0.02
mask = prob > threshold
tail_prob = prob.loc[~mask].sum()
prob = prob.loc[mask]
prob['other'] = tail_prob
prob.plot(kind='bar')
plt.xticks(rotation=25)
plt.show()

