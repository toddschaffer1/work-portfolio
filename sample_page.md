## Spark SQL Project: Analysis of Various Colleges

**Project description:** This project involved using Spark SQL to extract and analyze a college dataset with more than 5000 different universities. The purpose was to showcase how Spark SQL could be used to run different machine learning techniques to find valuable insight that could help colleges better understand how to make their marketing strategies more effective.

### 1. Business Problem

In the United States, there are over 5300 different colleges and universities that prospective students can choose from. Colleges are now facing an even more challenging competitive climate with universities having the capabilities of offering the same educational content on an online platform device without requiring students to be on campus. This has made colleges adapt to the changes in the educational system and look for ways to differentiate themselves from not only regional competitors but global competitors as well. 

A linear regression model will be run on the student-faculty ratio and graduation rate to understand if this metric is important in reducing the churn rate that most colleges face. A regression decision tree model will also be run on the ten different metrics which include, student-faculty ratio, New students from top 10% of high school class, New students from top 25% of high school class, Number of full-time undergraduates, Number of part-time undergraduates, Instructional expenditure per student, Estimated student personal spending, out of state tuition price, room/board cost and cost of books to understand which features help provide the greatest influence to graduation rate. The higher the graduation rate for schools, the higher probability of keeping students for the entire life cycle. 

A CSV file containing over 700 different colleges with these metrics will be used to help address this business problem.

### 2. Loading the SQL packages and Creating the Dataframe from the CSV

The data was read in and loaded using the sqlContext.read.format function in python. Both models then used the VectorAssembler function to subset the columns needed to create the feature x/’s and y variable. The data was then randomly split with 80% of the data in a training variable and 20% in a test variable.

```python
from __future__ import print_function
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import functions
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.regression import DecisionTreeRegressor

sc= SparkContext()
sqlContext = SQLContext(sc)

# Create the SQL dataframe by loading in the College csv.file
# Ensured that headers were set to true so we did not have to manually type in the column names
college_df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('UpdatedCollege.csv')
college_df.cache()

# Subset and extracted the feature and y column and transformed it into the needed shape for the linear regression model
vectorAssembler = VectorAssembler(inputCols = ['SFRatio'], outputCol = 'S-F Ratio')
vectorizedcollege_df = vectorAssembler.transform(college_df)
vectorizedcollege_df = vectorizedcollege_df.select(['S-F Ratio', 'GradRate'])

# used the randomsplit function to subset 80% of the data in one variable and the other 20% in another variable for training/testing
randomsplits = vectorizedcollege_df.randomSplit([0.8, 0.2])
trainingDF = randomsplits[0]
testDF = randomsplits[1]

```

### 3. Analyzing the College Data using a Linear Regression and Regression Decision Tree Model
The decision tree was created using the DecisionTreeRegressor function and the linear regression model was run for 15 iteration cycles using the LinearRegression function within Pyspark. 

#### Linear Regression Model
```python
print('\n')
print("Linear Regression Model")
print('\n')

# called the linear regression function from pyspark ML library and performed 15 iterations
lir = LinearRegression(featuresCol = 'S-F Ratio', labelCol='GradRate', maxIter=15)
lirmodel = lir.fit(trainingDF)

# created a training summary variable to extract the training RMSE and R2 to see if there was statistical signficance
train_summary = lirmodel.summary
print("Linear Regression Training RMSE: %f" % train_summary .rootMeanSquaredError)
print("Linear Regression Training R2: %f" % train_summary .r2)
# outputted the summary to see a list of summary statisitcs
trainingDF.describe().show()

# created a set of predictions to see how close our predictions were to the test data frame and outputted the prediction along with the acutal values
predictions = lirmodel.transform(testDF)
predictions.select("prediction","GradRate","S-F Ratio").show(3)

# Used the regression evaluator function to assess the accuracy of the predictions to output the R2 and RSME
RegEval = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="GradRate",metricName="r2")
print("Linear Regression R Squared (R2) on test data = %g" % RegEval.evaluate(predictions))

results = lirmodel.evaluate(testDF)
print("Linear Regression Root Mean Squared Error (RMSE) on test data = %g" % results.rootMeanSquaredError)

```
#### Regression Decision Tree Model

```python
print('\n')
print("Decision Tree Model")
print('\n')
# Since SF Ratio wasn't a high determinant in graduation rate, we incorporated 9 more coloumns to see the effect on graduation rate but now using a regression decision tree model
vectorAssembler = VectorAssembler(inputCols = ['SFRatio', 'Top10perc', 'Top25perc', 'FUndergrad', 
                                               'PUndergrad', 'Expend', 'Personal', 'Outstate', 
                                               'RoomBoard', 'Books'], outputCol = 'features')
vectorizedcollege_df = vectorAssembler.transform(college_df)
vectorizedcollege_df = vectorizedcollege_df.select(['features', 'GradRate'])


randomsplits = vectorizedcollege_df.randomSplit([0.8, 0.2])
trainingDF = randomsplits[0]
testDF = randomsplits[1]

# Peforming the same steps as above but calling the function needed for decision tree. Also since we have joined all 10 columns in the vector assembler step, we just need to call the features
decisiontree = DecisionTreeRegressor(featuresCol ='features', labelCol = 'GradRate')
decisiontree_model = decisiontree.fit(trainingDF)
decisiontree_model_predictions = decisiontree_model.transform(testDF)
decisiontree_model_evaluation = RegressionEvaluator(
    labelCol="GradRate", predictionCol="prediction", metricName="rmse")
rmse = decisiontree_model_evaluation.evaluate(decisiontree_model_predictions)
print("Decision Tree Root Mean Squared Error (RMSE) on test data = %g" % rmse)

# Outputted the percentage of importance that each feature had on explaining the decision tree regression model. Top25 and SFRatio were the highest
print('Decision Tree Most Important features: ')
print( decisiontree_model.featureImportances)
# !spark-submit BigData_GroupAssignment.py > CollegeOutput.txt

```
### 4. Model Results

  The linear regression model had training RMSE of 15.99 and test RMSE of 15.43 indicating that there was high variance within data. The training R2 was .07 and test R2 was .15 indicating that there does not seem to be a high correlation between graduation rate and the student-faculty ratio. The average graduation rate was 65% with a 17% standard deviation.
  The regression decision tree model produced a similar test RMSE of 15.19. After using the function that extracts the most important features for a decision tree model, it was apparent that out-of-state tuition was the most influential feature in determining the graduation rate with 48%. The second most important feature was new students from the top 25% of their high school class with 19.8%. The other features had little effect on graduation with most being under 10% or 5%.


### 5. Conclusion
  Both models found that the student to faculty ratio is not a hard determinant in the graduation rate for universities. The most influential metrics that affect graduation rate is bringing students who are in the top 25% in their high school class and the cost of out-of-state tuition. In the future, colleges and universities should strategically focus on targeting seniors who fall within the top 25% as these students will have a longer life-cycle at the school. They could also look at implementing different procedures for charging out-of-state tuition prices by either offering more scholarships or possibly allowing out-of-state students to pay in-state tuition prices after being at the school for one year to increase the probability of these students staying.  
