import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.{StructType,StructField,StringType};
import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import sqlContext.implicits._
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.PCA


val data = sc.textFile("/Users/avikalchhetri/Desktop/BigData/Midterm/YearPredictionMSD.txt")

val parsedData = data.map(_.split(",")).map(p=>LabeledPoint(p(0).toDouble,Vectors.dense(p.slice(1,91).map(_.toDouble)))).cache()

val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0).cache()
val test = splits(1)

val pca = new PCA(training.first().features.size/2).fit(parsedData.map(_.features))
val training_pca = training.map(p => p.copy(features = pca.transform(p.features)))
val test_pca = test.map(p => p.copy(features = pca.transform(p.features)))

val lr = new LinearRegression()

println("LinearRegression parameters:\n" + lr.explainParams() + "\n")
lr.setMaxIter(10)
lr.setRegParam(0.01)
val pipeline = new Pipeline()
pipeline.setStages(Array(lr))
  
val crossval = new CrossValidator()
crossval.setEstimator(pipeline)
crossval.setEvaluator(new RegressionEvaluator)

val paramGrid = new ParamGridBuilder()
paramGrid.addGrid(lr.regParam, Array(0.1, 0.01))
paramGrid.build()
crossval.setEstimatorParamMaps(paramGrid.build())
crossval.setNumFolds(3) // Use 3+ in practice

val cvModel = crossval.fit(training_pca.toDF)
cvModel.transform(test_pca.toDF)
val preds=cvModel.transform(test_pca.toDF)

val ev= new RegressionEvaluator()
ev.setLabelCol("label")
ev.evaluate(preds)

//Linear Regression PCA Test Output Double = 10.391483371665563 
// Linear Regression PCA Train Double = 10.465358262933185