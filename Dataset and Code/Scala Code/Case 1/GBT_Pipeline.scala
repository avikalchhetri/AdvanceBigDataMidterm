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
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.mllib.feature.PCA

val data = sc.textFile("/Users/avikalchhetri/Desktop/BigData/Midterm/YearPredictionMSD.txt")

val parsedData = data.map(_.split(",")).map(p=>LabeledPoint((if(p(0).toInt<1965){0.0}else{1.0}),Vectors.dense(p.slice(1,91).map(_.toDouble)))).cache()

val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)

val training = splits(0).cache()
val test = splits(1)

val lr = new GBTRegressor()

println("GBTRegressor parameters:\n" + lr.explainParams() + "\n")

lr.setMaxIter(10)
lr.setStepSize(0.1)
lr.setMaxDepth(3)
val pipeline = new Pipeline()
pipeline.setStages(Array(lr))
  
val crossval = new CrossValidator()
crossval.setEstimator(pipeline)
crossval.setEvaluator(new RegressionEvaluator())

val paramGrid = new ParamGridBuilder()
paramGrid.addGrid(lr.stepSize, Array(0.1, 0.5))
paramGrid.addGrid(lr.maxBins, Array(40, 45))
paramGrid.build()
crossval.setEstimatorParamMaps(paramGrid.build())
crossval.setNumFolds(3) // Use 3+ in practice

val cvModel = crossval.fit(training.toDF)

cvModel.transform(test.toDF)
val preds=cvModel.transform(test.toDF)

val ev= new RegressionEvaluator()
ev.setLabelCol("label")
ev.evaluate(preds)

//GBT Pipeline Output Test Double  = 0.11843286077328224 
// GBT Pipeline Training Double = 0.12049626317897658