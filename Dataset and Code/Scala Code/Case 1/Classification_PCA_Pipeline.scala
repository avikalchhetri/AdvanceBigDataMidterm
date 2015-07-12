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
import org.apache.spark.mllib.feature.PCA

val data = sc.textFile("/Users/avikalchhetri/Desktop/BigData/Midterm/YearPredictionMSD.txt")
val parsedData = data.map(_.split(",")).map(p=>LabeledPoint((if(p(0).toInt<1965){0.0}else{1.0}),Vectors.dense(p.slice(1,91).map(_.toDouble)))).cache()
val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)

val training = splits(0).cache()
val test = splits(1)

val pca = new PCA(training.first().features.size/2).fit(parsedData.map(_.features))
val training_pca = training.map(p => p.copy(features = pca.transform(p.features)))
val test_pca = test.map(p => p.copy(features = pca.transform(p.features)))

val lr = new LogisticRegression()
println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

lr.setMaxIter(10)
lr.setRegParam(0.01)
lr.setThreshold(0.55)

val pipeline = new Pipeline()
pipeline.setStages(Array(lr))
  
val crossval = new CrossValidator()
crossval.setEstimator(pipeline)
crossval.setEvaluator(new BinaryClassificationEvaluator)

val paramGrid = new ParamGridBuilder()
paramGrid.addGrid(lr.threshold, Array(0.2))
paramGrid.addGrid(lr.regParam, Array(0.1, 0.01))
paramGrid.build()
crossval.setEstimatorParamMaps(paramGrid.build())
crossval.setNumFolds(3) // Use 3+ in practice

val cvModel = crossval.fit(training_pca.toDF)

cvModel.transform(test_pca.toDF)
val preds=cvModel.transform(test_pca.toDF)

val ev= new BinaryClassificationEvaluator()
ev.setLabelCol("label")
ev.evaluate(preds)

//Classification PCA Pipeline Output Test Double = 0.8545708480214418
//Classification PCA Training Double = 0.8574986434563465
 