import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.{StructType,StructField,StringType};
import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.{Vector, Vectors}

//val sqlContext = new org.apache.spark.sql.SQLContext(sc)
// Load and parse the data


val data = sc.textFile("/Users/avikalchhetri/Desktop/BigData/Midterm/adult_withdummies.csv")

val parsedData = data.map(_.split(",")).map(p=>LabeledPoint(if(p(82)=="<=50K"){0}else{1} ,Vectors.dense(p.slice(0,82).map(_.toDouble)))).cache()


val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0).cache()
val test = splits(1)
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

val cvModel = crossval.fit(training.toDF)
val test = splits(1)
cvModel.transform(test.toDF)
val testpreds=cvModel.transform(test.toDF)
cvModel.transform(training.toDF)
val preds=cvModel.transform(training.toDF)

val ev= new BinaryClassificationEvaluator()
ev.setLabelCol("label")
ev.evaluate(preds)
ev.evaluate(testpreds)

println("Test Output is:" +ev.evaluate(testpreds))
println("Training Output is:" +ev.evaluate(preds))

//Test Output Double = 0.8925648193179296
//Training Double = 0.8930300922430976