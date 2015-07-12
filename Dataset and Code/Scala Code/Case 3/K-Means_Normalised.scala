import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.rdd.RDD


val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "/Users/avikalchhetri/Desktop/BigData/Midterm/TVcombined.txt")

val normalizer1 = new Normalizer()

// Each sample in data1 will be normalized using $L^2$ norm.
val norm_data = data.map(x => (normalizer1.transform(x.features)))

val numClusters = 2
val numIterations = 20
val clusters = KMeans.train(norm_data, numClusters, numIterations)

// Evaluate clustering by computing Within Set Sum of Squared Errors
val WSSSE = clusters.computeCost(norm_data)
println("Within Set Sum of Squared Errors = " + WSSSE)

// Save and load model
clusters.save(sc, "myModelPath")
val sameModel = KMeansModel.load(sc, "myModelPath")

//Outputs:

numClusters = 2
numIterations = 20
Sum of Squared Errors = 2605.2540840819343 

numClusters = 3
numIterations = 20
Sum of Squared Errors = 1602.0297950672048  

numClusters = 4
numIterations = 20
Sum of Squared Errors = 1133.9341159579435  

numClusters = 5
numIterations = 20
Sum of Squared Errors = 928.2588138608354  

numClusters = 6
numIterations = 20
Sum of Squared Errors = 807.6327045338741  

numClusters = 7
numIterations = 20
Sum of Squared Errors = 698.324164966572  

numClusters = 8
numIterations = 20
Sum of Squared Errors = 638.0527278734086  

numClusters = 9
numIterations = 20
Sum of Squared Errors = 583.3397620174323  

numClusters = 10
numIterations = 20
Sum of Squared Errors = 552.8560206593854 

numClusters = 11
numIterations = 20
Sum of Squared Errors = 516.3461122909365 

numClusters = 12
numIterations = 20
Sum of Squared Errors = 478.47017309247667 

numClusters = 13
numIterations = 20
Sum of Squared Errors = 468.8158253601142 

numClusters = 14
numIterations = 20
Sum of Squared Errors = 432.6391959617786 

numClusters = 15
numIterations = 20
Sum of Squared Errors = 411.56842266444505 

