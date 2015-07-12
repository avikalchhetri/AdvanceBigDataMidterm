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
//val normalizer2 = new Normalizer(p = Double.PositiveInfinity)

// Each sample in data1 will be normalized using $L^2$ norm.
val norm_data = data.map(x => (normalizer1.transform(x.features)))
val gmm = new GaussianMixture().setK(2).run(norm_data)

// Save and load model
gmm.save(sc, "myGMMModel")
val sameModel = GaussianMixtureModel.load(sc, "myGMMModel")

// output parameters of max-likelihood model
for (i <- 0 until gmm.k) {
  println("weight=%f\nmu=%s\nsigma=\n%s\n" format
    (gmm.weights(i), gmm.gaussians(i).mu, gmm.gaussians(i).sigma))
}


//java.lang.OutOfMemoryError: Java heap space