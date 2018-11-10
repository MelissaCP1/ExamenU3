import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression

//1
val spar = SparkSession.builder().getOrCreate()

//2
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

//3
val spark = SparkSession.builder().getOrCreate()

//4
import org.apache.spark.ml.clustering.KMeans

//5
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Wholesale.csv")
df.show()
//7
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

//6

val feature_data = df.select($"Fresh",$"Milk", $"Grocery", $"Frozen", $"Detergents_Paper",$"Delicassen")


//8
 val assembler = new VectorAssembler().setInputCols(Array("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")).setOutputCol("features")


//9
val output = assembler.transform(df).select($"features")

//10
val kmeans = new KMeans().setK(2).setSeed(1L)
val model = kmeans.fit(output)

//11
val WSSE = model.computeCost(output)
println(s"Within set sum of Squared Errors = $WSSE")

//12
println("Cluster Centers: ")
model.clusterCenters.foreach(println)
