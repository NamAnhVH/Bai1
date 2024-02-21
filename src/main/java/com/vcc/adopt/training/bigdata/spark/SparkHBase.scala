import com.vcc.adopt.config.ConfigPropertiesLoader

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.sql.{DataFrame, Row, SparkSession}



object SparkHBase {

  val spark: SparkSession = SparkSession.builder().getOrCreate()
  spark.sparkContext.setLogLevel("WARN")
  spark.conf.set("spark.sql.debug.maxToStringFields", 10000)

  private val bai1 = ConfigPropertiesLoader.getYamlConfig.getProperty("bai1")

  private def findKmean(k : Int): Unit = {
    val data: DataFrame = spark.read.option("header", "false").csv(bai1)
      .toDF("x", "y")
    val kmeans = new KMeans()
      .setK(k) // Số lượng cụm
      .setSeed(1L) // Seed để tái tạo kết quả

    val model = kmeans.fit(data)

    val centroids = model.clusterCenters

    centroids.foreach(println)

    val predictions = model.transform(data)

    predictions.show()

  }

  def main(args: Array[String]): Unit = {
    findKmean(3);
  }
}