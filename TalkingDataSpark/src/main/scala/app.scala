
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation._
import org.apache.spark.mllib.evaluation._

object app {

  lazy val spark: SparkSession = {
    SparkSession
      .builder()
      .master("yarn")
      //.master("local")
      .appName("TalkingData")
      .getOrCreate()
  }
  spark.sparkContext.setLogLevel("WARN")

  def main(args: Array[String]): Unit = {

    val trainSchema = StructType(
      StructField("ip", IntegerType, nullable = false) ::
      StructField("app", IntegerType, nullable = false) ::
      StructField("device", IntegerType, nullable = false) ::
      StructField("os", IntegerType, nullable = false) ::
      StructField("channel", IntegerType, nullable = false) ::
      StructField("click_time", TimestampType, nullable = false) ::
      StructField("attributed_time", TimestampType, nullable = true) ::
      StructField("is_attributed", IntegerType, nullable = true) ::
      Nil
    )

    val train  = spark.read
      .option("header", "true")
      .schema(trainSchema)
      //.csv("/Users/chosia/codersco/TalkingData/data/train_sample.csv")
      .csv("s3://coderscotalkingdata/train")
    train.createOrReplaceTempView("train")

    val features = spark.sql(
      """
        select ip, app, device, os, channel, hour(click_time) as hour, is_attributed
        from train
      """.stripMargin
    )
    val feature_cols = features.columns.diff(List("is_attributed"))
    println(feature_cols.mkString(","))
    val assembler = new VectorAssembler().setInputCols(feature_cols).setOutputCol("features")
    val full_model_input = assembler.transform(features)

    val model_input = full_model_input.sample(false, 0.01)
    model_input.rdd.cache()

    println(s"Num lines: ${model_input.count}")

    println(s"Num partitions: ${model_input.rdd.getNumPartitions}")

    //val randomForest = new RandomForestClassifier().setLabelCol("is_attributed").setFeaturesCol("features")
    val gbt = new GBTClassifier().setLabelCol("is_attributed").setFeaturesCol("features").setMaxIter(2)

    println("Creating training and test sets")

    val Array(training, test) = model_input.randomSplit(Array(0.7, 0.3))

    println("Fitting model")

    //val model = randomForest.fit(training)
    val model = gbt.fit(training)

    println("Generating predictions")

    val model_output = model.transform(test)

    val validator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")
      .setLabelCol("is_attributed")

    println("Computing AUC")

    // Compute AUC
    val auc = validator.evaluate(model_output)
    println(s"Model AUC is: $auc")

    model_output.show

    println("Computing confusion matrix")

    // Compute confusion matrix
    val predictionsAndLabels = model_output.select("prediction", "is_attributed").rdd
      .map(row => (row(0).asInstanceOf[Double], row(1).asInstanceOf[Int].toDouble))
    val metrics = new MulticlassMetrics(predictionsAndLabels)

    println(metrics.confusionMatrix.toString())

    spark.close()

  }

}
