import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.SparkSession

// Initialize a Spark session
val spark = SparkSession.builder.appName("DecisionTreeRegression").getOrCreate()

// Load your JSON data into a DataFrame
val data = spark.read.json("dataset.json")

// Split data into training and testing sets (80-20 split)
val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2), seed = 42)

// Create a DecisionTreeRegressor
val dt = new DecisionTreeRegressor()
  .setLabelCol("label")
  .setFeaturesCol("features")

// Train the model
val model = dt.fit(trainingData)

// Make predictions on the testing data
val predictions = model.transform(testData)

// Evaluate the model
val evaluator = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("mae")

val mae = evaluator.evaluate(predictions)
println(s"Mean Absolute Error: $mae")

// Stop the Spark session
spark.stop()
