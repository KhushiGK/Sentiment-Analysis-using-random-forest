import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.RandomForestClassifier
import com.databricks.spark.corenlp.functions._
import org.apache.spark.sql.functions._

val dataset = spark.read
  .option("delimiter", "\t")
  .csv("path/to/dataset.tsv")

val stopWords = List("the", "is", "of", "and", "a", "to", "in", "it", "that", "this", "was", "he", "she", "they", "are", "as", "with", "at", "by", "for", "on", "about", "an", "be", "or", "which", "have", "from", "him", "her", "its")
val removeStopWords = udf{(text: String) => text.split(" ").filter(!stopWords.contains(_)).mkString(" ")}
val preprocessedData = dataset.withColumn("preprocessed_text", removeStopWords(col("text")))

val lemmatizer = new Lemmatizer()
val lemmatizedData = preprocessedData.withColumn("lemmatized_text", lemmatizer.transform(col("preprocessed_text")))

val word2vec = new Word2Vec()
  .setInputCol("lemmatized_text")
  .setOutputCol("word2vec_vectors")
  .setVectorSize(10000)

val word2vecModel = word2vec.fit(lemmatizedData)
val word2vecData = word2vecModel.transform(lemmatizedData)
val (trainingData, testData) = word2vecData.randomSplit(Array(0.8, 0.2))

val randomForestClassifier = new RandomForestClassifier()
  .setFeaturesCol("word2vec_vectors")
  .setLabelCol("score")

val randomForestModel = randomForestClassifier.fit(trainingData)

val predictions = randomForestModel.transform(testData)
val accuracy = predictions.filter(col("prediction") === col("score")).count().toDouble / predictions.count().toDouble
println(s"Accuracy on testing set: ${accuracy}")

