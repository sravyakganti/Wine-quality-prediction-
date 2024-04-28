import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import SparkSession


def main():
    spark = SparkSession.builder.appName("WineQualityTesting").getOrCreate()

    # Load models (modify paths as needed)
    model_dt_path = "s3a://"+bucket_name+"/models/model_dt.model"
    model_rf_path = "s3a://"+bucket_name+"/models/model_rf.model"
    model_dt = LinearRegressionModel.load(model_dt_path)
    model_rf = RandomForestClassificationModel.load(model_rf_path)
    print("Models loaded successfully.")

    # Load validation data (modify bucket name and file key)
    bucket_name = "sravya-cs643"
    file_key = "ValidationDataset.csv"
    validation_path = f"s3a://{bucket_name}/{file_key}"
    validation = spark.read.csv(validation_path, inferSchema=True, header=True, sep=";")

    # Preprocess data (adapt based on data types)
    validation = validation.withColumnRenamed('"quality"', "label")  # Assuming "quality" is the label
    clean_columns = [col(c).cast("double") for c in validation.columns[:-1]]  # Cast features to doubles
    validation = validation.select(*clean_columns, "label")
    print("Data preprocessed.")

    # Make predictions
    validation_dt = validation.rdd.map(lambda row: (row[-1], list(row[:-1])))
    predictions_dt = model_dt.transform(validation).select("prediction").rdd.flatMap(lambda x: x)
    labels_and_predictions_dt = validation_dt.zip(predictions_dt)

    validation_rf = validation.rdd.map(lambda row: (row[-1], list(row[:-1])))
    predictions_rf = model_rf.transform(validation).select("prediction").rdd.flatMap(lambda x: x)
    labels_and_predictions_rf = validation_rf.zip(predictions_rf)

    # Evaluate models using RegressionEvaluator for linear regression and MulticlassClassificationEvaluator for random forest
    # (Modify based on your evaluation metrics)
    evaluator_dt = RegressionEvaluator(metricName="rmse")  # Assuming RMSE for linear regression
    rmse_dt = evaluator_dt.evaluate(labels_and_predictions_dt)
    print(f"Decision Tree Model - RMSE: {rmse_dt:.4f}")

    # evaluator_rf = MulticlassClassificationEvaluator()  # Uncomment if using MulticlassClassificationEvaluator
    # f1_rf = evaluator_rf.evaluate(labels_and_predictions_rf)
    # print(f"Random Forest Model - F1 Score: {f1_rf:.4f}")

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
