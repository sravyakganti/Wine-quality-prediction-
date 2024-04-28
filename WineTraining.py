import boto3
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression  # Not used in training, but included for completeness
from pyspark.mllib.util import MLUtils  # Not used in this code, but might be for future extensibility
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import DoubleType, IntegerType  


def main():
    # Spark session initialization
    spark = SparkSession.builder.appName("WineQuality Training").getOrCreate()

    # Training data path
    train_path = "s3a://dataset-programming-assignment-2/TrainingDataset.csv"
    print("Importing:", train_path)

    # Determine model path based on S3 structure
    parsed_url = urlparse(train_path)
    s3_model_path = os.path.join(parsed_url.netloc, os.path.dirname(parsed_url.path), "models")
    print(">>>> Model Path set:", s3_model_path)

    # Load and preprocess data
    df_train = spark.read.csv(train_path, header=True, sep=";")
    df_train = preprocess_data(df_train)

    # Convert data to LabeledPoint format for training
    df_train = df_train.rdd.map(lambda row: LabeledPoint(row[-1], row[:-1]))

    # Train Decision Tree model
    model_dt = train_decision_tree(df_train)
    save_model(model_dt, s3_model_path + "/model_dt.model")
    print(">>>>> DecisionTree model saved")

    # Train Random Forest model
    model_rf = train_random_forest(df_train)
    save_model(model_rf, s3_model_path + "/model_rf.model")
    print(">>>>> RandomForest model saved")

    print("Data Training Completed")

    spark.stop()  # Stop Spark session


def preprocess_data(df):
    """
    Preprocesses the training data by:
        - Renaming the label column
        - Removing quotes from column names
        - Casting features to appropriate types (double or integer)
    """

    df = df.withColumnRenamed('""""quality"""""', "myLabel")
    for col in df.columns:
        df = df.withColumnRenamed(col, col.replace('"', ''))

    for idx, col_name in enumerate(df.columns):
        if idx not in [6 - 1, 7 - 1, len(df.columns) - 1]:
            df = df.withColumn(col_name, col(col_name).cast(DoubleType()))
        else:
            df = df.withColumn(col_name, col(col_name).cast(IntegerType()))

    return df


def train_decision_tree(data):
    """
    Trains a DecisionTreeClassifier model with specified parameters.
    """

    model_dt = DecisionTreeClassifier(numClasses=10, categoricalFeaturesInfo={},
                                     impurity='gini', maxDepth=10, maxBins=32)
    model_dt = model_dt.fit(data)
    print("Model - DecisionTree Created")
    return model_dt


def train_random_forest(data):
    """
    Trains a RandomForestClassifier model with specified parameters.
    """

    model_rf = RandomForestClassifier(numClasses=10, categoricalFeaturesInfo={},
                                     numTrees=10, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=10, maxBins=32)
    model_rf = model_rf.fit(data)
    print("Model - RandomForest Created")
    return model_rf


def save_model(model, path):
    """
    Saves the model to the specified S3 path.
    """

    model.write().overwrite().save(path)
    print(f">>>>> Model saved to: {path}")


if __name__ == "__main__":
    main()
