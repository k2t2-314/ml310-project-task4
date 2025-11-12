import findspark
import os

import pyspark
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import col, count, desc
from pyspark.sql.functions import avg
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import row_number
import matplotlib.pyplot as plt

# Docker set up
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "20030218")

JDBC_URL = f"jdbc:postgresql://{DB_HOST}:{DB_PORT}/{DB_NAME}"
DB_PROPS = {
    "user": DB_USER,
    "password": DB_PASSWORD,
    "driver": "org.postgresql.Driver"
}

spark = (
    SparkSession.builder
    .appName("Project-1-Task2")
    .master("local[*]")
    .config("spark.driver.host", "0.0.0.0")
    .config("spark.jars", "/app/jars/postgresql-42.7.3.jar")
    .getOrCreate()
)

df = spark.read.jdbc(
    url=JDBC_URL,
    table="fifa.all_data",
    properties=DB_PROPS
)

print("Schema:")
df.printSchema()
print("Sample rows:")
df.select("short_name", "long_name", "age", "club_name", "nationality_name") \
  .show(5, truncate=False)

# Filter and save male-only data
male_df = df.filter(df.gender == "male")

male_df.write.jdbc(
    url=JDBC_URL,
    table="fifa.man_player_data",
    mode="overwrite",
    properties=DB_PROPS
)

df = male_df

def clubs_with_contracts(df, X, Y, Z):
    # X = year
    # Y = top N clubs
    # Z = contract end year threshold
    return (df.filter((col("resource_year") == X) & 
                      (col("club_contract_valid_until") >= Z))
              .groupBy("club_name")
              .agg(count("*").alias("player_count"))
              .orderBy(desc("player_count"))
              .limit(Y))

clubs_with_contracts(df, 2020, 5, 2023).show()

def clubs_by_avg_age(df, X, Y, highest=True):
    # X = top N clubs
    # Y = target year   
    if X <= 0:
        raise ValueError("X must be a positive integer.")
    if Y < 2015 or Y > 2022:
        raise ValueError("Y must be between 2015 and 2022.")

    avg_age_df = (df.filter(col("resource_year") == Y)
                    .groupBy("club_name")
                    .agg(avg("age").alias("avg_age"))
                    .filter(col("club_name").isNotNull()))

    order_col = col("avg_age").desc() if highest else col("avg_age").asc()

    topX = avg_age_df.orderBy(order_col).limit(X).collect()
    if not topX:
        return None
    threshold = topX[-1]["avg_age"]
    if highest:
        result = avg_age_df.filter(col("avg_age") >= threshold).orderBy(order_col)
    else:
        result = avg_age_df.filter(col("avg_age") <= threshold).orderBy(order_col)
    
    return result

clubs_by_avg_age(df, 5, 2020, highest=True).show()

def most_popular_nationality(df):
    nationality_counts = (df.groupBy("resource_year", "nationality_name")
                            .agg(count("*").alias("player_count")))
    windowSpec = Window.partitionBy("resource_year").orderBy(desc("player_count"))
    top_nationalities = (nationality_counts
                         .withColumn("rank", row_number().over(windowSpec))
                         .filter(col("rank") == 1)
                         .orderBy("resource_year"))
    return top_nationalities

most_popular_nationality(df).show()

def nationality_histogram(df, top_n=20):

    dedup = df.dropDuplicates(["sofifa_id"]) \
              .groupBy("nationality_name") \
              .agg(count("*").alias("unique_players")) \
              .orderBy(col("unique_players").desc())

    pdf = dedup.limit(top_n).toPandas()

    plt.figure(figsize=(8, 6))
    plt.bar(pdf["nationality_name"], pdf["unique_players"])
    plt.title(f"Top {top_n} Nationalities (Unique Players Across All Years)")
    plt.ylabel("Number of Unique Players")
    plt.xlabel("Nationality")
    plt.xticks(rotation=90, ha="right")
    plt.show()

    return dedup

nationality_histogram(df, top_n=15)