import findspark

import pyspark
import psycopg2
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import lit
from pyspark.sql.types import IntegerType, StringType
from pyspark.sql.functions import col, when, regexp_extract, to_date
from pyspark.sql.types import IntegerType, DoubleType, DateType, StringType
import os

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "20030218")


spark = (
    SparkSession.builder
    .appName("Project-1")
    .master("local[*]")
    .config("spark.driver.host", "0.0.0.0")
    .config("spark.jars", "/app/jars/postgresql-42.7.3.jar")
    .getOrCreate()
)

CSV_list = [
    "data/players_15.csv",
    "data/players_16.csv",
    "data/players_17.csv",
    "data/players_18.csv",
    "data/players_19.csv",
    "data/players_20.csv",
    "data/players_21.csv",
    "data/players_22.csv",
    "data/female_players_16.csv",
    "data/female_players_17.csv",
    "data/female_players_18.csv",
    "data/female_players_19.csv",
    "data/female_players_20.csv",
    "data/female_players_21.csv",
    "data/female_players_22.csv"

]
YEAR_list = [2015,2016,2017,2018,2019,2020,2021,2022,2016,2017,2018,2019,2020,2021,2022]
GENDER_list = ["male","male","male","male","male","male","male","male","female","female","female","female","female","female","female"]

db_properties={
    "user":"postgres",
    "password":"20030218",
    "url":f"jdbc:postgresql://{DB_HOST}:{DB_PORT}/{DB_NAME}",
    "dbtable" : "fifa.all_data",
    "driver": "org.postgresql.Driver"
}

def reset_fifa_all_data():
    conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("CREATE SCHEMA IF NOT EXISTS fifa;")
    cur.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'fifa'
                  AND table_name   = 'all_data'
            ) THEN
                TRUNCATE TABLE fifa.all_data;
            END IF;
        END;
        $$;
    """)

    cur.close()
    conn.close()
    print("Checked fifa.all_data, truncated if existed.")

# Clean the data
def clean_and_cast_fifa_data(df):
    def safe_cast_to_numeric(df, col_name, target_type):
        return df.withColumn(
            col_name,
            when(col(col_name).isin("NULL", "", " ", "NaN"), None)
            .otherwise(col(col_name))
            .cast(target_type)
        )    
    
    double_cols = ["value_eur", "wage_eur", "release_clause_eur"]
    
    for col_name in double_cols:
        df = safe_cast_to_numeric(df, col_name, DoubleType())
        
    integer_cols = [
        "sofifa_id", "overall", "potential", "age", "height_cm", "weight_kg",
        "league_level", "club_jersey_number", "nationality_id", "nation_jersey_number", 
        "weak_foot", "skill_moves", "international_reputation", "pace", "shooting", 
        "passing", "dribbling", "defending", "physic", "attacking_crossing", 
        "attacking_finishing", "attacking_heading_accuracy", "attacking_short_passing", 
        "attacking_volleys", "skill_dribbling", "skill_curve", "skill_fk_accuracy", 
        "skill_long_passing", "skill_ball_control", "movement_acceleration", 
        "movement_sprint_speed", "movement_agility", "movement_reactions", 
        "movement_balance", "power_shot_power", "power_jumping", "power_stamina", 
        "power_strength", "power_long_shots", "mentality_aggression", 
        "mentality_interceptions", "mentality_positioning", "mentality_vision", 
        "mentality_penalties", "defending_marking_awareness", 
        "defending_standing_tackle", "defending_sliding_tackle", "goalkeeping_diving", 
        "goalkeeping_handling", "goalkeeping_kicking", "goalkeeping_positioning", 
        "goalkeeping_reflexes", "goalkeeping_speed", "club_contract_valid_until"
    ]
    
    for col_name in integer_cols:
        df = safe_cast_to_numeric(df, col_name, IntegerType())
        
    df = safe_cast_to_numeric(df, "club_team_id", IntegerType())
    df = safe_cast_to_numeric(df, "nation_team_id", IntegerType())
    df = safe_cast_to_numeric(df, "mentality_composure", IntegerType())
    
    date_cols = ["dob", "club_joined"]
    for col_name in date_cols:
        df = df.withColumn(
            col_name,
            to_date(col(col_name))
        )
        
    string_cols = [
        "player_url", "short_name", "long_name", "player_positions", "club_name", 
        "league_name", "club_position", "club_loaned_from", "nationality_name",
        "nation_position", "preferred_foot", "work_rate", "body_type", "real_face", 
        "player_tags", "player_traits", "player_face_url", "club_logo_url", 
        "club_flag_url", "nation_logo_url", "nation_flag_url"
    ]
    
    for col_name in string_cols:
        df = df.withColumn(col_name, col(col_name).cast(StringType()))
        
    return df

reset_fifa_all_data()

for i in range(len(CSV_list)):
    # Load CSV to Spark dataframe
    df = spark.read.csv(CSV_list[i],header=True,inferSchema=True)
    
    df_cleaned = clean_and_cast_fifa_data(df)
    
    # Add YEAR column
    YEAR = YEAR_list[i]
    df_1 = df_cleaned.withColumn(
        "resource_year", 
        lit(YEAR).cast(IntegerType())
    )
    
    # Add GENDER column
    GENDER = GENDER_list[i]
    df_2 = df_1.withColumn(
        "gender", 
        lit(GENDER).cast(StringType())
    )

    # Write Spark dataframe to PSQL
    df_2.write.jdbc(
    url=db_properties["url"],
    table="fifa.all_data",
    mode="append",
    properties=db_properties
    )

    
    print(f"{CSV_list[i]} uploaded")

print("-----------------------------------")
print("All CSV tables uploaded to PSQL!")
print("-----------------------------------")
print("Show Schema: ")
df_read = spark.read.format("jdbc").options(**db_properties).load()
df_read.printSchema()
print("-----------------------------------")



