import findspark
import pyspark
from pyspark.sql import SparkSession, SQLContext
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.feature import Imputer,StandardScaler,StringIndexer, OneHotEncoder,VectorAssembler

from pyspark.sql.functions import *
from pyspark.sql.types import * 
import numpy as np

from pyspark.ml.linalg import VectorUDT
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import torch
from torch import nn

import itertools

import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score

import os
import pandas as pd


from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder,CrossValidator
from pyspark.ml.regression import RandomForestRegressor

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "20030218")

db_properties = {
    "user": DB_USER,
    "password": DB_PASSWORD,
    "url": f"jdbc:postgresql://{DB_HOST}:{DB_PORT}/{DB_NAME}",
    "dbtable": "fifa.man_player_data",
    "driver": "org.postgresql.Driver"
}

spark = (
    SparkSession.builder
    .appName("Project-1-ML")
    .master("local[*]")
    .config("spark.driver.host", "0.0.0.0")
    .config("spark.jars", "/app/jars/postgresql-42.7.3.jar")
    .getOrCreate()
)

df = spark.read.format("jdbc").options(**db_properties).load()

ordinal_cols = [
    "league_level","weak_foot", "skill_moves","international_reputation"
]

nominal_cols = [
    "preferred_foot", "work_rate", "body_type",
    "league_name", "nationality_name"
]

continuous_cols = [
    "value_eur","wage_eur","age","height_cm","weight_kg",
    "pace","shooting","passing","dribbling","defending","physic",
    "attacking_crossing","attacking_finishing","attacking_heading_accuracy",
    "attacking_short_passing","attacking_volleys",
    "skill_dribbling","skill_curve","skill_fk_accuracy",
    "skill_long_passing","skill_ball_control",
    "movement_acceleration","movement_sprint_speed","movement_agility",
    "movement_reactions","movement_balance",
    "power_shot_power","power_jumping","power_stamina","power_strength","power_long_shots",
    "mentality_aggression","mentality_interceptions","mentality_positioning","mentality_vision","mentality_penalties","mentality_composure",
    "defending_marking_awareness","defending_standing_tackle","defending_sliding_tackle",
    "goalkeeping_diving","goalkeeping_handling","goalkeeping_kicking",
    "goalkeeping_positioning","goalkeeping_reflexes"
]

useless_cols= [
    'sofifa_id', 'player_url', 'short_name', 'long_name', 
    'player_positions','dob', 'club_team_id', 
    'club_name', 'club_position', 'club_jersey_number', 
    'club_loaned_from', 'club_joined', 'club_contract_valid_until',
    'nationality_id', 'nation_team_id', 'nation_position', 
    'nation_jersey_number', 'real_face', 'release_clause_eur', 
    'player_tags', 'player_traits', 'goalkeeping_speed', 
    'player_face_url', 'club_logo_url', 'club_flag_url', 
    'nation_logo_url', 'nation_flag_url', 'resource_year'
]

position_cols = [
    "ls","st","rs","lw","lf","cf","rf","rw",
    "lam","cam","ram","lm","lcm","cm","rcm","rm",
    "lwb","ldm","cdm","rdm","rwb","lb","lcb","cb","rcb","rb","gk"
]

corelated_cols = ['skill_ball_control', 'defending_sliding_tackle', 
                  'defending_marking_awareness', 'defending_standing_tackle', 
                  'goalkeeping_positioning', 'goalkeeping_kicking', 
                  'goalkeeping_handling', 'goalkeeping_reflexes', 
                  'movement_sprint_speed','potential','value_eur', 'wage_eur']

cols_to_impute= cols_to_impute= [
    'league_level', 'pace', 'shooting', 
    'passing', 'dribbling', 'defending', 'physic', 'mentality_composure'
]

target = ["overall"]


numeric_cols = continuous_cols+ordinal_cols

class PositionCaster(Transformer):
    def __init__(self):
        super().__init__()
        self.position_cols = position_cols

    def _transform(self,dataset):
        output_df = dataset
        for col_name in position_cols:
            output_df = output_df.withColumn(col_name,regexp_replace(col(col_name), "([+-].*)$", "").cast(DoubleType()))
        return output_df

class FeatureTypeCaster(Transformer):
    def __init__(self):
        super().__init__()
        self.numeric_cols = numeric_cols

    def _transform(self,dataset):
        output_df = dataset
        for col_name in numeric_cols:
            output_df = output_df.withColumn(col_name,col(col_name).cast(DoubleType()))
        return output_df

class ColumnDropper(Transformer):
    def __init__(self, columns_to_drop = None):
        super().__init__()
        self.columns_to_drop = columns_to_drop

    def _transform(self, dataset):
        output_df = dataset
        for col_name in self.columns_to_drop:
            output_df = output_df.drop(col_name)
        return output_df
    

def get_preprocess_pipeline():
    # Stage where position cols are converted into DoubleType
    stage_position_casted = PositionCaster()

    # Stage where columns are casted as appropriate types
    stage_typecaster = FeatureTypeCaster()

    # Stage where imputs the missing numeric values
    stage_imputer = Imputer(
        inputCols=cols_to_impute,
        outputCols=cols_to_impute,
        strategy = "median"
    )

    # Stage where nominal columns are transformed to index columns using StringIndexer
    nominal_id_cols = [x+"_index" for x in nominal_cols]
    stage_nominal_indexer = StringIndexer(
        inputCols = nominal_cols,
        outputCols = nominal_id_cols,
        handleInvalid="keep"
    )

    # Stage where the index columns are further transformed using OneHotEncoder
    nominal_onehot_cols = [x+"_encoded" for x in nominal_cols]
    stage_nominal_onehot_encoder = OneHotEncoder(
        inputCols = nominal_id_cols,
        outputCols = nominal_onehot_cols
    )

    # Stage where all relevant features are assembled into a vector (and dropping a few)
    feature_cols = (continuous_cols + nominal_onehot_cols + ordinal_cols +
                    position_cols)
    for col_name in corelated_cols:
        if col_name in feature_cols:
            feature_cols.remove(col_name)
            
    stage_vector_assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="vectorized_features"
    )

    # Stage where we scale the columns
    stage_scaler = StandardScaler(
        inputCol='vectorized_features',
        outputCol = 'features'
    )

    # Drop all unnecessary columns, only keeping the 'features' and 'outcome'
    columns_to_drop=(
        nominal_cols + nominal_id_cols + nominal_onehot_cols + 
        continuous_cols + ['vectorized_features'] + position_cols +
        useless_cols + ordinal_cols
    )
    stage_column_dropper = ColumnDropper(columns_to_drop)

    # # Create label
    # stage_label_creator = SQLTransformer(
    #     statement="SELECT *, overall AS label FROM __THIS__"
    # )

    # Connect the columns into a pipeline
    pipeline = Pipeline(stages=[
        stage_position_casted,
        stage_typecaster,
        stage_imputer,
        stage_nominal_indexer,
        stage_nominal_onehot_encoder,
        stage_vector_assembler,
        stage_scaler,
        stage_column_dropper
    ])

    return pipeline

preprocess_pipeline=get_preprocess_pipeline()
preprocess_pipeline_model=preprocess_pipeline.fit(df)

df_preprocessed = preprocess_pipeline_model.transform(df)

# Split Training, Validating and Testing Dataset
df_train, df_val, df_test = df_preprocessed.randomSplit([0.8,0.1,0.1])
print(f"Train count: {df_train.count()}")
print(f"Validation count: {df_val.count()}")
print(f"Test count: {df_test.count()}")

outdir = "/app/out/spark_ml"
df_train.write.mode("overwrite").parquet(f"{outdir}/train")
df_val.write.mode("overwrite").parquet(f"{outdir}/val")
df_test.write.mode("overwrite").parquet(f"{outdir}/test")

df_train = spark.read.parquet(f"{outdir}/train")
df_val = spark.read.parquet(f"{outdir}/val")
df_test = spark.read.parquet(f"{outdir}/test")


# Cross-Validation and ParamGrid
lr = LinearRegression(featuresCol="features", labelCol="overall")
# lr_paramGrid=(ParamGridBuilder()
#              .addGrid(lr.regParam,[0.01,0.1])
#              .addGrid(lr.maxIter,[5,20])
#              .build())

lr_paramGrid=(ParamGridBuilder()
             .addGrid(lr.regParam,[0.1])
             .addGrid(lr.maxIter,[10])
             .build())

evaluator_rmse = RegressionEvaluator(predictionCol='prediction',
                                     labelCol='overall', metricName='rmse')

lr_cv=CrossValidator(estimator=lr,
                     estimatorParamMaps=lr_paramGrid,
                     evaluator=evaluator_rmse,
                     numFolds=4)

# Train
cv_model = lr_cv.fit(df_train)

best_model = cv_model.bestModel
print("Best regParam:", best_model._java_obj.getRegParam())
print("Best maxIter:", best_model._java_obj.getMaxIter())

# Evaluate the training Result
cv_pred_train = best_model.transform(df_train)
cv_pred_val = best_model.transform(df_val)
cv_pred_test = best_model.transform(df_test)

evaluator_r2 = RegressionEvaluator(predictionCol='prediction',
                                     labelCol='overall', metricName='r2')

rmse_train = evaluator_rmse.evaluate(cv_pred_train)
rmse_val = evaluator_rmse.evaluate(cv_pred_val)
rmse_test = evaluator_rmse.evaluate(cv_pred_test)
r2_train = evaluator_r2.evaluate(cv_pred_train)
r2_val = evaluator_r2.evaluate(cv_pred_val)
r2_test = evaluator_r2.evaluate(cv_pred_test)

print("For LinearRegression: ")
print(f"Train RMSE: {rmse_train}, Train R2: {r2_train}")
print(f"Validation RMSE: {rmse_val}, Validation R2: {r2_val}")
print(f"Test RMSE: {rmse_test}, Validation R2: {r2_test}")


rf = RandomForestRegressor(featuresCol="features",labelCol="overall")

# Cross-Validation and ParamGrid
# rf_paramGrid=(ParamGridBuilder()
#              .addGrid(rf.numTrees,[10,20])
#              .addGrid(rf.maxDepth,[5,10])
#              .build())

rf_paramGrid=(ParamGridBuilder()
             .addGrid(rf.numTrees,[20])
             .addGrid(rf.maxDepth,[10])
             .build())

evaluator_rmse = RegressionEvaluator(predictionCol='prediction',
                                     labelCol='overall', metricName='rmse')

evaluator_r2 = RegressionEvaluator(predictionCol='prediction',
                                     labelCol='overall', metricName='r2')

rf_cv=CrossValidator(estimator=rf,estimatorParamMaps=rf_paramGrid,
                     evaluator=evaluator_rmse,numFolds=4)

rf_cv_model = rf_cv.fit(df_train)


