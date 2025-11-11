import findspark
findspark.init()
findspark.find()

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


df = df.withColumn("league_name",
    when(col("league_name").isNull(), lit("Unknown")).otherwise(col("league_name"))
)

df_train_raw, df_val_raw, df_test_raw = df.randomSplit([0.8, 0.1, 0.1], seed=42)

preprocess_pipeline = get_preprocess_pipeline()
preprocess_pipeline_model = preprocess_pipeline.fit(df_train_raw)

df_train = preprocess_pipeline_model.transform(df_train_raw)
df_val   = preprocess_pipeline_model.transform(df_val_raw)
df_test  = preprocess_pipeline_model.transform(df_test_raw)

print("Train count:", df_train.count())
print("Validation count:", df_val.count())
print("Test count:", df_test.count())

def spark_to_numpy(df):
    pdf = (
        df.select("features", "overall")
          .dropna(subset=["overall"])
          .toPandas()
    )

    X = np.array([v.toArray() for v in pdf["features"]], dtype=np.float32)
    y = pdf["overall"].to_numpy().astype(np.float32)

    return X, y

Xtr, ytr = spark_to_numpy(df_train)
Xval, yval = spark_to_numpy(df_val)
Xte, yte = spark_to_numpy(df_test)

class PlayerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"features": self.X[i], "label": self.y[i]}

train_ds = PlayerDataset(Xtr, ytr)
val_ds   = PlayerDataset(Xval, yval)
test_ds  = PlayerDataset(Xte, yte)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

for i, data in enumerate(train_loader):
    print(f"Batch {i}: data['features'].shape = {data['features'].shape}")
    print(f"Batch {i}: data['label'].shape = {data['label'].shape}")
    print(f"Sample labels (first 5): {data['label'][:5]}")
    if i == 1:
        break

in_dim = train_ds[0]['features'].shape[0]
print("input dim:", in_dim)

class ShallowMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, dropout=False):
        super().__init__()
        if dropout:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class DeepMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims=(256, 128, 64), dropout=False):
        super().__init__()
        layers = []
        last_dim = in_dim
        if dropout:
            for h in hidden_dims:
                layers.append(nn.Linear(last_dim, h))
                layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
                last_dim = h
            layers.append(nn.Linear(last_dim, 1))
            self.net = nn.Sequential(*layers)
        else:
            for h in hidden_dims:
                layers.append(nn.Linear(last_dim, h))
                layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                last_dim = h
            layers.append(nn.Linear(last_dim, 1))
            self.net = nn.Sequential(*layers)
            
    def forward(self, x):
        return self.net(x).squeeze(-1)
    
# Shallow MLP
shallow_lrs = [0.005, 0.01, 0.015]
shallow_batches = [128, 256]
shallow_epochs = [20, 40]
shallow_hypers = [
    {"lr": lr, "batch_size": bs, "epochs": ep}
    for lr, bs, ep in itertools.product(shallow_lrs, shallow_batches, shallow_epochs)
]

# Deep MLP
deep_lrs = [0.001, 0.003, 0.005]
deep_batches = [128, 256]
deep_epochs = [20, 40, 60]
deep_hypers = [
    {"lr": lr, "batch_size": bs, "epochs": ep}
    for lr, bs, ep in itertools.product(deep_lrs, deep_batches, deep_epochs)
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

criterion = nn.MSELoss()

def evaluate_rmse(model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            X = batch["features"].to(device)
            y = batch["label"].to(device)
            preds = model(X)
            loss = criterion(preds, y)
            bs = y.size(0)
            total_loss += loss.item() * bs
            n += bs
    mse = total_loss / n
    rmse = mse ** 0.5
    return mse, rmse

results = []
best_shallow_rmse = float("inf")
best_shallow_cfg = None

for i, cfg in enumerate(shallow_hypers):
    lr = cfg["lr"]
    batch_size = cfg["batch_size"]
    N_epochs = cfg["epochs"]

    print(f"Shallow model {i} | lr={lr}, batch_size={batch_size}, epochs={N_epochs}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=2048, shuffle=False)

    model = ShallowMLP(in_dim, hidden_dim=64, dropout=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_train_mses = []
    epoch_val_mses = []
    epoch_train_rmses = []
    epoch_val_rmses = []

    best_val_rmse = float("inf")
    best_state_dict = None

    for epoch in range(N_epochs):
        model.train()
        running_loss = 0.0
        n = 0

        for batch in train_loader:
            X = batch["features"].to(device)
            y = batch["label"].to(device)

            preds = model(X)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            running_loss += loss.item() * bs
            n += bs

        train_mse = running_loss / n
        train_rmse = train_mse ** 0.5

        val_mse, val_rmse = evaluate_rmse(model, val_loader, device)

        epoch_train_mses.append(train_mse)
        epoch_val_mses.append(val_mse)
        epoch_train_rmses.append(train_rmse)
        epoch_val_rmses.append(val_rmse)

        print(f"Epoch {epoch+1}: train_RMSE={train_rmse:.4f}, val_RMSE={val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state_dict = model.state_dict()

    model_path = f"models/shallow_best_{i}_rmse_{best_val_rmse:.4f}.pt"
    torch.save(best_state_dict, model_path)

    results.append({
    "config": cfg,
    "best_val_rmse": float(best_val_rmse),
    "train_mse_history": epoch_train_mses,
    "val_mse_history": epoch_val_mses,
    "train_rmse_history": epoch_train_rmses,
    "val_rmse_history": epoch_val_rmses,
    "model_path": model_path,
    })

    print(f"Config {i} best val_RMSE = {best_val_rmse:.4f}")

    if best_val_rmse < best_shallow_rmse:
        best_shallow_rmse = best_val_rmse
        best_shallow_cfg = {
            **cfg,
            "best_val_rmse": float(best_val_rmse),
            "model_path": model_path,
        }

    pd.DataFrame({
    "epoch": list(range(1, N_epochs + 1)),
    "train_mse": epoch_train_mses,
    "val_mse": epoch_val_mses,
    "train_rmse": epoch_train_rmses,
    "val_rmse": epoch_val_rmses
}).to_csv(f"logs/shallow_log_{i}.csv", index=False)

print("Best Shallow MLP confi")
print(best_shallow_cfg)
print("Best val_RMSE:", best_shallow_rmse)

deep_results = []
best_deep_rmse = float("inf")
best_deep_cfg = None

for i, cfg in enumerate(deep_hypers):
    lr = cfg["lr"]
    batch_size = cfg["batch_size"]
    N_epochs = cfg["epochs"]

    print(f"\nDeep model {i} | lr={lr}, batch_size={batch_size}, epochs={N_epochs}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=2048, shuffle=False)

    model = DeepMLP(in_dim, hidden_dims=(256, 128, 64), dropout=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_train_mses = []
    epoch_val_mses = []
    epoch_train_rmses = []
    epoch_val_rmses = []

    best_val_rmse = float("inf")
    best_state_dict = None

    for epoch in range(N_epochs):
        model.train()
        running_loss = 0.0
        n = 0

        for batch in train_loader:
            X = batch["features"].to(device)
            y = batch["label"].to(device)

            preds = model(X)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            running_loss += loss.item() * bs
            n += bs

        train_mse = running_loss / n
        train_rmse = train_mse ** 0.5

        val_mse, val_rmse = evaluate_rmse(model, val_loader, device)

        epoch_train_mses.append(train_mse)
        epoch_val_mses.append(val_mse)
        epoch_train_rmses.append(train_rmse)
        epoch_val_rmses.append(val_rmse)

        print(f"Epoch {epoch+1}: train_RMSE={train_rmse:.4f}, val_RMSE={val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state_dict = model.state_dict()

    model_path = f"models/deep_best_{i}_rmse_{best_val_rmse:.4f}.pt"
    torch.save(best_state_dict, model_path)

    log_path = f"logs/deep_log_{i}.csv"
    pd.DataFrame({
        "epoch": list(range(1, N_epochs + 1)),
        "train_mse": epoch_train_mses,
        "val_mse": epoch_val_mses,
        "train_rmse": epoch_train_rmses,
        "val_rmse": epoch_val_rmses
    }).to_csv(log_path, index=False)

    deep_results.append({
        "config": cfg,
        "best_val_rmse": float(best_val_rmse),
        "train_mse_history": epoch_train_mses,
        "val_mse_history": epoch_val_mses,
        "train_rmse_history": epoch_train_rmses,
        "val_rmse_history": epoch_val_rmses,
        "model_path": model_path,
        "log_path": log_path
    })

    print(f"Config {i} best val_RMSE = {best_val_rmse:.4f}")

    if best_val_rmse < best_deep_rmse:
        best_deep_rmse = best_val_rmse
        best_deep_cfg = {
            **cfg,
            "best_val_rmse": float(best_val_rmse),
            "model_path": model_path,
            "log_path": log_path,
        }

print("Best Deep MLP config:")
print(best_deep_cfg)
print("Best val_RMSE:", best_deep_rmse)


criterion = nn.MSELoss()
total_loss = 0.0
n = 0

test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)

def eval_on_test(model, loader, device):
    mse, rmse = evaluate_rmse(model, loader, device)
    return mse, rmse

shallow_model = ShallowMLP(in_dim, hidden_dim=64, dropout=False).to(device)
shallow_model.load_state_dict(torch.load(best_shallow_cfg["model_path"], map_location=device))
shallow_model.eval()

shallow_test_mse, shallow_test_rmse = eval_on_test(shallow_model, test_loader, device)
print(f"[Shallow Best] val_RMSE = {best_shallow_cfg['best_val_rmse']:.4f}, "
      f"test_MSE = {shallow_test_mse:.4f}, test_RMSE = {shallow_test_rmse:.4f}")

deep_model = DeepMLP(in_dim, hidden_dims=(256, 128, 64), dropout=False).to(device)
deep_model.load_state_dict(torch.load(best_deep_cfg["model_path"], map_location=device))
deep_model.eval()

deep_test_mse, deep_test_rmse = eval_on_test(deep_model, test_loader, device)
print(f"[Deep Best] val_RMSE = {best_deep_cfg['best_val_rmse']:.4f}, "
      f"test_MSE = {deep_test_mse:.4f}, test_RMSE = {deep_test_rmse:.4f}")
