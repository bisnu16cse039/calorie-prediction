import os
import yaml
import logging
import warnings

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ----------------- CONFIG & LOGGING ----------------- #
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Load configuration
CONFIG_PATH = os.getenv('CALORIES_CONFIG', 'config.yaml')
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

# MLflow setup
mlflow.set_tracking_uri(cfg.get('mlflow_tracking_uri', 'file:./mlruns'))
mlflow.set_experiment(cfg.get('experiment_name', 'calories_pred'))

# ----------------- UTILITY FUNCTIONS ----------------- #
def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


def build_preprocessor(df: pd.DataFrame, target: str):
    df = df.drop(columns=[target], errors='ignore')
    num_feats = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_feats = df.select_dtypes(include=['object']).columns.tolist()

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer([
        ('num', num_pipe, num_feats),
        ('cat', cat_pipe, cat_feats)
    ], remainder='passthrough')


def rmsle(y_true, y_pred):
    y_p = np.maximum(y_pred, 0)
    y_t = np.maximum(y_true, 0)
    return np.sqrt(mean_squared_error(np.log1p(y_t), np.log1p(y_p)))

# ----------------- TRAINING & EVAL ----------------- #
def train_and_evaluate(X: pd.DataFrame, y: pd.Series, model_cfg: dict, preprocessor):
    name = model_cfg['name']
    logger.info(f"Starting training: {name}")
    with mlflow.start_run(run_name=name):
        if name == 'xgboost':
            base = XGBRegressor(
                random_state=cfg['seed'],
                objective='reg:squarederror',
                tree_method='gpu_hist', gpu_id=0
            )
            pipe = Pipeline([('prep', preprocessor), ('model', base)])
            gs = GridSearchCV(pipe, model_cfg['param_grid'], cv=cfg['cv_folds'],
                              scoring='neg_mean_squared_error', verbose=1)
            gs.fit(X, y)
            best = gs.best_estimator_
            rmse = np.sqrt(-gs.best_score_)
            for p, v in gs.best_params_.items(): mlflow.log_param(p, v)
        else:
            clf = {'linear':LinearRegression(),
                   'ridge':Ridge(), 'lasso':Lasso(),
                   'rf':RandomForestRegressor(random_state=cfg['seed']),
                   'gb':GradientBoostingRegressor(random_state=cfg['seed'])}[name]
            pipe = Pipeline([('prep', preprocessor), ('model', clf)])
            scores = cross_val_score(pipe, X, y, cv=cfg['cv_folds'],
                                     scoring='neg_mean_squared_error')
            rmse = np.sqrt(-scores.mean())
            params = model_cfg.get('param_grid', 'N/A')
            logger.info(f"Params: {params}, CV RMSE = {rmse:.4f}")
            best = pipe.fit(X, y)
        mlflow.log_metric('cv_rmse', rmse)
        mlflow.sklearn.log_model(best, 'model')
        logger.info(f"Finished {name}, CV RMSE = {rmse:.4f}")
    return best

# ----------------- MAIN ----------------- #
def main():
    # Load and prepare data
    train = load_data(cfg['train_path'])
    target = cfg.get('target', 'Calories')
    X = train.drop(columns=[target])
    y = train[target]
    preprocessor = build_preprocessor(train, target)

    # Train models defined in config
    models = []
    for m in cfg['models']:
        best_model = train_and_evaluate(X, y, m, preprocessor)
        models.append((m['name'], best_model))

    # Save final model(s)
    out_dir = cfg.get('output_dir', 'models')
    os.makedirs(out_dir, exist_ok=True)
    for name, mdl in models:
        path = os.path.join(out_dir, f"{name}.pkl")
        joblib.dump(mdl, path)
        logger.info(f"Saved {name} to {path}")

if __name__ == '__main__':
    main()
