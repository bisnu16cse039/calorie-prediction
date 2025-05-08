# Calories Prediction Project

This project provides a baseline solution for predicting calories burned based on given features such as Sex, Age, Height, Weight, Duration, Heart Rate, and Body Temperature. The evaluation metric is RMSLE (Root Mean Squared Logarithmic Error).

## Project Structure

```
calories_prediction/
│
├── main.py              # Main script for running the pipeline
├── requirements.txt     # Project dependencies
├── data/                # Data directory
│   └── data.csv         # Input data file
│
├── models/              # Saved model files
│   └── .gitkeep         
│
├── mlruns/              # MLflow tracking data
│
├── notebooks/           # Jupyter notebooks for exploration
│   └── exploratory_analysis.ipynb
│
└── README.md            # Project documentation
```

## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the baseline model:

```bash
python main.py --train_path data/train.csv --test_path data/test.csv --model random_forest
```

### Available Models

The following models are implemented:
- `linear_regression`: Simple Linear Regression
- `ridge`: Ridge Regression
- `lasso`: Lasso Regression
- `random_forest`: Random Forest Regressor (default)
- `gradient_boosting`: Gradient Boosting Regressor

### Hyperparameter Tuning

To run with hyperparameter tuning:

```bash
python main.py --data_path data/data.csv --model random_forest --tune
```

### Experiment Tracking

View MLflow dashboard:

```bash
mlflow ui
```

Then open your browser at http://localhost:5000 to view the experiments and runs.

## Adding New Models

To add a new model, modify the `main.py` file:

1. Import your model:
```python
from sklearn.ensemble import XGBRegressor  # For example
```

2. Add the model to the `models` dictionary in the `train_model` function:
```python
models = {
    # Existing models...
    'xgboost': XGBRegressor(random_state=42)
}
```

3. Add hyperparameter grid for tuning in the `hyperparameter_tuning` function:
```python
param_grids = {
    # Existing grids...
    'xgboost': {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7]
    }
}
```

4. Update the CLI choices:
```python
parser.add_argument('--model', type=str, default='random_forest', 
                   choices=['linear_regression', 'ridge', 'lasso', 'random_forest', 'gradient_boosting', 'xgboost'],
                   help='Model to use for prediction')
```

## Feature Engineering

To add new features:

1. Create a feature engineering function:
```python
def engineer_features(df):
    # Create new features
    df['bmi'] = df['Weight'] / ((df['Height']/100) ** 2)
    df['intensity'] = df['Heart_Rate'] / df['Age']
    return df
```

2. Add it to the preprocessing pipeline in the `preprocess_data` function:
```python
# Before splitting into train/test
df = engineer_features(df)
```

## Best Practices

1. **Experiment Organization**: Use meaningful experiment names and run names in MLflow.
2. **Versioning**: Keep track of model versions and dataset versions.
3. **Parameter Tuning**: Start with a coarse grid search, then refine around promising values.
4. **Feature Selection**: Use feature importance to identify and focus on the most relevant features.
5. **Cross-Validation**: Always use cross-validation to ensure robust model evaluation.