# House Price Prediction

Machine learning regression model for predicting house prices using LightGBM.

## Dataset

Training dataset contains 1460 observations with 80 features including:
- `SalePrice` - Target variable
- `OverallQual` - Overall material and finish quality
- `GrLivArea` - Above grade living area
- `GarageArea` - Garage size
- `TotalBsmtSF` - Basement area
- `YearBuilt` - Construction year
- Additional features related to house characteristics, location, and condition

## Project Structure

```
House-Price-Prediction/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration and constants
│   ├── data_loader.py         # Data loading and inspection
│   ├── preprocessing.py       # Missing values, outliers, encoding
│   ├── feature_engineering.py # Feature analysis and correlation
│   ├── visualization.py       # Plotting functions
│   └── model.py               # Model training and evaluation
├── main.py                    # Main execution script
├── requirements.txt
├── HousePrice_train.csv
└── README.md
```

## Installation

```bash
git clone https://github.com/yourusername/House-Price-Prediction.git
cd House-Price-Prediction
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Pipeline

### 1. Data Loading
- CSV import with pandas
- Initial inspection (shape, types, memory, missing values, duplicates)
- Column categorization (numerical, categorical, cardinality)

### 2. Preprocessing
- Missing value handling:
  - Domain-specific fills (e.g., "No" for absent features like pool/fence)
  - Mode imputation for categorical variables
  - Median imputation for numerical variables
- Outlier treatment using IQR method (10th-90th percentile)
- Rare category encoding (threshold: 1%)

### 3. Feature Engineering
- Correlation analysis with 0.70 threshold
- Feature selection based on target correlation
- Dropping redundant columns

### 4. Encoding
- Label encoding for binary categorical variables
- One-hot encoding for multi-class categorical variables

### 5. Model Training
- Train/test split: 80/20
- Log transformation of target variable
- StandardScaler normalization
- LightGBM regression with default parameters

### 6. Evaluation
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Feature importance visualization

## Key Functions

| Module | Function | Purpose |
|--------|----------|---------|
| `data_loader` | `check_df()` | Dataset overview |
| `data_loader` | `grab_col_names()` | Column type identification |
| `preprocessing` | `quick_missing_imp()` | Missing value imputation |
| `preprocessing` | `replace_with_thresholds()` | Outlier capping |
| `preprocessing` | `rare_encoder()` | Rare category grouping |
| `feature_engineering` | `high_correlated_cols()` | Correlation detection |
| `visualization` | `plot_importance()` | Feature importance chart |
| `model` | `train_model()` | LightGBM training |
| `model` | `evaluate_model()` | Metric calculation |

## Configuration

Edit `src/config.py` to modify:
- Random state
- Test size ratio
- Quantile thresholds for outliers
- Correlation threshold
- Rare category percentage
- Columns to drop

## Dependencies

- Python 3.7+
- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn
- lightgbm
