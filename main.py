import pandas as pd
from src.config import *
from src.data_loader import load_data, check_df, grab_col_names, convert_num_but_cat_to_object
from src.preprocessing import (missing_values_table, quick_missing_imp, check_outlier, 
                               replace_with_thresholds, rare_analyser, rare_encoder,
                               label_encoder, one_hot_encoder)
from src.feature_engineering import high_correlated_cols, target_summary_with_cat
from src.visualization import (num_summary, plot_correlation_heatmap, 
                                plot_target_distribution, plot_importance)
from src.model import prepare_data, scale_features, train_model, evaluate_model

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def main():
    # Load data
    df = load_data("HousePrice_train.csv")
    
    # Initial overview
    check_df(df)
    df.describe().T
    
    # Identify column types
    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
    df = convert_num_but_cat_to_object(df, num_but_cat)
    
    # Numerical summaries
    for col in num_cols:
        num_summary(df, col, plot=True)
    
    # Target analysis by categories
    for col in cat_cols:
        target_summary_with_cat(df, "SalePrice", col)
    
    # Target distribution
    plot_target_distribution(df, "SalePrice")
    
    # Correlation analysis
    plot_correlation_heatmap(df, num_cols)
    drop_list = high_correlated_cols(df, plot=False)
    print(drop_list)
    
    corr = df[num_cols].corr()
    saleprice_corr = corr['SalePrice'].abs().sort_values(ascending=False)
    print(saleprice_corr)
    
    # Missing values
    missing_values_table(df)
    
    for col in NO_FILL_COLS:
        df[col].fillna("No", inplace=True)
    
    df = quick_missing_imp(df, num_method="median", cat_length=CAT_LENGTH_THRESHOLD)
    
    # Outliers
    for col in num_cols:
        if col != "SalePrice":
            print(col, check_outlier(df, col))
    
    for col in num_cols:
        if col != "SalePrice":
            replace_with_thresholds(df, col)
    
    # Rare categories
    rare_analyser(df, "SalePrice", cat_cols)
    rare_encoder(df, RARE_PERC)
    
    # Encoding
    df.drop(DROP_COLS, axis=1, inplace=True)
    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
    
    binary_cols = [col for col in df.columns if df[col].dtypes == "O" and len(df[col].unique()) == 2]
    for col in binary_cols:
        label_encoder(df, col)
    
    df = one_hot_encoder(df, cat_cols, drop_first=True)
    
    # Model training
    X_train, X_test, y_train, y_test, X = prepare_data(df, "SalePrice", ["Id", "SalePrice"], 
                                                        test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_train, X_test, scaler = scale_features(X_train, X_test)
    
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluation
    metrics = evaluate_model(y_test, y_pred)
    plot_importance(model, X, num=20)


if __name__ == "__main__":
    main()
