import pandas as pd
import numpy as np
from apps_context import read_listing, save_to_feature, save_model
from rental_price.models.model_build import train_test_split, features_trf, get_col_feature_names, model_eval, rf_pipe, plot_search_results

def save_train_test(target, features_trf=features_trf):
    df = read_listing()
    feature_cols = df.columns.tolist()
    feature_cols.remove(target)
    X = df[feature_cols]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=2021)
    
    features_trf.fit(X_train, y_train)
    feature_names = get_col_feature_names(features_trf)

    X_train_trfed = pd.DataFrame(features_trf.transform(X_train), columns=feature_names)
    X_test_trfed = pd.DataFrame(features_trf.transform(X_test), columns=feature_names)

    train_trfed = X_train_trfed.join(y_train.reset_index(drop=True))
    test_trfed = X_test_trfed.join(y_test.reset_index(drop=True))

    save_to_feature('train_trfed', train_trfed)
    save_to_feature('test_trfed', test_trfed)


def model_training(target, model, params=None):
    df = read_listing()
    train = model_eval(df, target)
    train.gen_train_test_set()
    train.model_fit(model, params=params)
    train.train_test_predict()
    scores = train.model_scores()
    print(scores)
    return train.model


def model_tuning(target, model, params=None):
    df = read_listing()
    train = model_eval(df, target)
    train.gen_train_test_set()
    gs = train.model_tuning(model, params=params)
    plot_search_results(gs)
    print(train.model_sum)
    train.train_test_predict()
    scores = train.model_scores()
    print(scores)
    return train.model

if __name__ == '__main__':
    # save_train_test('revenue_30')
    params = {
        'rf_reg__criterion': ['mse'],
        'rf_reg__max_depth': [8, 10],
        'rf_reg__max_features': [0.33, 0.5, 0.8]
    }
    # model_tuning('revenue_30', rf_pipe, params=params)
    rf_model = model_tuning('price', rf_pipe, params=params)
    save_model('rf_price', rf_model)