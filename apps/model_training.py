import pandas as pd
import numpy as np
from apps_context import read_listing, save_to_feature, save_model
from rental_price.models.model_build import train_test_split, features_trf, get_col_feature_names, model_eval, rf_pipe, plot_search_results

def save_train_test(target, model=features_trf):
    df = read_listing()
    train = model_eval(df, target)
    train.gen_train_test_set()
    train.model_fit(model)
    
    feature_names = get_col_feature_names(train.model)

    X_train_trfed = pd.DataFrame(train.model.transform(train.X_train), columns=feature_names)
    X_test_trfed = pd.DataFrame(train.model.transform(train.X_test), columns=feature_names)

    train_trfed = X_train_trfed.join(train.y_train.reset_index(drop=True))
    test_trfed = X_test_trfed.join(train.y_test.reset_index(drop=True))

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
    save_train_test('price')
    # params = {
    #     'rf_reg__criterion': ['mse'],
    #     'rf_reg__max_depth': [8, 10],
    #     'rf_reg__max_features': [0.33, 0.5, 0.8]
    # }
    # model_tuning('revenue_30', rf_pipe, params=params)
    # rf_model = model_tuning('price', rf_pipe, params=params)
    # save_model('rf_price', rf_model)