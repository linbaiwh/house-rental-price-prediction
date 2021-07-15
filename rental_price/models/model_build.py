from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import rental_price.models.trfs as trfs

features_trf = ColumnTransformer([
    ('latitude', 'passthrough', ['latitude']),
    ('longitude', 'passthrough', ['longitude']),
    ('host_response_time', trfs.host_response_time(), ['host_response_time']),
    ('instant_bookable', trfs.zero_imputer(), 'instant_bookable'),
    ('host_is_superhost', trfs.zero_imputer(), 'host_is_superhost'),
    ('large_host', trfs.host_is_large(), 'host_listings_count'),
    ('host_identity_verified', trfs.zero_imputer(), 'host_identity_verified'),
    ('room_type', trfs.OneHotEncoder(handle_unknown='ignore'), ['room_type']),
    ('house_size', trfs.house_size(), 'accommodates'),
    ('bathroom_sufficient', trfs.bathroom_sufficient(), ['accommodates', 'bathrooms']),
    ('bedroom_sufficient', trfs.bedroom_sufficient(), ['accommodates', 'bedrooms']),
    ('bed_sufficient', trfs.bed_sufficient(), ['accommodates', 'beds']),
    ('rental_period', trfs.rental_period(), 'minimum_nights'),
    ('all_guests_included', trfs.all_guests_included(), ['accommodates', 'guests_included']),
    ('cleaning_fee', trfs.zero_imputer(), 'cleaning_fee'),
    ('extra_people', trfs.zero_imputer(), 'extra_people'),
    ('cancellation_policy', trfs.OneHotEncoder(handle_unknown='ignore'), ['cancellation_policy']),
    ('number_of_reviews', trfs.zero_imputer(), 'number_of_reviews'),
    ('number_of_reviews_ltm', trfs.zero_imputer(), 'number_of_reviews_ltm'),
    ('review_scores_rating', trfs.review_scores_rating(), 'review_scores_rating'),
    ('review_scores_accuracy', trfs.review_scores_accuracy(), 'review_scores_accuracy'),
    ('review_scores_cleanliness', trfs.review_scores_cleanliness(), 'review_scores_cleanliness'),
    ('review_scores_checkin', trfs.review_scores_checkin(), 'review_scores_checkin'),
    ('review_scores_communication', trfs.review_scores_communication(), 'review_scores_communication'),
    ('review_scores_location', trfs.review_scores_location(), 'review_scores_location'),
    ('review_scores_value', trfs.review_scores_value(), 'review_scores_value'),
    ('neighbourhood_listing_count', trfs.neighbourhood_listing_count(), 'neighbourhood_cleansed'),
    ('amenities', trfs.amenities, 'amenities')
], n_jobs=-1, sparse_threshold=0
)

features_to_keep = [
    'latitude', 'longitude', 'neighbourhood_cleansed', 
    'host_response_time', 'instant_bookable', 'host_is_superhost','host_listings_count','host_identity_verified',
    'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'amenities', 
    'minimum_nights', 'guests_included', 'cleaning_fee', 'extra_people', 'cancellation_policy',
    'number_of_reviews', 'number_of_reviews_ltm', 'review_scores_rating', 'review_scores_accuracy',
    'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
    'review_scores_location', 'review_scores_value'
]

def get_col_feature_names(columntransformer):
    feature_names = []
    for col, trf, _ in columntransformer.transformers_:
        if isinstance(trf, str) == False:
            feature_name = trf.get_feature_names()
            if isinstance(feature_name, str):
                feature_names.append(feature_name)
                continue
            if isinstance(feature_name, np.ndarray):
                try:
                    feature_name = trf.get_feature_names(input_features=[col]).tolist()
                except:
                    feature_name = trf.get_feature_names().tolist()
            feature_names += feature_name
        elif trf == 'passthrough':
            feature_names.append(col)
    return feature_names


class model_eval():
    def __init__(self, data, target, X_cols=features_to_keep):
        self.data = data
        self.target = target
        self.X_cols = X_cols
        self.model = None

    def gen_train_test_set(self, test_size=0.2):
        X = self.data[self.X_cols]
        y = self.data[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y,test_size=test_size,random_state=2021)
        return self

    def model_fit(self, model, params=None):
        if params is not None:
            model.set_params(params)
        model.fit(self.X_train, self.y_train)
        self.model = model
        return self

    def params_prepare(self, model, params):
        param_elgible = model.get_params().keys()
        return {k:v for k, v in params.items() if k in param_elgible}

    def model_tuning(self, model, params):
        model_spec = dict((name, type(step).__name__) for name, step in model.steps[1:])

        params = self.params_prepare(model, params)

        gs = GridSearchCV(model, params, n_jobs=-1, scoring='neg_mean_squared_error', cv=3, return_train_score=True)
        gs.fit(self.X_train, self.y_train)

        self.model = gs.best_estimator_

        self.model_sum = {**model_spec, **gs.best_params_}
        return gs

    def train_test_predict(self):
        self.y_train_pred = self.model.predict(self.X_train)
        self.y_test_pred = self.model.predict(self.X_test)
        return self

    def model_scores(self):
        return {
            'train_MAE': mean_absolute_error(self.y_train, self.y_train_pred),
            'test_MAE': mean_absolute_error(self.y_test, self.y_test_pred),

            'train_MSE': mean_squared_error(self.y_train, self.y_train_pred),
            'test_MSE': mean_squared_error(self.y_test, self.y_test_pred),

            'train_R2': r2_score(self.y_train, self.y_train_pred),
            'test_R2': r2_score(self.y_test, self.y_test_pred)
        }

def plot_search_results(grid):
    """Plot training/validation scores against hyperparameters

    Args:
        grid (GridSearchCV): GridSearchCV Instance that have cv_results
    """
    cv_results = pd.DataFrame(grid.cv_results_)
    params = []
    for param in cv_results.columns:
        if 'param_' in param:
            try:
                unique_param = cv_results[param].nunique()
            except TypeError:
                unique_param = 1
            except:
                unique_param = 1
            
            if unique_param > 1:
                params.append(param[6:])

    # params = [param[6:] for param in cv_results.columns if 'param_' in param and cv_results[param].nunique() > 1]
    num_params = len(params)
    scores = [score[11:] for score in cv_results.columns if 'mean_train_' in score]
    num_scores = len(scores)

    fig = plt.figure(figsize=(5 * num_params, 5 * num_scores))
    subfigs = fig.subfigures(num_scores, 1, squeeze=False, wspace=0.05, hspace=0.05)
    
    for j in range(num_scores):
        axes = subfigs[j,0].subplots(1, num_params, squeeze=False)
        subfigs[j,0].suptitle(f'{scores[j]} per Parameters')
        subfigs[j,0].supylabel('Best Score')
        
        for i, param in enumerate(params):
            plot_cv = pd.melt(
                cv_results, id_vars=[f'param_{param}'], 
                value_vars=[f'mean_train_{scores[j]}', f'mean_test_{scores[j]}'], 
                var_name='type', value_name=scores[j]
                )            
            try:
                sns.lineplot(x=f'param_{param}', y=scores[j], data=plot_cv, hue='type', ax=axes[0,i])
            except TypeError:
                sns.violinplot(x=f'param_{param}', y=scores[j], data=plot_cv, hue='type', ax=axes[0,i], palette="Set3")
            except ValueError:
                plot_cv[f'param_{param}'] = plot_cv[f'param_{param}'].map(str)
                sns.lineplot(x=f'param_{param}', y=scores[j], data=plot_cv, hue='type', ax=axes[0,i])
            except Exception:
                print(f'cannot plot search results for {scores[j]} {param}')
            
            axes[0, i].set_xlabel(param.upper())

    plt.subplots_adjust(top=0.92)
    plt.show()
    return fig

rf_pipe = Pipeline([
    ('features_trf', features_trf),
    ('rf_reg', RandomForestRegressor(n_jobs=-1))
])