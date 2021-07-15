from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd



class zero_imputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_name = None
    
    def fit(self, X, y=None):
        self.feature_name = X.name
        return self

    def transform(self, X, y=None):
        return pd.DataFrame({self.feature_name: X.fillna(0)})

    def get_feature_names(self):
        return self.feature_name


class host_response_time(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.pipe = make_pipeline(
            SimpleImputer(strategy='constant', fill_value='NA'), 
            OneHotEncoder()
            )

    def fit(self, X, y=None):
        self.pipe.fit(X, y)
        return self

    def transform(self, X, y=None):
        return self.pipe.transform(X)

    def get_feature_names(self):
        return self.pipe.named_steps.onehotencoder.get_feature_names(input_features=['host_response_time'])


class host_is_large(BaseEstimator, TransformerMixin):
    def __init__(self, cutoff=3):
        self.cutoff = cutoff

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.fillna(0)
        return pd.DataFrame({'large_host': X.map(lambda x: 1 if x > self.cutoff else 0)})

    def get_feature_names(self):
        return 'large_host'

class house_size(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.small = 2
        self.large = 6

    def house_size_map(self, x):
        if x <= self.small:
            return 0
        elif x > self.large:
            return 2
        else:
            return 1

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.fillna(1)
        return pd.DataFrame({'house_size': X.map(self.house_size_map)})

    def get_feature_names(self):
        return 'house_size'

class bathroom_sufficient(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cutoff = 0.5

    def bathroom_map(self, x):
        if x < self.cutoff:
            return 0
        elif x > self.cutoff:
            return 2
        else:
            return 1


    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        bathrooms = X['bathrooms'].fillna(0)
        accommodates = X['accommodates'].fillna(1)
        bathroom_per_person = bathrooms / accommodates
        return pd.DataFrame({'bathroom_sufficient': bathroom_per_person.map(self.bathroom_map)})

    def get_feature_names(self):
        return 'bathroom_sufficient'

class bedroom_sufficient(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cutoff = 0.5

    def bedroom_map(self, x):
        if x == 0:
            return 0
        elif x < self.cutoff:
            return 1
        elif x > self.cutoff:
            return 3
        else:
            return 2


    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        bedrooms = X['bedrooms'].fillna(0)
        accommodates = X['accommodates'].fillna(1)
        bedroom_per_person = bedrooms / accommodates
        return pd.DataFrame({'bedroom_sufficient': bedroom_per_person.map(self.bedroom_map)})

    def get_feature_names(self):
        return 'bedroom_sufficient'

class bed_sufficient(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cutoff = 0.5

    def bed_map(self, x):
        if x < self.cutoff:
            return 0
        elif x > self.cutoff:
            return 2
        else:
            return 1


    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        beds = X['beds'].fillna(0)
        accommodates = X['accommodates'].fillna(1)
        bed_per_person = beds / accommodates
        return pd.DataFrame({'bed_sufficient': bed_per_person.map(self.bed_map)}) 

    def get_feature_names(self):
        return 'bed_sufficient'

class rental_period(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cutoff = 7

    def period_map(self, x):
        if x == 1:
            return 0
        elif x > self.cutoff:
            return 2
        else:
            return 1


    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.fillna(1)
        return pd.DataFrame({'rental_period': X.map(self.period_map)})

    def get_feature_names(self):
        return 'rental_period'

class all_guests_included(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        all_include = X['guests_included'] >= X['accommodates']
        return pd.DataFrame({'all_guests_included': all_include.map(lambda a: 1 if True else 0)})

    def get_feature_names(self):
        return 'all_guests_included'

class review_scores_rating(BaseEstimator, TransformerMixin):
    def socre_map(self, x):
        if 0 < x < 80:
            return 0
        elif 80 <= x < 90:
            return 1
        elif 90 <= x < 95:
            return 2
        elif 95 <= x < 97:
            return 3
        elif 97 <= x < 100:
            return 4
        else:
            return 5

    def fit(self, X, y=None):
        self.mean = X.mean()
        return self

    def transform(self, X ,y=None):
        X = X.fillna(self.mean)
        return pd.DataFrame({'review_score_rating': X.map(self.socre_map)})
    
    def get_feature_names(self):
        return 'review_score_rating'


class review_scores_accuracy(BaseEstimator, TransformerMixin):
    def socre_map(self, x):
        if 0 < x < 9:
            return 0
        elif x > 9:
            return 2
        else:
            return 1

    def fit(self, X, y=None):
        self.mean = X.mean()
        return self

    def transform(self, X ,y=None):
        X = X.fillna(self.mean)
        return pd.DataFrame({'review_score_accuracy': X.map(self.socre_map)})
    
    def get_feature_names(self):
        return 'review_score_accuracy'


class review_scores_cleanliness(BaseEstimator, TransformerMixin):
    def socre_map(self, x):
        if 0 < x < 8:
            return 0
        elif x == 8:
            return 1
        elif x == 9:
            return 2
        else:
            return 3

    def fit(self, X, y=None):
        self.mean = X.mean()
        return self

    def transform(self, X ,y=None):
        X = X.fillna(self.mean)
        return pd.DataFrame({'review_score_cleanliness': X.map(self.socre_map)})
    
    def get_feature_names(self):
        return 'review_scores_cleanliness'


class review_scores_checkin(BaseEstimator, TransformerMixin):
    def socre_map(self, x):
        if 0 < x < 9:
            return 0
        elif x > 9:
            return 2
        else:
            return 1

    def fit(self, X, y=None):
        self.mean = X.mean()
        return self

    def transform(self, X ,y=None):
        X = X.fillna(self.mean)
        return pd.DataFrame({'review_score_checkin': X.map(self.socre_map)})
    
    def get_feature_names(self):
        return 'review_scores_checkin'


class review_scores_communication(BaseEstimator, TransformerMixin):
    def socre_map(self, x):
        if 0 < x < 9:
            return 0
        elif x > 9:
            return 2
        else:
            return 1

    def fit(self, X, y=None):
        self.mean = X.mean()
        return self

    def transform(self, X ,y=None):
        X = X.fillna(self.mean)
        return pd.DataFrame({'review_score_communication': X.map(self.socre_map)})
    
    def get_feature_names(self):
        return 'review_scores_communication'


class review_scores_location(BaseEstimator, TransformerMixin):
    def socre_map(self, x):
        if 0 < x < 9:
            return 0
        elif x > 9:
            return 2
        else:
            return 1

    def fit(self, X, y=None):
        self.mean = X.mean()
        return self

    def transform(self, X ,y=None):
        X = X.fillna(self.mean)
        return pd.DataFrame({'review_score_location': X.map(self.socre_map)})
    
    def get_feature_names(self):
        return 'review_scores_location'


class review_scores_value(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mean = None

    def socre_map(self, x):
        if 0 < x < 9:
            return 0
        elif x > 9:
            return 2
        else:
            return 1

    def fit(self, X, y=None):
        self.mean = X.mean()
        return self

    def transform(self, X ,y=None):
        X = X.fillna(self.mean)
        return pd.DataFrame({'review_score_value': X.map(self.socre_map)})
    
    def get_feature_names(self):
        return 'review_scores_value'

class neighbourhood_listing_count(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.neighbourhood_listing_map = None

    def fit(self, X, y=None):
        self.neighbourhood_listing_map = X.value_counts().to_dict()
        return self

    def transform(self, X, y=None):
        return pd.DataFrame({'neighbourhood_listing_count': X.map(self.neighbourhood_listing_map)})

    def get_feature_names(self):
        return 'neighbourhood_listing_count'


amenities_vocab = [
    'kitchen', 'parking', 'alarm', 'hair dryer', 'iron',
    'coffee maker', 'fire extinguisher', 'crib', 'first aid', 'washer',
    'fireplace'
    ]
amenities = CountVectorizer(ngram_range=(1, 2), vocabulary=amenities_vocab, binary=True)
