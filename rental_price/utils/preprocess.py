import logging
import pandas as pd

logger = logging.getLogger(__name__)


# exclude irregular listings:
    # no bookings in the next 30 days
    # no availability in the next 365 days and number of reviews less than 5

def drop_irr(df, review_cutoff=5):
    df_norm = df.loc[df['availability_30'] < 30]
    df_norm = df_norm.loc[(df['availability_365'] > 0) | (df['number_of_reviews'] >= review_cutoff)]
    return df_norm


def neighborhood_selection(df, neighborhood='Manhattan'):
    return df.loc[df['neighbourhood_group_cleansed'] == neighborhood]

# raw features to keep
raw_features = [
    # primary key of a listing
    'id', 

    # ease of booking 
    'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'instant_bookable',
    'require_guest_profile_picture', 'require_guest_phone_verification',
    
    # host credibility
    'host_is_superhost', 'host_listings_count', 'host_verifications', 
    'host_has_profile_pic', 'host_identity_verified',
    
    # house hardware
    'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type',
    
    # house location
    'latitude', 'longitude', 'neighbourhood_cleansed', 'zipcode',
    
    # house software
    'amenities', 'house_rules', 'is_business_travel_ready',
    
    # booking flexibility
    'minimum_nights', 'maximum_nights', 'guests_included', 
    'cleaning_fee', 'extra_people', 'cancellation_policy',
    
    # reviews
    'number_of_reviews', 'number_of_reviews_ltm', 'review_scores_rating', 'review_scores_accuracy',
    'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
    'review_scores_location', 'review_scores_value'
]

# raw targets
raw_targets = [
    'price', 'availability_30', 'availability_60', 'availability_90', 'availability_365'
]

def col_selection(df):
    cols = raw_features + raw_targets
    df_select = df[cols]
    return df_select

remove_dollar_sign = lambda x: float(x.strip('$').replace(',',''))

def price_trf(df):
    df = df.assign(price=df['price'].map(remove_dollar_sign, na_action='ignore'))
    df = df.assign(cleaning_fee=df['cleaning_fee'].map(remove_dollar_sign, na_action='ignore'))
    df = df.assign(extra_people=df['extra_people'].map(remove_dollar_sign, na_action='ignore'))
    df = df.dropna(subset=['price'])
    df = df.assign(revenue_30=df['price'] * (30 - df['availability_30'])) 
    return df


remove_percentage_sign = lambda x: float(x.strip('%'))/100
true_false_map = {'t': 1, 'f': 0}

def numeric_trf(df):
    df = df.assign(host_response_rate=df['host_response_rate'].map(remove_percentage_sign, na_action='ignore'))
    df = df.assign(host_acceptance_rate=df['host_acceptance_rate'].map(remove_percentage_sign, na_action='ignore'))
    
    df = df.assign(instant_bookable=df['instant_bookable'].map(true_false_map, na_action='ignore'))
    df = df.assign(require_guest_profile_picture=df['require_guest_profile_picture'].map(true_false_map, na_action='ignore'))
    df = df.assign(require_guest_phone_verification=df['require_guest_phone_verification'].map(true_false_map, na_action='ignore'))
    df = df.assign(host_is_superhost=df['host_is_superhost'].map(true_false_map, na_action='ignore'))
    df = df.assign(host_has_profile_pic=df['host_has_profile_pic'].map(true_false_map, na_action='ignore'))
    df = df.assign(host_identity_verified=df['host_identity_verified'].map(true_false_map, na_action='ignore'))
    df = df.assign(is_business_travel_ready=df['is_business_travel_ready'].map(true_false_map, na_action='ignore'))   
    
    return df
