# %%
import matplotlib.pyplot as plt
import seaborn as sns
from apps_context import read_listing, save_to_feature, save_to_processed
import rental_price.features.num_features as num_feat
import rental_price.features.nlp_features as nlp_feat

# %%
df = read_listing()

#%%
# target
target_cols = [
    'price', 'availability_30', 'revenue_30'
]
target_is_cat = [
    False, False, False
]
#%%
target_dist = num_feat.plot_dist(df, target_cols, target_is_cat)
# Result analysis
# price is long-tailed, can drop outliers, drop negative values
# %%
num_feat.print_outlier(df, target_cols)
# price upper bound is 357.5
#%%
# %%
df_high_price = df.loc[df['price'] > 357.5]
df_norm_price = df.loc[df['price'] <= 357.5]
# %%
sns.pairplot(data=df_high_price, vars=target_cols)
sns.pairplot(data=df_norm_price, vars=target_cols)
# %%
df = df.loc[df['price'] <= 357.5]
# %%
target_dist = num_feat.plot_dist(df, target_cols, target_is_cat)
# %%
num_feat.print_outlier(df, target_cols)
# %%
# easy of booking

easy_of_booking_raw_features = [
    'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'instant_bookable',
    'require_guest_profile_picture', 'require_guest_phone_verification'
]
easy_of_booking_is_cat = [
    True, False, False, True, True, True
]
easy_of_booking_dist = num_feat.plot_dist(df, easy_of_booking_raw_features, easy_of_booking_is_cat)

# Results analysis
# 49.3% of host_response_time is missing, non-missing samples shows some variantion
    # impute missing with constant 'Not Applicable'
    # should be highly correlated with instant_bookable
    # need further investigation, treat missing as a category
# 49.3% of host_response_rate is missing, same as host_response_time,
    # most of non-missing values are 1, can be dropped
# 22.0% of host_acceptance_rate is missing
    # most of non-missing values are 1
    # 0 acceptance_rate may be irregular, need further investigation
    # no irregularity found
    # this variable should be highly correlated with instant_bookable
# about 60% listings are not instant_bookable
# over 90% listings do not require guest profile picture or phone verification
    # the two variables can be dropped

# %%
# investigate distribution of target variables for missing features
response_na, response_notna = num_feat.separate_missing(df, 'host_response_time')
num_feat.plot_dist(response_na, target_cols, target_is_cat)
num_feat.plot_dist(response_notna, target_cols, target_is_cat)
# %%
acceptance_na, acceptance_notna = num_feat.separate_missing(df, 'host_acceptance_rate')
num_feat.plot_dist(acceptance_na, target_cols, target_is_cat)
num_feat.plot_dist(acceptance_notna, target_cols, target_is_cat)

# %%
acceptance_0 = df.loc[df['host_acceptance_rate']==0]
num_feat.plot_dist(acceptance_0, target_cols, target_is_cat)
# %%
features_keep = ['host_response_time', 'instant_bookable']

# %%
host_credibility_raw_features = [
    'host_is_superhost', 'host_listings_count',  
    'host_has_profile_pic', 'host_identity_verified'
]
host_credibility_is_cat = [
    True, False, True, True
]

num_feat.print_outlier(df, host_credibility_raw_features)
# Result analysis
# 22% hosts are superhost
    # impute missing variables as 0
# separate hosts into large hosts and small host
    # impute missing variables as 0
    # if a host has more than 3 listings, he/she is large host
# over 99% hosts have profile pictures
    # drop this variable
# about 60% hosts' identity verified
    # impute missing variables as 0

#%%
host_credibility_dist = num_feat.plot_dist(df, host_credibility_raw_features, host_credibility_is_cat)

# %%
# %%
small_host = df.loc[df['host_listings_count'] <= 3]
num_feat.plot_dist(small_host, target_cols, target_is_cat)
# %%
large_host = df.loc[df['host_listings_count'] > 3]
num_feat.plot_dist(large_host, target_cols, target_is_cat)

#%%
features_keep += [
    'host_is_superhost', 'host_listings_count',  
    'host_identity_verified']

# %%
house_hardware_raw_features = [
    'property_type', 'room_type', 'accommodates', 
    'bathrooms', 'bedrooms', 'beds', 'bed_type'
]

house_hardware_is_cat = [
    True, True, False, 
    False, False, False, True
]

# %%
num_feat.plot_dist(df, house_hardware_raw_features, house_hardware_is_cat)
# Result analysis
# almost 90% properties are apartment
    # can drop this variable
# over 95% rooms are entire home or private room
# few rooms can hold more than 10 guests
    # treat accommodates as categorical variables
    # small_room : accommodates <= 2
    # medium_room : 2 < accommodates <= 6
    # large_room : accommodates > 6
#  almost 90% rooms have 1 bathrooms
    # impute missing variables as 0
    # this variable should be highly correlated with accommodates
    # calculate bathroom_per_person
    # treat bathroom_per_person as categorical variables
        # bathroom_tight : bathroom_per_person < 0.5
        # bathroom_regular: bathroom_per_person == 0.5
        # bathroom_ample: bathroom_per_person > 0.5
# almost 90% rooms have 1 bedroom or 0 bedroom
    # impute missing variables as 0
    # this variable should be highly correlated to accommodates and beds
    # calculate bedroom_per_person
    # treat bedroom_per_person as categorical variables
        # bedroom_non : bedroom_per_person == 0
        # bedroom_tight : 0 < bedroom_per_person < 0.5
        # bedroom_regular: bedroom_per_person == 0.5
        # bedroom_ample: bedroom_per_person > 0.5
# almost 80% rooms have 1 beds
    # impute missing variables as 0
    # this variable should be highly correlated to accommodates and bedrooms
    # calculate bed_per_person
    # treat bed_per_person as categorical variable
        # bed_tight: bed_per_person < 0.5
        # bed_regular: bed_per_person == 0.5
        # bed_ample: bed_per_person > 0.5
# nearly all beds are real beds
    # drop this variable

# %%
features_keep += [
    'room_type', 'accommodates', 
    'bathrooms', 'bedrooms', 'beds'
]


# %%
booking_flexibility_raw_features = [
    'minimum_nights', 'maximum_nights', 'guests_included', 
    'cleaning_fee', 'extra_people', 'cancellation_policy'
]
booking_flexibility_is_cat = [
    True, True, True,
    False, False, True
]

# %%
num_feat.plot_dist(df, booking_flexibility_raw_features, booking_flexibility_is_cat)
# Results analysis
# less than 80% listings require minimum nights < 7 days
    # treat the variable as categorical variable
        # minimum_nights_no_req : minimum_nights == 1
        # minimum_nights_short_term : 1 < minimum_nights <= 7
        # minimum_nights_long_term: minimum_nights > 7
# almost all listings does not restrict maximum nights
    # drop this variable
# less than 80% listings has fewer guests_included than accommodates
    # treat the variable as categorical variable
        # whether guests_included < accommodates
# 14% cleaning fees are missing
    # impute missing variable with 0
# around 60% listings do not charge extra people
# around half listings has moderate/flexible cancellation policy

#%%
features_keep += [
    'minimum_nights', 'guests_included', 
    'cleaning_fee', 'extra_people', 'cancellation_policy'
]
# %%
reviews_raw_features = [
    'number_of_reviews', 'number_of_reviews_ltm', 
    'review_scores_rating', 'review_scores_accuracy',
    'review_scores_cleanliness', 'review_scores_checkin', 
    'review_scores_communication',
    'review_scores_location', 'review_scores_value'
]

reviews_is_cat = [
    False, False,
    False, False,
    False, False,
    False,
    False, False
]

#%%
num_feat.print_outlier(df, reviews_raw_features)
# %%
num_feat.plot_dist(df, reviews_raw_features, reviews_is_cat)
# Result analysis
# number_of_reviews directly related to the existing period of a listing
# number_of_reviews_ltm reflect the reviews of the recent year
# when number_of_reviews == 0, review_scores_rating is missing
    # treat the variable as categorical
        # Fail, Acceptable, Good, Excellent, Outstanding, Perfect, Not Applicable
        # [0, 80), [80, 90), [90, 95), [95, 97), [97, 99), 100, missing 
# treat review_scores_accuracy as categorical
    # Fail, Good, Perfect, Not Applicable
    # < 9, == 9, > 9, missing
# treat review_scores_cleanliness as categorical
    # Fail, Acceptable, Good, Perfect, Not Applicable
    # < 8, == 8, == 9, == 10, missing
# treat review_scores_checkin as categorical
    # Fail, Good, Perfect, Not Applicable
    # < 9, == 9, > 9, missing
# treat review_scores_communication as categorical
    # Fail, Good, Perfect, Not Applicable
    # < 9, == 9, > 9, missing
# treat review_scores_location as categorical
    # Fail, Good, Perfect, Not Applicable
    # < 9, == 9, > 9, missing
# treat review_scores_value as categorical
    # Fail, Good, Perfect, Not Applicable
    # < 9, == 9, > 9, missing

#%%
# neighborhood availability (supply)
neighborhood_listings_count = df['neighbourhood_cleansed'].value_counts()

#%%
# amenities
amenities = nlp_feat.amenities_sum(df)
amenities.mean(axis=0)
# Results analysis
# 90% houses have kitchen
# 90% houses have alarms
# 73% houses have hair dryer
# 67% houses have iron
# 46% houses have washer & dryer
# 40% houses have coffee maker
# 38% houses have paid or free parking
# 32% houses have fire extinguisher
# 28% houses have first aid kit
# 4% houses have crib
# 4% houses have fireplaces
# %%
