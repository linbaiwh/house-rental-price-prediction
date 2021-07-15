from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField, IntegerField, FloatField, BooleanField, TextAreaField
from wtforms.validators import DataRequired, NumberRange

neighbourhood_options = [
    'Harlem', "Hell's Kitchen", 'Upper West Side',
    'East Village', 'Upper East Side', 'Midtown', 'East Harlem',
    'Chelsea', 'Washington Heights', 'Lower East Side', 'West Village',
    'Financial District', 'Kips Bay', 'Murray Hill', 'Nolita',
    'Theater District', 'Chinatown', 'Gramercy', 'SoHo',
    'Greenwich Village', 'Inwood', 'Morningside Heights', 'Tribeca',
    'Little Italy', 'Roosevelt Island', 'Battery Park City', 'Two Bridges',
    'NoHo', 'Flatiron District', 'Stuyvesant Town', 'Civic Center', 'Marble Hill'
                        ]

response_time_options = [
    None,
    'within an hour',
    'within a few hours',
    'within a day',
    'a few days or more'
]

room_type_options = [
    'Entire home/apt',
    'Private room',
    'Hotel room',
    'Shared room'
]

cancellation_policy_options = [
    'flexible',
    'moderate',
    'strict_14_with_grace_period',
    'super_strict_30',
    'super_strict_60'
]

class features_form(FlaskForm):
    latitude = FloatField('Latitude', validators=[DataRequired()])
    longitude = FloatField('Longitude', validators=[DataRequired()])
    neighbourhood_cleansed = SelectField('Neighbourhood', choices=neighbourhood_options, validators=[DataRequired()])

    host_response_time = SelectField('Host Response Time', choices=response_time_options)
    instant_bookable = BooleanField('Instant Bookable?')

    host_is_superhost = BooleanField('Is Host Super Host?')
    host_listings_count = IntegerField('# Host Listing', validators=[DataRequired()])
    host_identity_verified = BooleanField('Is Host Identity Verified?')

    room_type = SelectField('Room Type', choices=room_type_options, validators=[DataRequired()])
    accommodates = IntegerField('Accommodates', validators=[DataRequired()])
    bathrooms = FloatField('# Bathroom')
    bedrooms = IntegerField('# Bedroom')
    beds = IntegerField('# Beds')
    amenities = TextAreaField('Amenities', validators=[DataRequired()])

    minimum_nights = IntegerField('# Minimum Nights', validators=[NumberRange(min=1)])
    guests_included = IntegerField('# Guests included', validators=[DataRequired(), NumberRange(min=1)])
    cleaning_fee = FloatField('Cleaning Fee')
    extra_people = FloatField('Charge for Extra Person')
    cancellation_policy = SelectField('Cancellation Policy', choices=cancellation_policy_options, validators=[DataRequired()])

    number_of_reviews = IntegerField('# Total Reviews', validators=[DataRequired()])
    number_of_reviews_ltm = IntegerField('# Last Year Reviews', validators=[DataRequired()])
    review_scores_rating = IntegerField('Overall Score', validators=[NumberRange(min=10, max=100)])
    review_scores_accuracy = IntegerField('Accuracy Score', validators=[NumberRange(min=1, max=10)])
    review_scores_cleanliness = IntegerField('Cleanliness Score', validators=[NumberRange(min=1, max=10)])
    review_scores_checkin = IntegerField('Check In Score', validators=[NumberRange(min=1, max=10)])
    review_scores_communication = IntegerField('Communication Score', validators=[NumberRange(min=1, max=10)])
    review_scores_location = IntegerField('Location Score', validators=[NumberRange(min=1, max=10)])
    review_scores_value = IntegerField('Value Score', validators=[NumberRange(min=1, max=10)])

    submit = SubmitField('Submit')


