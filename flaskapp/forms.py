from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField, IntegerField, FloatField
from wtforms.validators import DataRequired


response_time_options = [
    ('within an hour', 'within an hour'),
    ('within a few hours', 'within a few hours'),
    ('within a day', 'within a day'),
    ('a few days or more', 'a few days or more')
]

cancellation_policy_options = [
    ('flexible', 'flexible'),
    ('moderate', 'moderate'),
    ('strict_14_with_grace_period', 'strict_14_with_grace_period'),
    ('super_strict_30', 'super_strict_30'),
    ('super_strict_60', 'super_strict_60')
]

class features_form(FlaskForm):
    latitude = FloatField('Latitude', validators=[DataRequired()])
    longtitude = FloatField('Longtitude', validators=[DataRequired()])

    # host_response_time = SelectField('Host Response Time', choices=[response_time_options])

    submit = SubmitField('Submit')


