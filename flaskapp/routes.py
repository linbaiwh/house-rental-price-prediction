import pandas as pd
from flask import render_template, flash

from flaskapp import app
from flaskapp.forms import features_form
from flaskapp.models import model

@app.route("/", methods=['GET', 'POST'])
def pred():
    form = features_form()
    if form.validate_on_submit():
        test_data = {}
        for field_name, value in form.data.items():
            if field_name not in ('submit', 'csrf_token'):
                test_data[field_name] = value
        X = pd.DataFrame(test_data, index=[0])
        prediction = model.predict(X)
        flash(f'Predicted Price is {prediction}', 'success')
    return render_template('prediction.html', title='Prediction', form=form)