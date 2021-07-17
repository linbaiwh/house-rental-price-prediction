import pandas as pd
from flask import render_template, flash, redirect, url_for

from flaskapp import app
from flaskapp.forms import features_form
from flaskapp.models import rf_model

@app.route("/", methods=['GET', 'POST'])
def pred():
    form = features_form()
    if form.validate_on_submit():
        test_data = {}
        for field_name, value in form.data.items():
            if field_name not in ('submit', 'csrf_token'):
                test_data[field_name] = value
        X = pd.DataFrame(test_data, index=[0])
        prediction = rf_model.predict(X)
        flash(f'Predicted Price is {prediction}', 'success')
        return redirect(url_for('analysis'))
    return render_template('prediction.html', title='Prediction', form=form)

@app.route('/analysis')
def analysis():
    return render_template('analysis.html', title='Analysis')
