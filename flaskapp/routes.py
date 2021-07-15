from flask import render_template

from flaskapp import app
from flaskapp.forms import features_form

@app.route("/", methods=['GET', 'POST'])
def pred():
    form = features_form()
    return render_template('prediction.html', title='Prediction', form=form)