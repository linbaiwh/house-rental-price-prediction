from flask import Flask

app = Flask(__name__)
app.config['SECRET_KEY'] = '3952bcf7f9f9858971fa1a70472fabe4'


from flaskapp import routes