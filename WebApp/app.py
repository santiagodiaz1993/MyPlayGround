from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def home():
        return render_template("home.html")




@app.route('/regression')
def regression():
        return render_template("regression.html", stock_key="WIKI.GOOGL")

if __name__ == "__main__":
    app.run(debug=True)
