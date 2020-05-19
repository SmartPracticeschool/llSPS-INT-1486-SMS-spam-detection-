# -*- coding: utf-8 -*-
"""
Created on Mon May 18 18:10:25 2020

@author: sneha
"""
from flask import render_template, Flask, request
from keras.models import load_model
import pickle 
with open(r'C:\Users\sneha\Desktop\flask1\CountVectorizer','rb') as file:
    cv =pickle.load(file)
cl=load_model(r'C:\Users\sneha\Desktop\flask1\smsmodel.h5')
cl.compile(optimizer='adam',loss='binary_crossentropy')


app2 = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/login', methods = ['POST'])
def login():
    p = request.form['SMS']
    entered_input = p
    x_intent=cv.transform([entered_input])
    y_pred=cl.predict(x_intent)
    
    if(y_pred>0.5):
        return render_template("index.html",showcase="spam")
    else:
        return render_template("index.html",showcase="not spam")


if __name__ == '__main__':
    app2.run(debug = False)