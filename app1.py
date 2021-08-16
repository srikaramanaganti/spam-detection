from flask import Flask,render_template,url_for,request,redirect,session

from markupsafe import escape


import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()

app = Flask(__name__)


# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


@app.route('/')
def index():
    return render_template("index.html")
@app.route('/result',methods=['GET','POST'])
def result():
    if request.method =="POST":
        msg = request.form['msg']
        if msg =='':
            return "enter message !"
        else:
            NB_spam_model = open('NB_spam_model.pkl','rb') 
            nb = joblib.load(NB_spam_model)
            cv_model = open('cv.pkl', 'rb')
            count_vector = joblib.load(cv_model)
            data = [preprocess(msg)]
            vect = count_vector.transform(data)
            mypredict = int(nb.predict(vect))
            answ = ["Non-Spam","Spam"]
            return answ[mypredict]
        # return render_template('index.html')

@app.errorhandler(404)
def invalid_route(e):
    return "error page"
def preprocess(msg):
    txt = re.sub('[^A-Za-z]'," ",msg)
    txt = txt.lower()
    txt = txt.split()
    
    txt = [ps.stem(word) for word in txt if word not in stopwords.words("english")]
    mesg = ' '.join(txt)
    return mesg
if __name__== "__main__":
    app.run(debug=True)