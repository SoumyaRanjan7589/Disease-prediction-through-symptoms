from flask import Flask,redirect,url_for,render_template,request
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

app=Flask(__name__)


@app.route('/')
def first():
    return render_template('index1.html')



@app.route('/submit', methods=['POST']) 
def registration():
    name=request.form.get('hello')
    # pickle file for vectorize
    pickled_model = pickle.load(open('prediction_model_tfi.pkl',"rb"))
    features = [name]
    #pickle file for prediction.
    pickled_model1 = pickle.load(open('prediction_model_pac.pkl',"rb"))
    
    text=pickled_model.transform(features)

    pred1=pickled_model1.predict(text)[0]
    pred1="You are affected by the disease "+pred1
    
   
    
    
        
    return render_template('index1.html',result=pred1)

        
        

if __name__=='__main__':
    app.run(debug=True)