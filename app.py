#importing required libraries

from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


app = Flask(__name__)

#reading csv file

df = pd.read_csv("C:/Users/HP/Downloads/mail_data.csv")

#print(df)

data = df.where((pd.notnull(df)), ' ')

#classifying spam and ham messages

data.loc[data['Category'] == 'spam', 'Category',] = 0
data.loc[data['Category'] == 'ham', 'Category',] = 1

X = data['Message']
Y = data['Category']

#training and testing the data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 3)

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True, ngram_range=(1, 2))

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

#training model
model = LogisticRegression(C=1.0)
model.fit(X_train_features, Y_train)

#prediction of data
prediction = model.predict(X_train_features)
accuracy = accuracy_score(Y_train, prediction)

input_your_mail = ["This is a friendly reminder about our meeting scheduled for tomorrow at 10:00 AM in the conference room. Please make sure to prepare your presentation and bring any necessary materials."]
input_data_features = feature_extraction.transform(input_your_mail)
prediction = model.predict(input_data_features)
if(prediction[0] == 1):
    print('Ham Mail')
else:
    print('Spam Mail')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/second')
def second():
    return render_template('spamDetector.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    input_data_features = feature_extraction.transform([message])
    prediction = model.predict(input_data_features)
    if prediction[0] == 1:
        result = "Ham mail"
    else:
        result = "Spam mail"
    return render_template('spamDetector.html', message=message, result=result)

if __name__ == '__main__':
    app.run(debug=True)
