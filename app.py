from flask import Flask,render_template,request
import pickle

# load the model from disk
filename = 'model/clickbait_model.pkl'
naive_bayes_model = pickle.load(open(filename, 'rb'))
cv_naive=pickle.load(open('model/clickbait_transform.pkl','rb'))

app = Flask(__name__)

@app.route('/',methods=['POST','GET'])
def home():
    if request.method == 'POST':
        message = request.form['message'].lower()
        data = [message]
        vect_naive = cv_naive.transform(data).toarray()
        # for naive bayes classifier
        my_nb_prediction = naive_bayes_model.predict(vect_naive)[0]
        print(my_nb_prediction)
        return render_template('index.html',my_nb_prediction = my_nb_prediction)
    return render_template('index.html')
    

if __name__ == '__main__':
    app.run(debug=True)