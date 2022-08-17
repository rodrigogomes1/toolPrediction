from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)


@app.route("/")
def hello():
    return "<h1>Welcome to our Tool Recognition Machine Learning-API! By Ulisses!</h1>"

@app.route('/predict', methods=['POST'])
def predict():
    ml_algorithm = joblib.load("modelTrainedHome2.pkl")
    json_ = request.json
    input=[]
    for key, value in json_[0].items():
        input.append(value)
    inputEntry=[input]
    print(inputEntry)
    prediction = ml_algorithm.predict(inputEntry)
    #jsonify({'prediction': list(prediction)})
    #"<h1>Prediction"+str(prediction)+"!</h1>",jsonify({'prediction': list(prediction)})
    return str(prediction[0])


if __name__ == '__main__': # If you don't provide any port then the port will be set to 12345
    ml_algorithm = joblib.load("model2.pkl") # Load "model.pkl"
    print ('Model loaded')
    app.run(debug=True)
