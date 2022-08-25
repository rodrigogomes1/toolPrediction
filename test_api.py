from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)


@app.route("/")
def hello():
    return "<h1>Welcome to our Tool/Location Recognition Machine Learning-API! By Ulisses!</h1>"

@app.route('/predict/tool', methods=['POST'])
def predictTool():
    ml_algorithm = joblib.load("modeltrainedHome2.pkl")
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

@app.route('/predict/location', methods=['POST'])
def predictLocation():
    ml_algorithm = joblib.load("modelLocation.pkl")
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
    #ml_algorithm = joblib.load("modelTrainedHome2.pkl") # Load "model.pkl"
    print ('Model loaded')
    app.run(debug=True)
