from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)


@app.route("/", methods=['GET'])
def hello():
    return "Welcome to machine learning mo2222del2 APIs-Waitress!"

@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.json
    input=[]
    for key, value in json_[0].items():
        input.append(value)
    inputEntry=[input]
    print(inputEntry)
    prediction = ml_algorithm.predict(inputEntry)
    return jsonify({'prediction': list(prediction)})


if __name__ == '__main__':
    port = 12121 # If you don't provide any port then the port will be set to 12345
    ml_algorithm = joblib.load("model2.pkl") # Load "model.pkl"
    print ('Model loaded')
    app.run(port=port, debug=True)