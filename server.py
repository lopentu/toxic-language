from flask import request, Flask
from flask import jsonify
from flask_cors import CORS
from textgen import NNTextGen

app = Flask(__name__)
CORS(app)
model = NNTextGen()

@app.route("/textgen", methods=["GET"])
def on_get():
    resp = jsonify({"message":"endpoint OK. use POST method instead"}) 
    resp.status_code = 200
    return resp

@app.route("/textgen", methods=["POST"])
def on_post():
    try:
        data = request.get_json(force=True)
        intext = data["intext"]
    except (KeyError, TypeError) as ex:
        resp = jsonify({"error": "expect intext field"})
        resp.status_code = 400
        return resp
    except Exception as ex:
        print(ex)
        resp = jsonify({"error": "not a valid json"})
        resp.status_code = 400
        return resp

    comments = model.predict(intext)
    resp = jsonify({"comments": comments})
    resp.status_code = 200
    return resp

app.config["JSON_AS_ASCII"] = False
