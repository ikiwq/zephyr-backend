from zephyr import *
zephyr = Zephyr()

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-type"

@app.route("/zephyrbot", methods=["POST"])
@cross_origin()
def chat():
    message = request.get_json().get("message")
    tag = zephyr.chat(message)
    return jsonify(message = tag)

if __name__ == "__main__":
    app.run()


