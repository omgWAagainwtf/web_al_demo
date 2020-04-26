from flask import Flask, request, Response, render_template, url_for, jsonify
import requests
import json
from flask_cors import CORS, cross_origin
from bson.objectid import ObjectId
import math
from myjieba.analyse import textrank

import logging
import os
from pymongo import MongoClient

import base64

client = MongoClient(host="192.168.87.229", username="gais", password="legbone")
db = client["seg_news"]

app = Flask(__name__)
app.debug = True
category_dict = {}
acc = 0
mis = 0

def cut(sentence, pos=True):
    text = requests.get(
        "http://192.168.87.16:5010/gais_ckipNLP",
        data={"text": sentence, "deli": " ", "pos": pos},
    )
    return text.text


@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)


def dated_url_for(endpoint, **values):
    if endpoint == "static":
        filename = values.get("filename", None)
        if filename:
            file_path = os.path.join(app.root_path, endpoint, filename)
            values["q"] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)


@app.route("/", methods=["GET"])
@cross_origin()
def mainpage():
    return render_template("index.html")


# segment
@app.route("/E1", methods=["POST"])
@cross_origin()
def E1():
    # text = request.form.get("text")
    text = request.get_json()
    print(text)
    seg_str = cut(text["text"], pos=False)
    return jsonify({"text": seg_str})


# get keyword
@app.route("/E2", methods=["POST"])
@cross_origin()
def E2():
    text = request.get_json()
    pos_ws = cut(text["text"], pos=True)
    keywords_list = textrank([pos_ws], 1, topK=10)
    return jsonify({"text": keywords_list})


# submit labeled data to database
@app.route("/E3", methods=["POST"])
@cross_origin()
def E3():
    res = request.get_json()
    col = db["ettoday"]
    for data in res["data"]:
        print(data)
    categories = set()
    for cur in col.find({}, {"_id": 0, "category": 1}):
        categories.add(cur["category"])
    i = 0
    for c in categories:
        category_dict[c] = i
        i += 1
    return jsonify({"status": "201"})


# init data in db output text_obj_id text and category
@app.route("/M1", methods=["POST"])
@cross_origin()
def M1():
    col = db["ettoday"]
    cursor = col.find_one({})
    status = 200
    if cursor == None:
        status = 201
    return jsonify(
        {
            "text_id": str(cursor["_id"]),
            "text": cursor["content"],
            "class": cursor["category"],
            "status": status,
        }
    )


# init data in db to get all categories and category id
@app.route("/M2", methods=["POST"])
@cross_origin()
def M2():
    l = []
    for c in category_dict:
        l.append([category_dict[c], c])
    return jsonify({"class": l})


# get text_id and delete from col
@app.route("/M3", methods=["POST"])
@cross_origin()
def M3():
    data = request.get_json()
    text_id = data["data"]
    # change data select column to 0 by ObjectId(text_id)
    return jsonify({"status": 200})


# submit to compute machine db
@app.route("/M4", methods=["POST"])
@cross_origin()
def M4():
    data = request.get_json()
    text_id = data["text_id"]
    class_id = data["class_id"]
    print(text_id, class_id)
    # change data select column to 0 by ObjectId(text_id)
    return jsonify({"status": 200})


# init models from compute machine
@app.route("/M5", methods=["POST"])
@cross_origin()
def M5():
    model = []
    model.append([0, "textcnn"])
    model.append([1, "han"])
    model.append([2, "myclassifier"])
    model.append([3, "svm"])
    # change data select column to 0 by ObjectId(text_id)
    return jsonify({"data": model})


@app.route("/M1_1", methods=["POST"])
@cross_origin()
def M1_1():
    res = request.get_json()
    page = res["page"]
    col = db["ettoday"]
    data = []
    for cursor in col.find({}).limit(10).skip(10 * page - 1):
        data.append([cursor["content"], cursor["category"]])
    return jsonify({"data": data})

@app.route("/T1", methods=["POST"])
@cross_origin()
def T1():
    col = db["ettoday"]
    cursor = col.find_one({})
    status = 200
    if cursor == None:
        status = 201
    return jsonify({"data": [cursor["content"], cursor["category"]],"status": status})

# true false to measure accuracy

@app.route("/T2", methods=["POST"])
@cross_origin()
def T2():
    res=  request.get_json()
    boolean = res["data"]
    if boolean == "True":
        acc += 1
    else:
        mis += 1
    return jsonify({"status": 200})

@app.route("/A1", methods=["POST"])
@cross_origin()
def A1():
    with open("cat.jpeg", "rb") as img_file:
        img_str = base64.b64decode(img_file.read())
    return jsonify({"data":"data:image/jpeg;base64,{}".format(img_str)})
if __name__ == "__main__":
    app.config["JSON_AS_ASCII"] = False
    app.run("192.168.87.231", port=5002)
    CORS(app)
