import base64
from flask import Flask, render_template, Response
from flask import request, jsonify
import os
import cv2
import numpy as np
from flask import make_response

from tracking_main import HeadMovementTracker

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

HMT = HeadMovementTracker()

photo_base64 = ''

@app.route('/')
def photo():
    resp = make_response(render_template('flip_v4.html'))
    resp.set_cookie('sessionID', '', expires=0)
    return resp

@app.route('/_photo_cap2')
def detect2():
    global photo_base64
    add_str = request.args.get('photo_cap')
    if add_str:
        photo_base64 += add_str
    return None

@app.route('/_photo_cap')
def detect():
    global photo_base64
    add_str = request.args.get('photo_cap')
    if add_str:
        photo_base64 += add_str
    prefix, encoded = photo_base64.split(",", 1)
    binary_data = base64.b64decode(encoded)
    nparr = np.fromstring(binary_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = HMT.track_head_movement(img)

    cv2.imwrite("static/frame.jpeg", img)
    _, buffer = cv2.imencode('.jpeg', img)
    jpg_as_text = base64.b64encode(buffer)
    new_photo_base64 = prefix + ',' + jpg_as_text.decode('utf-8')
    photo_base64 = ''
    return jsonify(photo_base64=new_photo_base64)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
