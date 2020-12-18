import base64
from flask import Flask, render_template, Response
from flask import request, jsonify
import os
import cv2
import numpy as np

# from tracking_main import HeadMovementTracker

app = Flask(__name__)

# HMT = HeadMovementTracker()

@app.route('/')
def photo():
    return render_template('hd.html')

# @app.route('/_photo_cap')
# def detect():
#     photo_base64 = request.args.get('photo_cap')
#     prefix, encoded = photo_base64.split(",", 1)
#     binary_data = base64.b64decode(encoded)
#     nparr = np.fromstring(binary_data, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     # img = HMT.track_head_movement(img)

#     cv2.imwrite("static/frame.jpeg", img)
#     _, buffer = cv2.imencode('.jpeg', img)
#     jpg_as_text = base64.b64encode(buffer)
#     new_photo_base64 = prefix + ',' + jpg_as_text.decode('utf-8')

#     return jsonify(photo_base64=new_photo_base64)

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=False)
