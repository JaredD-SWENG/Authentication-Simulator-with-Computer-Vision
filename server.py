# import Flask
import base64
from flask import Flask, send_from_directory, request, json
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from io import StringIO 
from PIL import Image
import cv2
import io
import numpy as np
import imutils
import os
from pathlib import Path
import time

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins for WebSocket connections


# import ml.py
import ml

# Send index.html
@app.route('/', methods=["GET"])
@app.route('/index.html', methods=["GET"])
def get_index():
    #return contents of index.html
    return send_from_directory('', 'index.html', mimetype='text/html')

# Send capture.html
@app.route('/capture.html', methods=["GET"])
def get_capture():
    #return contents of index.html
    return send_from_directory('', 'capture.html', mimetype='text/html')

# Send main.js
@app.route('/main.js', methods=["GET"])
def get_main():
     #return contents of main.js
    return send_from_directory('', 'main.js', mimetype='text/javascript')

# Send capture.js
@app.route('/capture.js', methods=["GET"])
def get_capture_js():
     #return contents of main.js
    return send_from_directory('', 'capture.js', mimetype='text/javascript')

# Send styles.css
@app.route('/styles.css', methods=["GET"])
def get_styles():
     #return contents of main.js
    return send_from_directory('', 'styles.css', mimetype='text/css')

@socketio.on('image')
def image(data_image):
    sbuf = StringIO()
    sbuf.write(data_image)

    # decode and convert into image
    b = io.BytesIO(base64.b64decode(data_image))
    pimg = Image.open(b)

    ## converting RGB to BGR, as opencv standards
    frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

    # Process the image frame
    frame = imutils.resize(frame, width=700)
    frame = cv2.flip(frame, 1)

    # Call is_authorized function
    authorized, boxes = ml.is_authorized(frame)
    
    color = (255, 255, 0)

    if authorized:
        print("Authorized person detected.")
        color = (0, 200, 0)
    else:
        print("Unauthorized person detected.")
        color = (0, 0, 255)

    # Draw bounding boxes on the image
    for box in boxes:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

        imgencode = cv2.imencode('.jpg', frame)[1]

        # base64 encode
        stringData = base64.b64encode(imgencode).decode('utf-8')
        b64_src = 'data:image/jpg;base64,'
        stringData = b64_src + stringData

        # emit the frame back
        emit('response_back', {'image': stringData, 'authorized': authorized})

    except AttributeError as e:
        # Handle the AttributeError
        print(f"AttributeError: {e}")
        emit('response_back', {'error': 'Face not detected in image'})

@socketio.on('new-auth-image')
def new_auth_image(data):
    data_image = data['image']

    name = data['name']
    print(f"name: {name}")

    sbuf = StringIO()
    sbuf.write(data_image)

    # decode and convert into image
    b = io.BytesIO(base64.b64decode(data_image))
    pimg = Image.open(b)

    ## converting RGB to BGR, as opencv standards
    frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

    # Process the image frame
    frame = imutils.resize(frame, width=700)
    frame = cv2.flip(frame, 1)

    # write to file
    path = Path("face_rec") / "pics"
    print(f"path {path}")
    directories = os.listdir(path)
    print(f"directories {directories}")

    if (name not in directories):
        os.mkdir(path / name)

    face_dir = path / name

    filename = name + '-' + time.strftime("%Y%m%d-%H%M%S") + ".jpg"
    file = face_dir / filename
    print(f"file {file}")

    cv2.imwrite(str(file), frame)

    # Check to make sure new face works
    authorized, boxes = ml.is_authorized(frame)
    
    color = (255, 255, 0)

    if authorized:
        print("Authorized person detected.")
        color = (0, 200, 0)
    else:
        print("Unauthorized person detected.")
        color = (0, 0, 255)

    # Draw bounding boxes on the image
    for box in boxes:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

    imgencode = cv2.imencode('.jpg', frame)[1]
    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_new_auth', {'image': stringData, 'name': name, 'authorized': authorized})

# Run the server
if __name__ == '__main__':
    socketio.run(app, port=8000)