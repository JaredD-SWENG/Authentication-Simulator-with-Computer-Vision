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

# Send main.js
@app.route('/main.js', methods=["GET"])
def get_main():
     #return contents of main.js
    return send_from_directory('', 'main.js', mimetype='text/javascript')

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
    
    if authorized:
        print("Authorized person detected.")
    else:
        print("Unauthorized person detected.")

    # Draw bounding boxes on the image
    for box in boxes:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 2)

    imgencode = cv2.imencode('.jpg', frame)[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', {'image': stringData, 'authorized': authorized})



# Run the server
if __name__ == '__main__':
     
    socketio.run(app, port=8000)