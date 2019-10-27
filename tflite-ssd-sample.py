from flask import Flask, Response, render_template
from imutils.video.pivideostream import PiVideoStream
import cv2
import time
import tensorflow as tf
import numpy as np


app = Flask(__name__)
camera = PiVideoStream(resolution=(400, 304), framerate=5).start()
time.sleep(2)
# TensorFlow Liteランタイムの初期化
interpreter = tf.lite.Interpreter(model_path='graph.tflite')
interpreter.allocate_tensors()
interpreter.set_num_threads(4)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

def detect(frame):
    initial_h, initial_w, channels = frame.shape
    resized = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(resized, axis=0)
    # 画像データの入力
    interpreter.set_tensor(input_details[0]['index'], input_data)
    # 推論実行
    interpreter.invoke()
    # 結果の取得
    detected_boxes = interpreter.get_tensor(output_details[0]['index'])
    detected_classes = interpreter.get_tensor(output_details[1]['index'])
    detected_scores = interpreter.get_tensor(output_details[2]['index'])
    num_boxes = interpreter.get_tensor(output_details[3]['index'])
    # 描画処理
    for i in range(int(num_boxes)):
        top, left, bottom, right = detected_boxes[0][i]
        classId = int(detected_classes[0][i])
        score = detected_scores[0][i]
        # COCOデータセットよりラベル0はperson
        if classId != 0:
            continue
        # 70%以上の尤度で描画
        if score < 0.7:
            continue
        # 矩形座標をオリジナル画像用にスケール
        xmin = int(left * initial_w)
        ymin = int(bottom * initial_h)
        xmax = int(right * initial_w)
        ymax = int(top * initial_h)
        label = 'person: {:.2f}%'.format(score * 100)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        cv2.putText(frame, label, (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return frame

def gen(camera):
    while True:
        frame = camera.read()
        processed = detect(frame.copy())
        ret, jpeg = cv2.imencode('.jpg', processed)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/')
def index():
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(
        host='0.0.0.0', 
        debug=False, 
        threaded=True,
        use_reloader=False
    )
