#!flask/bin/python
from flask import Flask, request, jsonify
import common.utils as utils
import common.tf_classify as image_classifier
import tensorflow as tf

LABELS_FILE = './tf_files/flowers_labels.txt'
MODEL_FILE = './tf_files/flowers_retrained_graph.pb'
UPLOAD_FOLDER = './temp'
PORT = 12480

app = Flask(__name__)


# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
               in tf.gfile.GFile(LABELS_FILE)]

# Unpersists graph from file
with tf.gfile.FastGFile(MODEL_FILE, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


sess = tf.Session()


@app.route('/')
def hello_world():
    return "Hello World!"


@app.route('/classify', methods=['POST', 'GET'])
def classify():
    try:
        filename = ''
        if request.method == 'POST':
            # check if the post request has the image part
            if 'image' not in request.files:
                return jsonify({
                    'message': 'No file part'
                }), 400
            file = request.files['image']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                return jsonify({
                    'message': 'No selected file'
                }), 400
            filename = utils.save_upload_file(path=UPLOAD_FOLDER, file=file)
        elif request.method == 'GET' and request.args.get('image_url', '') != '':
            image_url = request.args.get('image_url')
            filename = utils.download_image_from_url(path=UPLOAD_FOLDER, image_url=image_url)
        else:
            return jsonify({
                'message': 'Action is not defined!'
            }), 404
        if filename != '':
            output = image_classifier.label_image(image_name=filename, label_lines=label_lines, sess=sess)
            return jsonify({
                'filename': filename,
                'labels': output
            })
        else:
            return jsonify({
                'message': 'Something when wrong'
            }), 400
    except Exception as e:
        print(e)
        return repr(e), 500


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=PORT)
