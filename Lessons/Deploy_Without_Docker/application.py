import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
import tensorflow as tf
from keras.models import model_from_json
from PIL import Image
from werkzeug.datastructures import FileStorage
import numpy as np
from flask import Flask, request, jsonify
from flask_restplus import Api, Resource, fields
from keras.models import load_model
from keras.preprocessing.image import img_to_array

application = Flask(__name__)
api = Api(application, version='1.0', title='MNIST Classification',
          description='CNN for MNIST')
name_space = api.namespace('Make_School', description='Methods')

single_parser = api.parser()
single_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)

model = load_model('mnist_cnn.h5')
graph = tf.get_default_graph()


@name_space.route('/prediction')
class CNNPrediction(Resource):
    @api.doc(parser=single_parser, description='Upload an MNIST image')
    def post(self):
        # Load image file from user as a PIL image
        args = single_parser.parse_args()
        image_file = args.file
        img = Image.open(image_file)
        # Perform data preprocessing in image
        image_red = img.resize((28, 28))
        image = img_to_array(image_red)
        x = image.reshape(1, 28, 28, 1)
        x = x/255
        # Make and return the prediction to the user
        with graph.as_default():
            out = model.predict(x)
        return {'prediction': str(np.argmax(out[0]))}


if __name__ == '__main__':
    application.debug = True
    application.run()
