"""
Load team chauffeur steering model
"""
import argparse
import tempfile
from collections import deque

import cv2
import numpy as np
import rospy
from keras.models import load_model
from keras import metrics
from keras.models import load_model

from steering_node import SteeringNode


class ChauffeurModel(object):
    def __init__(self, cnn_path, lstm_path):
        self.encoder = self.load_encoder(cnn_path)
        self.lstm = load_model(lstm_path)

        # hardcoded from final submission model
        self.scale = 16.
        self.timesteps = 100

    def load_encoder(self, cnn_path):
        model = load_model(cnn_path)
        deep_copy.layers.pop()
        deep_copy.outputs = [deep_copy.layers[-1].output]
        deep_copy.layers[-1].outbound_nodes = []

        return deep_copy_model_weights(deep_copy)

    def make_stateful_predictor(self):
        steps = deque()

        def predict_fn(img):
            # preprocess image to be YUV 320x120 and equalize Y histogram
            img = cv2.resize(img, (320, 240))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img = img[120:240, :, :]
            img[:,:,0] = cv2.equalizeHist(img[:,:,0])
            img = ((img-(255.0/2))/255.0)

            # apply feature extractor
            img = self.encoder.predict_on_batch(img.reshape((1, 120, 320, 3)))

            # initial fill of timesteps
            if not len(steps):
                for _ in xrange(self.timesteps):
                    steps.append(img)

            # put most recent features at end
            steps.popleft()
            steps.append(img)

            timestepped_x = np.empty((1, self.timesteps, img.shape[1]))
            for i, img in enumerate(steps):
                timestepped_x[0, i] = img

            return self.lstm.predict_on_batch(timestepped_x)[0, 0] / self.scale

        return predict_fn


def deep_copy_model_weights(model):
    """
    Create a deep copy of a tensorflow model weights.

    @param model - tensorflow model
    @return - copy of model
    """
    _, tmp_path_json = tempfile.mkstemp()
    _, tmp_path_weights = tempfile.mkstemp()
    try:
        # serialize model to JSON
        model_json = model.to_json()
        with open(tmp_path_json, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(tmp_path_weights)

        # load json and create model
        json_file = open(tmp_path_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(tmp_path_weights)
        return loaded_model
    finally:
        os.remove(tmp_path_json)
        os.remove(tmp_path_weights)


def rmse(y_true, y_pred):
    '''Calculates RMSE
    '''
    return K.sqrt(K.mean(K.square(y_pred - y_true)))



metrics.rmse = rmse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Runner for team chauffeur')
    parser.add_argument('cnn_path', type=str, help='Path to cnn encoding model')
    parser.add_argument('lstm_path', type=str, help='Path to lstm model')
    args = parser.parse_args()

    def make_predictor():
        model = ChauffeurModel(args.cnn_path, args.lstm_path)
        return model.make_stateful_predictor()

    def process(predictor, img):
        return predictor(img)

    node = SteeringNode(make_predictor, process)
    rospy.spin()
