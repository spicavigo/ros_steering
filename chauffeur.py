"""
Load team chauffeur steering model
"""
import argparse
from collections import deque

import rospy
import numpy as np
from keras.models import load_model

from steering_node import SteeringNode


class ChauffeurModel(object):
    def __init__(self, cnn_path, lstm_path):
        self.encoder = self.load_encoder(cnn_path)
        self.lstm = load_model(lstm_path)

        # hardcoded from final submission model
        self.scale = 16.
        self.timesteps = 50

    def load_encoder(self, cnn_path):
        model = load_model(cnn_path)

        # pop off layers until we reach flatten
        while 'flatten' not in model.layers[-1].name:
            model.layers.pop()
            model.outputs = [model.layers[-1].output]
            model.layers[-1].outbound_nodes = []

        return model

    def make_stateful_predictor(self):
        steps = deque()

        def predict_fn(img):
            # TODO: should we do cropping and histogram equalization here?

            # apply feature extractor
            img = self.encoder.predict_on_batch(img.reshape((1, 120, 320, 3)))

            # initial fill of timesteps
            if not len(steps):
                for _ in xrange(self.timesteps):
                    steps.append(img)

            # put most recent features at end
            steps.popleft()
            steps.append(img)

            timestepped_x = np.empty((1, self.timesteps, 120, 320, 3))
            for i, img in enumerate(steps):
                timestepped_x[0, i] = img

            return self.lstm.predict_on_batch(timestepped_x)[0, 0] / self.scale

        return predict_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Runner for team chauffeur')
    parser.add_argument('cnn_path', type=str, help='Path to cnn encoding model path')
    parser.add_argument('lstm_path', type=str, help='Path to lstm model path')
    args = parser.parse_args()

    def make_predictor():
        model = ChauffeurModel(args.cnn_path, args.lstm_path)
        return model.make_stateful_predictor()

    def process(predictor, img):
        return predictor(img)

    node = SteeringNode(make_predictor, process)
    rospy.spin()
