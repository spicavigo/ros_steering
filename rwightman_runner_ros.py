# Udacity Self-Driving Car Challenge 2 ROS node runner
# Author: Ross Wightman, rwightman dot gmail
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import rospy
import argparse
import numpy as np
from rwightman_model import RwightmanModel
from steering_node import SteeringNode


def main():
    parser = argparse.ArgumentParser(description='Model Runner for team rwightman')
    parser.add_argument('--alpha', type=float, default=0.1, help='Path to the metagraph path')
    parser.add_argument('--graph_path', type=str, help='Path to the self contained graph def')
    parser.add_argument('--metagraph_path', type=str, help='Path to the metagraph path')
    parser.add_argument('--checkpoint_path', type=str, help='Path to the checkpoint path')
    parser.add_argument('--debug_print', dest='debug_print', action='store_true',
                        help='Debug print of predicted steering commands')
    args = parser.parse_args()

    def get_model():
        model = RwightmanModel(
            alpha=args.alpha,
            graph_path=args.graph_path,
            metagraph_path=args.metagraph_path,
            checkpoint_path=args.checkpoint_path)
        # Push one empty image through to ensure Tensorflow is ready.
        # There is typically a large wait on the first frame through.
        model.predict(np.zeros(shape=[480, 640, 3]))
        return model

    def process(model, img):
        steering_angle = model.predict(img)
        if args.debug_print:
            print(steering_angle)
        return steering_angle

    node = SteeringNode(get_model, process)
    rospy.spin()


if __name__ == '__main__':
    main()
