# Udacity Self-Driving Car Challenge 2 local file runner
# Author: Ross Wightman, rwightman dot gmail
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import pandas as pd
    has_pandas = True
except ImportError:
    has_pandas = False
import numpy as np
import threading
import queue
import time
import os
import sys
import cv2
import argparse
from datetime import datetime
from rwightman_model import RwightmanModel


def get_image_files(folder, types=('.jpg', '.jpeg', '.png')):
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if os.path.splitext(f)[1].lower() in types]


class ProcessorThread(threading.Thread):

    def __init__(self, q, get_model_fn, process_fn):
        super(ProcessorThread, self).__init__(name='processor')
        self.q = q
        self.model = get_model_fn()
        self.process_fn = process_fn
        self.outputs = []

    def run(self):
        print('Entering processing loop...')
        while True:
            item = self.q.get()
            if item is None:
                print("Exiting processing loop...")
                break
            output = self.process_fn(self.model, item)
            self.outputs.append(output)
            self.q.task_done()
        return


def feed_queue(q, image_files):
    timestamps = []
    for f in image_files:
        # Note, all cv2 based decode and split/merge color channel flip resulted in faster throughput, lower
        # CPU usage than PIL Image decode or cv2 + python based slice reversal of color channels.
        image = cv2.imread(f)
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        q.put(image)
        timestamps.append(os.path.splitext(os.path.basename(f))[0])
    q.put(None)
    return timestamps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.1, help='Path to the metagraph path')
    parser.add_argument('--graph_path', type=str, help='Path to the self contained graph def')
    parser.add_argument('--metagraph_path', type=str, help='Path to the metagraph path')
    parser.add_argument('--checkpoint_path', type=str, help='Path to the checkpoint path')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the images')
    parser.add_argument('--target_csv', type=str, help='Path to target csv for optional RMSE calc.')
    parser.add_argument('--debug_print', dest='debug_print', action='store_true',
                        help='Debug print of predicted steering commands')
    args = parser.parse_args()

    if not args.data_dir or not os.path.isdir(args.data_dir):
        print("Invalid data directory: %s" % args.data_dir)
        sys.exit(1)
    image_files = get_image_files(args.data_dir)
    if not image_files:
        print("No images found at: %s" % args.data_dir)
        sys.exit(1)

    def get_model():
        print('%s: Initializing model.' % datetime.now())
        model = RwightmanModel(
            alpha=args.alpha,
            graph_path=args.graph_path,
            metagraph_path=args.metagraph_path,
            checkpoint_path=args.checkpoint_path)
        print('%s: Pushing initial null frame through model.' % datetime.now())
        # Push one empty image through to ensure Tensorflow is ready.
        # There is typically a large wait on the first frame through.
        model.predict(np.zeros(shape=[480, 640, 3]))
        return model

    def process(model, img):
        steering_angle = model.predict(img)
        if args.debug_print:
            print(steering_angle)
        return steering_angle

    q = queue.Queue(20)
    processor = ProcessorThread(q, get_model, process)
    processor.start()

    print('%s: starting execution on (%s).' % (datetime.now(), args.data_dir))
    start_time = time.time()
    timestamps = feed_queue(q, image_files)
    processor.join()
    duration = time.time() - start_time
    images_per_sec = len(image_files) / duration
    print('%s: %d images processed in %s seconds, %.1f images/sec'
          % (datetime.now(), len(image_files), duration, images_per_sec))

    if has_pandas:
        columns_ang = ['frame_id', 'steering_angle']
        predictions_df = pd.DataFrame(
            data={columns_ang[0]: timestamps, columns_ang[1]: processor.outputs},
            columns=columns_ang)
        predictions_df.to_csv('./output_angle.csv', index=False)

        if args.target_csv:
            targets_df = pd.read_csv(args.target_csv, header=0, index_col=False)
            targets = np.squeeze(targets_df.as_matrix(columns=[columns_ang[1]]))
            predictions = np.asarray(processor.outputs)
            mse = ((predictions - targets) ** 2).mean()
            rmse = np.sqrt(mse)
            print("RMSE: %f, MSE: %f" % (rmse, mse))

if __name__ == '__main__':
    main()
