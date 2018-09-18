#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

import argparse
import sys
import json
import logging
import numpy as np

from sys import stdin
import time

# multithreading
import threading
from queue import Queue

queue_lock = threading.Lock()

import cv2
from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator import TfPoseEstimator
from feature_detection.source import UriPoller
from feature_detection.sink import OutputSink

from keypoint_extraction import __version__

_logger = logging.getLogger(__name__)


FIELDS_MAP = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
    "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye",
    "REar", "LEar", "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
]


# Mutation - add annotation
def annotate_human(human, ts):
    human['time'] = ts
    return human

def response_parser(res):
    timestamp = res.getheader('X-Timestamp')
    data = res.read()
    image = np.asarray(bytearray(data), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image, timestamp

def create_pose_estimator(args):
    w, h = model_wh(args.resize)
    w, h = (432, 368) if w == 0 or h == 0 else (w, h)
    estimator = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    resize_ratio = args.resize_out_ratio
    def process(image):
        humans_obj = estimator.inference(
            image, resize_to_default=True, upsample_size=resize_ratio
        )
        return [h.to_pyon() for h in humans_obj]

    return process


class QueueProcessor(object):
    def __init__(self, args):
        self.start_image_queue = Queue(maxsize=8) # max size should really be just the number of image processor threads
        self.image_queue = Queue(maxsize=8)
        self.human_queue = Queue(maxsize=12)
        self.print_lock = threading.Lock()
        self.args = args
    
    def tprint(self, arg):
        with self.print_lock:
            print(arg)
    def tinfo(self, arg):
        with self.print_lock:
            logger.info(arg)
    def info(self, arg):
        logger.info(arg)
    
    def add_images_to_queue(self):
        reader = UriPoller(self.args.image_url, response_parser)
        while True:
            #! start_time = time.time()
            # maybe block until ready to download new images
            #! self.tinfo('add_images_to_queue: (waiting       ) self.start_image_queue.get()')
            _req_time = self.start_image_queue.get()
            #! self.tinfo('add_images_to_queue: (waited %.4fs) continuing from self.start_image_queue.get()' % (time.time() - start_time))
            # download latest image
            #! t_dl = time.time()
            
            current_image, timestamp = reader.next()

            #! elapsed_dl = time.time() - t_dl
            #! self.tinfo('add_images_to_queue: (downlo %.4fs) downloaded image %s' % (elapsed_dl, args.image_url))
            self.start_image_queue.task_done()

            # add identifying info to item
            #! self.tinfo('add_images_to_queue: (waiting       ) self.image_queue.put((image_url, image))')
            #! t_put = time.time()
            self.image_queue.put((current_image, timestamp))
            #! with self.print_lock:
            #!     self.info('add_images_to_queue: (waited %.4fs) continuing from self.image_queue.put((image_url, image))' % (time.time() - t_put))
            #!     self.info('add_images_to_queue: (loop   %.4fs) completed loop' % (time.time() - start_time))

            time.sleep(0.1)

    def process_image_queue(self):
        estimator = create_pose_estimator(self.args)
        while True:
            #! start_time = time.time()
            #! self.tinfo('process_image_queue: (waiting       ) self.start_image_queue.put(time.time())')
            self.start_image_queue.put(time.time(), block=False)
            #! self.tinfo('process_image_queue: (waited %.4fs) continuing from self.start_image_queue.put(time.time())' % (time.time() - start_time))

            #! get_time = time.time()
            #! self.tinfo('process_image_queue: (waiting       ) self.image_queue.get()')
            current_image, timestamp = self.image_queue.get()
            #! self.tinfo('process_image_queue: (waited %.4fs) continuing from self.image_queue.get()' % (time.time() - get_time))
            humans = estimator(current_image)
            for h in humans:
                annotate_human(h, timestamp)

            self.image_queue.task_done()
            self.human_queue.put(humans, block=False)
            #! with self.print_lock:
            #!     # once completed, trigger getting new images
            #!     self.info('process_image_queue: (loop   %.4fs) completed loop' % (time.time() - start_time))


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Extract Keypoints")
    parser.add_argument(
        '--version',
        action='version',
        version='vzw-care-feature-detection {ver}'.format(ver=__version__))
    parser.add_argument(
        '-v',
        '--verbose',
        dest="loglevel",
        help="set loglevel to INFO",
        action='store_const',
        const=logging.INFO)
    parser.add_argument(
        '-vv',
        '--very-verbose',
        dest="loglevel",
        help="set loglevel to DEBUG",
        action='store_const',
        const=logging.DEBUG)
    
    parser.add_argument(
        '--image-url', type=str, default=''
    )
    parser.add_argument(
        '--model', type=str, default='mobilenet_thin',
        help='cmu / mobilenet_thin'
    )
    parser.add_argument(
        '--resize', type=str, default='0x0',
        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 '
    )
    parser.add_argument(
        '--resize-out-ratio', type=float, default=4.0,
        help='if provided, resize heatmaps before they are post-processed. default=4.0'
    )
    return parser.parse_args(args)

def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stderr, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )

def main(args):
    """Runs keypoint extraction from images to STDOUT.
        Args:
          args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    qp = QueueProcessor(args)

    writer = OutputSink()
    try:
        # start listening for ability to add images to process queue
        t = threading.Thread(target=qp.add_images_to_queue)
        t.daemon = True
        t.start()
        # start downloading the first image.
        # also should ensure that the image getter is always one image ahead of the processor
        qp.start_image_queue.put(time.time())

        # the processors will be ready
        for i in range(2):
            t = threading.Thread(target=qp.process_image_queue)
            t.daemon = True
            t.start()

        # huemon tracker
        t0 = None
        c0 = 0
        while True:
            if t0 is None:
                t0 = time.time()
            human = qp.human_queue.get()
            with queue_lock:
                writer.next(human)
                c0 += 1
            qp.human_queue.task_done()


    except (KeyboardInterrupt, SystemExit):
        elapsed = time.time() - t0
        print()
        print('total frames  %d' % c0)
        print('total elapsed %.4fs' % elapsed)
        print('avg fps       %.4f' % (c0 / elapsed))
        print('avg sec       %.4fs' % (elapsed / c0))

def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])

if __name__ == "__main__":
    run()
