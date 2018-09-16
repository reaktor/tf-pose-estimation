"""
file: run.3-ex-out-parallel.py

As able, get the latest image add to queue, then repeat
As able, get the latest queued and process, then repeat

Usage:
```sh
# as printed out humans
docker run --entrypoint="/usr/bin/python3" --volume="$(pwd)/out:/out" -it care-tpe-scripts:latest \
    run.3-ex-out-parallel.py --model=mobilenet_thin --resize=432x368 \
    --image-url="http://192.168.1.132:55627/camera.jpg"
```
"""
import argparse
import logging
import sys
import time
import urllib.request

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


# multithreading
import threading
from queue import Queue

print_lock = threading.Lock()
queue_lock = threading.Lock()

t0 = time.time()

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')

class QueueProcessor(object):
    def __init__(self):
        self.start_time = time.time()
        self.count_frames = 0
        self.start_image_queue = Queue(maxsize=12)
        self.image_queue = Queue(maxsize=6)
        self.print_lock = threading.Lock()
    
    def tprint(self, arg):
        with self.print_lock:
            print(arg)
    def tinfo(self, arg):
        with self.print_lock:
            logger.info(arg)
    def info(self, arg):
        logger.info(arg)
    
    def add_images_to_queue(self):
        while True:
            # maybe block until ready to download new images
            self.tinfo('add_images_to_queue: waiting on self.start_image_queue.put(time.time())')
            self.start_image_queue.put(time.time())
            self.tinfo('add_images_to_queue: continuing from self.start_image_queue.put(time.time())')

            # download latest image
            t_dl = time.time()
            resp = urllib.request.urlopen(image_url)
            elapsed_dl = time.time() - t_dl
            self.tinfo('downloaded image in %.4f seconds. (%s)' % (elapsed_dl, args.image_url))
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            if image is None:
                with self.print_lock:
                    logger.error('Image can not be read from "%s"' % args.image)
            else:
                # add identifying info to item
                self.tinfo('add_images_to_queue: waiting on self.image_queue.put((image_url, image))')
                self.image_queue.put((image_url, image))
                self.tinfo('add_images_to_queue: continuing from self.image_queue.put((image_url, image))')


    def process_image_queue(self):
        w, h = model_wh(args.resize)
        if w == 0 or h == 0:
            e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
        else:
            e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
        
        while True:
            self.tinfo('process_image_queue: waiting on self.image_queue.get()')
            (_image_url, current_image) = self.image_queue.get()
            self.tinfo('process_image_queue: continuing from self.image_queue.get()')
            humans = e.inference(current_image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            started_t = self.start_image_queue.get()
            with self.print_lock:
                self.count_frames += 1
                print(humans)
                self.info('started_t diff %.4fs' % (time.time() - started_t))

                # once completed, trigger getting new images
                self.info('process_image_queue: self.image_queue.task_done()')
                self.image_queue.task_done()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image-url', type=str, default='')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=4.0')

    args = parser.parse_args()

    image_url = args.image_url
    logger.info('Tracking image_url: %s' % image_url)
    elapsed = time.time() - t0
    logger.info('initialized imports and args in %.4f seconds.' % elapsed)

    qp = QueueProcessor()

    try:
        t = threading.Thread(target=qp.process_image_queue)
        t.daemon = True
        t.start()
        t = threading.Thread(target=qp.add_images_to_queue)
        t.daemon = True
        t.start()

        t0 = time.time()
        for i in range(4):
            qp.start_image_queue.put(time.time())

        print(threading.enumerate())
        qp.start_image_queue.join()

    except (KeyboardInterrupt, SystemExit):
        elapsed = time.time() - t0
        print('total frames  %d' % qp.count_frames)
        print('total elapsed %.4fs' % elapsed)
        print('fps           %.4f' % (elapsed / i))
