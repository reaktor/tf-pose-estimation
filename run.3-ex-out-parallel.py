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

queue_lock = threading.Lock()

t0 = time.time()

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')

class QueueProcessor(object):
    def __init__(self):
        self.start_image_queue = Queue(maxsize=8) # max size should really be just the number of image processor threads
        self.image_queue = Queue(maxsize=8)
        self.human_queue = Queue(maxsize=12)
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
            #! start_time = time.time()
            # maybe block until ready to download new images
            #! self.tinfo('add_images_to_queue: (waiting       ) self.start_image_queue.get()')
            _req_time = self.start_image_queue.get()
            #! self.tinfo('add_images_to_queue: (waited %.4fs) continuing from self.start_image_queue.get()' % (time.time() - start_time))
            # download latest image
            #! t_dl = time.time()
            try:
                resp = urllib.request.urlopen(image_url)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.tinfo('add_images_to_queue: (error         ) downloading image failed')
                continue

            #! elapsed_dl = time.time() - t_dl
            #! self.tinfo('add_images_to_queue: (downlo %.4fs) downloaded image %s' % (elapsed_dl, args.image_url))
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            self.start_image_queue.task_done()

            if image is None:
                with self.print_lock:
                    logger.error('Image can not be read from "%s"' % args.image)
            else:
                # add identifying info to item
                #! self.tinfo('add_images_to_queue: (waiting       ) self.image_queue.put((image_url, image))')
                #! t_put = time.time()
                self.image_queue.put((image_url, image))
                #! with self.print_lock:
                #!     self.info('add_images_to_queue: (waited %.4fs) continuing from self.image_queue.put((image_url, image))' % (time.time() - t_put))
                #!     self.info('add_images_to_queue: (loop   %.4fs) completed loop' % (time.time() - start_time))



    def process_image_queue(self):
        w, h = model_wh(args.resize)
        if w == 0 or h == 0:
            e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
        else:
            e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
        
        while True:
            #! start_time = time.time()
            # trigger ready to process another image
            #! self.tinfo('process_image_queue: (waiting       ) self.start_image_queue.put(time.time())')
            self.start_image_queue.put(time.time(), block=False)
            #! self.tinfo('process_image_queue: (waited %.4fs) continuing from self.start_image_queue.put(time.time())' % (time.time() - start_time))
            
            #! get_time = time.time()
            #! self.tinfo('process_image_queue: (waiting       ) self.image_queue.get()')
            (_image_url, current_image) = self.image_queue.get()
            #! self.tinfo('process_image_queue: (waited %.4fs) continuing from self.image_queue.get()' % (time.time() - get_time))
            humans = e.inference(current_image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

            self.image_queue.task_done()

            self.human_queue.put(humans, block=False)
            #! with self.print_lock:
            #!     # once completed, trigger getting new images
            #!     self.info('process_image_queue: (loop   %.4fs) completed loop' % (time.time() - start_time))

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

        print(threading.enumerate())

        # huemon tracker
        t0 = None
        c0 = 0
        while True:
            if t0 is None:
                t0 = time.time()
            human = qp.human_queue.get()
            with queue_lock:
                print(human)
                c0 += 1
            qp.human_queue.task_done()


    except (KeyboardInterrupt, SystemExit):
        elapsed = time.time() - t0
        print()
        print('total frames  %d' % c0)
        print('total elapsed %.4fs' % elapsed)
        print('avg fps       %.4f' % (c0 / elapsed))
        print('avg sec       %.4fs' % (elapsed / c0))
