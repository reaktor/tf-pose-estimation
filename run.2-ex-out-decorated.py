import argparse
import logging
import sys
import time
import urllib.request
# server
import cv2

t0 = time.time()
from tf_pose import common
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image-url', type=str, default='')
    parser.add_argument('--fps', type=int, default=4,
                        help='default=4')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=1.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()
    fps = args.fps
    image_url = args.image_url
    logger.info('Tracking image_url: %s' % image_url)

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    elapsed = time.time() - t0
    logger.info('initialized graph in %.4f seconds.' % elapsed)

    wait_gap_secs = 1 / fps
    wait_til = time.time()
    if True:
        wait = wait_til - time.time()
        if wait > 0:
            logger.info('Finished frame %.4f seconds early. Sleeping' % wait)
            time.sleep(wait)
        wait_til = time.time() + wait_gap_secs
        # download latest image
        t_dl = time.time()
        resp = urllib.request.urlopen(image_url)
        elapsed = time.time() - t_dl
        logger.info('downloaded image: %s in %.4f seconds.' % (args.image_url, elapsed))
        t_conv = time.time()
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        cv2.imwrite('/out/text.png', image)
        elapsed = time.time() - t_conv
        logger.info('converted image: %s in %.4f seconds.' % (args.image_url, elapsed))

        # process latest image
        # estimate human poses from a single image !
        if image is None:
            logger.error('Image can not be read, url=%s' % args.image_url)
            time.sleep(5)
        t = time.time()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        elapsed = time.time() - t
        logger.info('inference image: %s in %.4f seconds.' % (args.image_url, elapsed))
        print(humans)
