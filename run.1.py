import argparse
import logging
import sys
import time
# server
import falcon
import base64
import io
import cv2
from imageio import imread
from wsgiref import simple_server

t0 = time.time()
from tf_pose import common
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')

class RecognitionService(object):
    def __init__(self, estimator):
        self.e = estimator

    def on_get(self, req, resp):
        """Handles GET requests"""
        resp.status = falcon.HTTP_200  # This is the default status
        logger.info("get")
        logger.info(self.e)
        logger.info(req)
        resp.body = ('\nTwo things awe me most, the starry sky '
                     'above me and the moral law within me.\n'
                     '\n'
                     '    ~ Immanuel Kant\n\n')
    def on_post(self, req, resp):
        """Handles POST requests"""
        print(req)
        resp.status = falcon.HTTP_302
        resp.location = "/"

class HomeService(object):
    def __init__(self):
        self.index_file = open("index.html", "r").read()
    def on_get(self, req, resp):
        """Handles GET requests"""
        resp.status = falcon.HTTP_200  # This is the default status
        resp.content_type = "text/html"
        resp.body = self.index_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    elapsed = time.time() - t0
    logger.info('initialized graph in %.4f seconds.' % elapsed)
    t_falcon = time.time()
    # falcon.API instances are callable WSGI apps
    app = falcon.API()
    # Resources are represented by long-lived class instances
    recognition = RecognitionService(e)
    home = HomeService()
    # things will handle all requests to the '/things' URL path
    app.add_route('/recognize', recognition)
    app.add_route('/', home)

    httpd = simple_server.make_server('0.0.0.0', 8000, app)
    elapsed = time.time() - t_falcon
    logger.info('initialized falcon server in %.4f seconds.' % elapsed)
    logger.info('Serving at %s' % '0.0.0.0:8000')
    httpd.serve_forever()



    # estimate human poses from a single image !
    image = common.read_imgfile(args.image, None, None)
    if image is None:
        logger.error('Image can not be read, path=%s' % args.image)
        sys.exit(-1)
    t = time.time()
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    elapsed = time.time() - t

    logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))
    print(humans)

# based off of https://stackoverflow.com/questions/45923296/convert-base64-string-to-an-image-thats-compatible-with-opencv
def read_img_b64(b64_string, width=None, height=None):
    # reconstruct image as an numpy array
    img = imread(io.BytesIO(base64.b64decode(b64_string)))

    val_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if width is not None and height is not None:
        val_image = cv2.resize(val_image, (width, height))
    return val_image