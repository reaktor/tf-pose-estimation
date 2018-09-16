import argparse
import logging
import sys
import time
import urllib.request
# server
# import base64
# import io
import cv2
# from imageio import imread

t0 = time.time()
# from tf_pose import common
import numpy as np
# from tf_pose.estimator import TfPoseEstimator
# from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    # parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--image-url', type=str, default='')
    parser.add_argument('--fps', type=int, default=4,
                        help='default=4')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=4.0')

    args = parser.parse_args()
    fps = args.fps
    image_url = args.image_url
    logger.info('Tracking image_url: %s' % image_url)

    wait_gap_secs = 1 / fps
    wait_til = time.time()
    if True:
        wait = wait_til - time.time()
        if wait > 0:
            logger.info('Finished frame %.4f seconds early. Sleeping' % wait)
            time.sleep(wait)
        wait_til = time.time() + wait_gap_secs
        # download latest image
        logger.info('Downloading latest image')
        resp = urllib.request.urlopen(image_url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # process latest image
        logger.info('Processing latest image')
        cv2.imshow("Image", image)
        cv2.waitKey(0)



        



    # w, h = model_wh(args.resize)
    # if w == 0 or h == 0:
    #     e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    # else:
    #     e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    # elapsed = time.time() - t0
    # logger.info('initialized graph in %.4f seconds.' % elapsed)
    
    # # estimate human poses from a single image !
    # image = common.read_imgfile(args.image, None, None)
    # if image is None:
    #     logger.error('Image can not be read, path=%s' % args.image)
    #     sys.exit(-1)
    # t = time.time()
    # humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    # elapsed = time.time() - t

    # logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))
    # print(humans)

# based off of https://stackoverflow.com/questions/45923296/convert-base64-string-to-an-image-thats-compatible-with-opencv
# def read_img_b64(b64_string, width=None, height=None):
#     # reconstruct image as an numpy array
#     img = imread(io.BytesIO(base64.b64decode(b64_string)))

#     val_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     if width is not None and height is not None:
#         val_image = cv2.resize(val_image, (width, height))
#     return val_image