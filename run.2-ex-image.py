"""
file: run.2-ex-image.py

Simple example file for toying with how numpy images work

Usage:
```sh
# cannot be run from docker container since it tries using gui through cv
python ./run.2-ex-image.py --image_url="http://192.168.1.132:55627/camera.jpg"

# if you get trapped by cv
ps -ax | grep -i python
# and kill the process
kill $PID
```
"""


import argparse
import logging
import sys
import time
import urllib.request
# server
import base64
import io
import cv2
from imageio import imread

t0 = time.time()
import numpy as np

logger = logging.getLogger('ImageExample')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image_url', type=str, default='')

    args = parser.parse_args()
    fps = args.fps
    image_url = args.image_url
    logger.info('Tracking image_url: %s' % image_url)

    # download latest image
    logger.info('Downloading latest image')
    resp = urllib.request.urlopen(image_url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # process latest image
    logger.info('Processing latest image')
    cv2.imshow("Image", image)
    # don't close this window directly, or anticipate having to
    # find python `ps -ax | grep -i python` or `top` &
    # kill python `kill $PID`
    cv2.waitKey(0)

# I was playing with the idea of another service feeding the images in through post requests and base64
# based off of https://stackoverflow.com/questions/45923296/convert-base64-string-to-an-image-thats-compatible-with-opencv
def read_img_b64(b64_string, width=None, height=None):
    # reconstruct image as an numpy array
    img = imread(io.BytesIO(base64.b64decode(b64_string)))

    val_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if width is not None and height is not None:
        val_image = cv2.resize(val_image, (width, height))
    return val_image
