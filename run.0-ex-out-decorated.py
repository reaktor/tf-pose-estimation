"""
file: run.0-ex-out-decorate.py
Usage
```sh
docker run --entrypoint="/usr/bin/python3" --volume="$(pwd)/out:/out" --expose=8000/tcp -it care-tpe-scripts:latest \
    run.0-ex-out-decorated.py --model=mobilenet_thin --resize=656x368 \
    --out-dir=/out --prefix=dl-ball- \
    --image="https://img.freepik.com/free-photo/ball-guy-soccer-man-playing_1368-1897.jpg?size=338&ext=jpg"
open ./out/dl-ball-0-input.png
open ./out/dl-ball-2-decorated.png
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

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='./images/care-man-kicking-ball.jpg')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=4.0')
    parser.add_argument('--out-dir', type=str, default='./out',
                        help='put output images here')
    parser.add_argument('--prefix', type=str, default='',
                        help='if provided, prefixes the output images')

    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    image = None
    # estimate human poses from a single image !
    if args.image.startswith('http://') or args.image.startswith('https://'):
        image_url = args.image
        logger.info('Downloading latest image')
        resp = urllib.request.urlopen(image_url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    else:
        image = common.read_imgfile(args.image, None, None)

    if image is None:
        logger.error('Image can not be read from "%s"' % args.image)
        sys.exit(-1)

    out_dir, prefix = args.out_dir, args.prefix
    cv2.imwrite('%s/%s0-input.png' % (out_dir, prefix), image)
    t = time.time()
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    # check out these humans!
    print(humans)

    elapsed = time.time() - t

    logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    cv2.imwrite('%s/%s2-decorated.png' % (out_dir, prefix), image)
