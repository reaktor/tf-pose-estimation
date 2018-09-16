"""
file: run.3-ex-out-serial.py

As able, get the latest image and analyze in sequence, then repeat

Usage:
```sh
# as decorated images,
# notice the docker --volume setting matches python --out-dir value
docker run --entrypoint="/usr/bin/python3" --volume="$(pwd)/out:/out" -it care-tpe-scripts:latest \
    run.3-ex-out-serial.py --model=mobilenet_thin --resize=432x368 \
    --fps=4 --out-dir='/out' \
    --image-url="http://192.168.1.132:55627/camera.jpg"

# as printed out humans, use --out-dir=''
docker run --entrypoint="/usr/bin/python3" --volume="$(pwd)/out:/out" -it care-tpe-scripts:latest \
    run.3-ex-out-serial.py --model=mobilenet_thin --resize=432x368 \
    --fps=4 --out-dir='' \
    --image-url="http://192.168.1.132:55627/camera.jpg"
```
"""
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
                        help='Limit the frames per second to this value. default=4')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=4.0')
    # added
    parser.add_argument('--out-dir', type=str, default='',
                        help='if provided, write output images here')
    parser.add_argument('--prefix', type=str, default='',
                        help='if provided, prefixes the output images')

    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    fps = args.fps
    image_url = args.image_url
    out_dir, prefix = args.out_dir, args.prefix
    logger.info('Tracking image_url: %s' % image_url)
    elapsed = time.time() - t0
    logger.info('initialized graph in %.4f seconds.' % elapsed)

    wait_gap_secs = 1 / fps
    wait_til = start_time = time.time()
    tn_0 = tn_dl = tn_conv = tn_inf = tn_dec = tn_print = 0
    i = 0
    while True:
        try:
            wait = wait_til - time.time()
            if wait > 0:
                logger.info('Finished frame %.4f seconds early. Sleeping' % wait)
                time.sleep(wait)
            wait_til = time.time() + wait_gap_secs
            # download latest image
            t_0 = t_dl = time.time()
            resp = urllib.request.urlopen(image_url)
            elapsed_dl = time.time() - t_dl
            logger.info('downloaded image in %.4f seconds. (%s)' % (elapsed_dl, args.image_url))
            t_conv = time.time()
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            # write out the input image
            # if out_dir:
            #     cv2.imwrite('%s/%srun3-input-%d.png' % (out_dir, prefix, i), image)
            elapsed_conv = time.time() - t_conv
            logger.info('converted image in %.4f seconds. (%s)' % (elapsed_conv, args.image_url))

            # process latest image
            # estimate human poses from a single image !
            if image is None:
                logger.error('Image can not be read, url=%s' % args.image_url)
                time.sleep(5)
            t_inf = time.time()
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            elapsed_inf = time.time() - t_inf
            logger.info('inference image in %.4f seconds.' % (elapsed_inf))
            elapsed_print = elapsed_dec = 0
            if out_dir:
                t_dec = time.time()
                image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
                image_out_path = '%s/%srun3-decorated-%d.png' % (out_dir, prefix, i)
                cv2.imwrite(image_out_path, image)
                elapsed_dec = time.time() - t_dec
                logger.info('decorate and write image in %.4f seconds. (%s)' % (elapsed_dec, image_out_path))
            else:
                t_print = time.time()
                print(humans)
                elapsed_print = time.time() - t_print
                logger.info('print humans in %.4f seconds.' % (elapsed_print))
            
            elapsed_0 = time.time() - t_0
            logger.info('total frame in %.4f seconds.' % (elapsed_0))

            tn_dl += elapsed_dl
            tn_conv += elapsed_conv
            tn_inf += elapsed_inf
            tn_dec += elapsed_dec
            tn_print += elapsed_print
            tn_0 += elapsed_0
            i += 1

        except (KeyboardInterrupt, SystemExit):
            avg_fps = i / (time.time() - start_time)
            print('')
            logger.info('avg secs:  %.4f (dl)' % (tn_dl / i))
            logger.info('avg secs:  %.4f (conv)' % (tn_conv / i))
            logger.info('avg secs:  %.4f (inf)' % (tn_inf / i))
            logger.info('avg secs:  %.4f (dec)' % (tn_dec / i))
            logger.info('avg secs:  %.4f (print)' % (tn_print / i))
            logger.info('avg total: %.4f (0)' % (tn_0 / i))
            logger.info('====================')
            logger.info('overall: %.4f fps' % avg_fps)
            raise
