# copy over assets to this new docker image
source ./docker-build.sh
# docker run --entrypoint="/usr/bin/python3" --volume="$(pwd)/out:/out" --expose=8000/tcp -it care-tpe-scripts:latest run.2.py --model=mobilenet_thin --image-url="http://192.168.1.132:55627/camera.jpg" --fps=5 --resize=640x480
# docker run --entrypoint="/usr/bin/python3" --volume="$(pwd)/out:/out" --expose=8000/tcp -it care-tpe-scripts:latest run.2.py --model=mobilenet_thin --image-url="http://192.168.1.132:55627/camera.jpg" --fps=5 --resize=640x480

# Test downloading image
# docker run --entrypoint="/usr/bin/python3" --volume="$(pwd)/out:/out" --expose=8000/tcp -it care-tpe-scripts:latest \
#     run.0-ex-out-decorated.py --model=mobilenet_thin --resize=656x368 \
#     --out-dir=/out --prefix=dl-ball- \
#     --image="https://img.freepik.com/free-photo/ball-guy-soccer-man-playing_1368-1897.jpg?size=338&ext=jpg"
    # --out-dir=/out --prefix=dl-ball- \
    # --image="https://img.freepik.com/free-photo/ball-guy-soccer-man-playing_1368-1897.jpg?size=338&ext=jpg"
    # --out-dir=/out --prefix=dl-lg- \
    # --image="https://webbox.imgix.net/images/qibzrzupgftwhniw/95649c96-5ee8-4400-aef7-ca2fb8c065e8.jpg?auto=format,compress&fit=crop&crop=entropy"
# Test downloading images periodically
docker run --runtime=nvidia --entrypoint="/usr/bin/python3" --volume="$(pwd)/out:/out" -it care-tpe-scripts:latest \
    run.3.py --model=mobilenet_thin --resize=432x368 \
    --fps=4 --out-dir='' \
    --image-url="http://192.168.1.132:55627/camera.jpg"
    # --out-dir=/out/frames --prefix=cam- --fps=4 \
    # --image-url="https://img.freepik.com/free-photo/ball-guy-soccer-man-playing_1368-1897.jpg?size=338&ext=jpg"

# docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' container_id
