# copy over assets to this new docker image
source ./docker-build.sh

# Test downloading images periodically
docker run --runtime=nvidia --entrypoint="/usr/bin/python3" --volume="$(pwd)/out:/out" -it care-tpe-scripts:latest \
    run.3-ex-out-parallel.py --model=mobilenet_thin --resize=432x368 \
    --fps=4 --out-dir='' \
    --image-url="http://192.168.1.132:55627/camera.jpg"

# docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' container_id
