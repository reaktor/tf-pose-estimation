FROM care-tpe-full:preprocess

# RUN cd /root/tf-openpose/ && \
# apt-get update && \
# cd /root/tf-openpose/tf_pose/pafprocess/ && \
# apt-get -y install swig && \
# swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace

COPY *.txt /root/tf-openpose/
RUN cd /root/tf-openpose/ && pip3 install -r requirements.txt
ENV PYTHONPATH=/root/tf-openpose/tf_pose/:/root/tf-openpose/
RUN pip install -v --no-binary :all: falcon

# Only update source files - order starting with least frequently updated
COPY tf_pose/*.py /root/tf-openpose/tf_pose/
COPY images/* /root/tf-openpose/images/
COPY *.html /root/tf-openpose/
COPY *.py /root/tf-openpose/

# COPY models /root/tf-openpose/
WORKDIR /root/tf-openpose/

# ENTRYPOINT ["python3", "run.py"]
ENTRYPOINT ["/bin/bash"]
