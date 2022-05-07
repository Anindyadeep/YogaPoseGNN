docker build -t yogaposegnn_final .
docker run --privileged --device=/dev/video0:/dev/video0 yogaposegnn_final