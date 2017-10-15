#!/bin/bash
gpu_mode=0
while getopts g: opt; do
  case $opt in
  g)
    gpu_mode=$OPTARG
    ;;
  esac
done

shift $((OPTIND-1))
echo "gpu_mode = ${gpu_mode}"
source /opt/DL/caffe-ibm/bin/caffe-activate
source /opt/DL/openblas/bin/openblas-activate
source /opt/DL/tensorflow/bin/tensorflow-activate
source /opt/DL/theano/bin/theano-activate
source /opt/DL/torch/bin/torch-activate
source /opt/DL/digits/bin/digits-activate


if [ $gpu_mode == 0 ];then
  echo "Setting some cpumode configs"
  sudo ln -fs /usr/local/cuda-8.0/targets/ppc64le-linux/lib/stubs/libcuda.so /usr/local/cuda-8.0/targets/ppc64le-linux/lib/stubs/libcuda.so.1
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/targets/ppc64le-linux/lib/stubs/
else
  echo "GPU Mode"
fi

echo "Starting Jupyter"
jupyter notebook --ip=0.0.0.0 --allow-root --port=5050 --NotebookApp.token="" --no-browser> >(tee -a /tmp/tensorflow.stdout.log) 2> >(tee -a /tmp/tensorflow.log >&2) &
#tensorboard --logdir=/data/mldl-101/graphs &
