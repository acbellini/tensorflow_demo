1) Install Bazel
    (https://docs.bazel.build/versions/master/install-ubuntu.html#install-on-ubuntu)

    sudo apt-get install openjdk-8-jdk -y
    echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
    curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
    sudo apt-get update && sudo apt-get install bazel -y

2) Install Tensorflow dependencies
    sudo apt-get install python3-numpy python3-dev python3-pip python3-wheel -y
    sudo apt-get install libcupti-dev -y
    sudo apt-get install libprotobuf-dev -y

3) Configure Tensorflow

    git clone git@github.com:tensorflow/tensorflow.git

The sticky_lengths directory has to be copied to $TF_SRC/tensorflow/core/user_ops

Then 

cd $TF_SRC

./configure
        location of python: /usr/bin/python3
        python library path: /usr/lib/python3/dist-packages
        optimization flags: -msse4.2 -msse4.1 -mavx -mavx2 -mfma
        yes jemalloc
        no GCP
        no Hadoop
        no XLA JIT
        no VERBS
        no OpenCL
        yes CUDA
        CUDA 8.0
        CUDA path /usr/local/cuda
        no clang as CUDA compiler
        /usr/bin/gcc as nvcc
        cuDNN 6
        cuDNN path: /usr/lib/x86_64-linux-gnu/
        CUDA compute capability: use suggested (depends on a current gpu)
        no MPI

    bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

4) Install pip package

    pip install /tmp/tensorflow_pkg/<the name of the built whl file>

5) Compile custom kernels

    bazel build --config opt //tensorflow/core/user_ops/sticky_lengths:sticky_engths.so; rm resize.so -f; cp bazel-bin/tensorflow/core/user_ops/sticky_lengths/sticky_lengths.so .

Copy *.so files into /usr/local/lib

