
# Introduction 
Try to use one or several webcam for getting information about what is in the gate.
This project is a POC to determine if webcam can be used as sensor

# Getting Started

1. How to use it  
    This project can only be used in Jetson nano because of Jetson inference dependence
    If needed, change parameters in code for MQTT connection / camera / ROI / etc.
    `$ python3 my_jetson_program.py`

1. Tool  
    Generate gate ROI persistent file in JSON  
    TODO: Merge in main program

1. Version  
    See CHANGELOG.md
	
# Install Tegra for Jetson

1. Look to the back of the Jetson. A QRCode will redirect to the "Getting started" section of your jetson model.
   Here, this is a Jetson Nano Developer Kit : https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkitThe 
   Jetson Nano Developer Kit uses a microSD card as a boot device and for main storage. It’s important to have a card that’s fast and large enough for your projects; the minimum recommended is a 32 GB UHS-1 card.
2. Follow steps depending of your system : https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write 
   Disk image Jetson Nano Developer Kit : https://developer.nvidia.com/jetson-nano-sd-card-image (this image will be different depending of your jetson model)
3. Download a SD Card formatter. I did it with Balena Etcher https://www.balena.io/etcher/
4. Flash the micro SD Card with Etcher and the previous .zip disk image
5. Insert the micro SD Card in the jetson slot and boot
6. Follow the Linux installation guide
7. Check if OpenCV is correctly installed and check the version / build informations : 
   ```py
   import cv2
   print(cv2.__version__)
   print(cv2.getBuildInformation())
   ```
   or by running this command in terminal :
   `python3 -c "import cv2; print(cv2.__version__); print(cv2.getBuildInformation());"`


# Install jetson-inference
```
$ sudo apt-get update
$ sudo apt-get install git cmake python3-pip libpython3-dev python3-numpy
$ git clone --recursive https://github.com/dusty-nv/jetson-inference
$ cd jetson-inference
$ mkdir build
$ cd build
$ cmake ../
```

When the "Model Downloader" prompt, select SSD-Mobilenet-v2 and SSD-Inception-v2 with spacebar. Deselect all the others (*) except if you want to try other detection models.

```
$ make -j4
$ sudo make install
$ sudo ldconfig
```

# Install python dependencies
```
$ python3 -m pip install paho-mqtt 
```

# Things to setup
- MQTT Broker to receive messages sent by the program
- Configure in code the MQTT connection / camera etc.

# Links
* https://github.com/dusty-nv/jetson-inference  
    Official github page for jetson inference
* https://developer.nvidia.com/deepstream-sdk  
    Deepstream est le framework pour faire des applications temps réel sur une jetson
* https://devtalk.nvidia.com/default/topic/1058089/jetson-nano/what-is-pycapsule-objects-in-jetson-inference-python-scripts-/2  
    How to use jetson inference image object type
* https://devtalk.nvidia.com/default/topic/1065497/jetson-nano/can-not-use-opencv-to-display-image-from-jetson-utils-gstcamera  
    How to deal with jetson inference image with openCV or other framework
* https://devblogs.nvidia.com/transfer-learning-toolkit-pruning-intelligent-video-analytics/  
    Pruning
	
* https://github.com/opencv/opencv/archive/refs/tags/4.1.1.zip
* https://github.com/opencv/opencv_contrib/archive/refs/tags/4.1.1.zip
* OpenCV compilation : Executed from /home/jetson/opencv_build 
  ```
  cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_PYTHON_EXAMPLES=ON -D INSTALL_C_EXAMPLES=OFF -D OPENCV_ENABLE_NONFREE=ON -D WITH_CUDA=ON -D WITH_CUDNN=ON -D OPENCV_DNN_CUDA=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D CUDA_ARCH_BIN=10.2 -D WITH_CUBLAS=1 -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.1.1/modules -D HAVE_opencv_python3=ON -D WITH_GSTREAMER=ON ../opencv-4.1.1
  ```