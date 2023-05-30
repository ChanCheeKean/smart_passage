# Introduction 
Try to use one or several webcam for getting information about what is in the gate.
This project is a POC to determine if webcam can be used as sensor

# Getting Started
1. Requirement  
    - Jetson Nano
    - Python3 and OpenCV for Jetson Nano
    - Jetson inference with SSD-Mobilenet-v2 object detection model
    
2. Installation process  
    - Install NVIDIA Tegra from NVIDIA SDK Manager
        - https://developer.nvidia.com/nvidia-sdk-manager
        - https://docs.nvidia.com/sdk-manager/index.html
    
    - Install python3 pip missing modules


    
3. How to use it  
    This project can only be used in Jetson nano because of Jetson inference dependence
    If needed, change parameters in code for MQTT connection / camera / ROI / etc.
    Run python3 my_jetson_program.py

4. Tool  
    Generate gate ROI persistent file in JSON  
    TODO: Merge in main program

5. Version  
    See CHANGELOG.md

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