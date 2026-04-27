## NN Profiling 
### The device's information must exists in Device/config.py

- cd Device
- To profile Training on Jetson Nano for a gpu frequency with applicable batch sizes run: <br>
sudo python3 run_measurements.py --device_name NANO --mode TRAIN --model_type resnet18 --static_gpu_frequency 921600000 --ml_device GPU --static_cpu_frequency 825600 

- For the whole combinations profiling run:  <br>
sudo bash profile_nano.sh 

Power and Time for the given model will be saved in {model_type}_power.csv and {model_type}_time.csv

### Note: It is a must to have sudo rights to be able to change the device frequencies. For torch and torchvision download version 1.10 and 0.10 compiled for arm.    

### To add another the device's information does not exist in the config.py

- Add your device <br>
  "device_name": {
    <br>
        "dir_path": "", # (Power measurements directory path) <br>
        "channels": [""] # (available power channels),<br>
        "gpu_path": "", (To add set new frequency)<br> 
        "cpu_path": "", (To add set new frequency)<br>
        "available_gpu_frequencies": [] <br>
    }



## Proxy Dataset Trainer
- cd Server
- To run the trainer, the batch efficiency relation r will be printed at the end: <br>

  - python3 main.py --model_type mobilenet_v2 --dataset SVHN --device cuda:0 --epochs 20 --num_classes 10 --optimizer ADAM --accuracy_threshold 0.2 --pretrained_weights_path <path for trained checkpoint>

- For transformers:
  - python3 main_transformers.py --dataset shakespeare --pretrained_weights_path <path for trained checkpoint>

* Notes:
  * MobileNetV2 does not reach the accuracies that ResNet18 reach so reduce the accuracy threshold. 
  * The number of epochs is an upper bound for search such that a training run ends when reaching the accuracy threshold. Setting epochs to 30 is a safe choice.
  * The interface between the server and devices is not included.
  * For CINIC dataset, it must be manualy download and added to data/CINIC-10.(Steps for downloading https://github.com/BayesWatch/cinic-10)
  * Install nltk package for austen dataset.

## Evaluation 

To run evaluation:
- cd Evaluation
- python3 evaluate.py --model_type resnet18
- Profiling LUTs are provided
 


