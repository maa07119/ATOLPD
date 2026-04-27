devices_channels = {
    "NANO": {
        "dir_path": '/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/',
        "channels": ['VDD_IN', 'VDD_GPU', 'VDD_CPU'],
        "gpu_path": "/sys/devices/gpu.0/devfreq/57000000.gpu",
        "cpu_path": "/sys/devices/system/cpu",
        "available_gpu_frequencies": [153600000, 307200000, 460800000, 614400000, 768000000, 921600000]
    },
    "TX2NX": {
        "dir_path": '/sys/bus/i2c/drivers/ina3221x/2-0040/iio:device0',
        "channels": ['VDD_IN', 'VDD_CPU_GPU', 'VDD_SOC'],
        "gpu_path": "/sys/devices/gpu.0/devfreq/17000000.gp10b",
        "cpu_path": "/sys/devices/system/cpu",
        "available_gpu_frequencies": [114750000, 318750000, 522750000, 726750000, 930750000, 1134750000, 1300500000]
    }
}

vision_models = ['resnet18', 'resnet50', 'mobilenet_v2', 'densenet']
language_models = ['lstm', 'transformer']
