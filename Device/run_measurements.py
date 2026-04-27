import argparse
from nn_test import get_network_power_stats
from set_frequencies import set_static_cpu_frequency, set_static_gpu_frequency
import time
import config
import csv

parser = argparse.ArgumentParser(description='Run Measurements')
parser.add_argument('--device_name', help='Device name to make measurements on. Avialiable in config (NANO, TX2NX)')
parser.add_argument('--mode', help='IDLE, TRAIN, INFERENCE')
parser.add_argument('--ml_device', help='CPU or GPU', default='CPU')
parser.add_argument('--samples_number', help='Number of samples to scale', default=4096, type=int)
parser.add_argument('--epochs', help='Number of epochs (For Training mode)', default=1, type=int)
parser.add_argument('--static_cpu_frequency')
parser.add_argument('--static_gpu_frequency')
parser.add_argument('--model_type', help = 'Model type (Currently avialable resnet18, mobilenet_v2)', default='resnet18')
args = parser.parse_args()


def profile_batch_sizes(model_type, gpu_frequency, batch_sizes, cpu_frequency, m, s, epochs = 1, mode = "TRAIN", ml_device = "GPU"):
    LUT_T = {}
    LUT_P = {}
    print("PROFILING STARTED")
    set_static_gpu_frequency(device = device_name, frequency = gpu_frequency)
    
    for batch_size in batch_sizes[:1]:
        print(batch_size, gpu_frequency)
        time_average, peak_power = get_network_power_stats(device_name=device_name,
                            model_type=model_type,
                            mode = mode,
                            ml_device=ml_device,
                            epochs = epochs,
                            batch_size=batch_size,
                            static_cpu_frequency = cpu_frequency,
                            static_gpu_frequency = gpu_frequency,
                            m = m
                            )

        LUT_T[batch_size] = time_average * (s / batch_size) # scaling to time for s 
        LUT_P[batch_size] = peak_power
        time.sleep(10)

    return LUT_T, LUT_P
        





if __name__ == '__main__':
    print('start')

    mode = args.mode
    device_name = args.device_name
    static_cpu_frequency = args.static_cpu_frequency
    static_gpu_frequency = args.static_gpu_frequency
    ml_device = args.ml_device
    model_type = args.model_type
    epochs = args.epochs
    s = args.samples_number


    print('Setting static cpu frequencies')
    set_static_cpu_frequency(device = device_name, frequency = static_cpu_frequency)
    time.sleep(5)

    print(model_type)
    if model_type == "resnet18":
        batch_sizes = [4, 8, 16, 32, 64, 128]
    elif model_type == "mobilenet_v2":
        batch_sizes = [4, 8, 16, 32, 64]
    
    
    gpu_frequencies = config.devices_channels[device_name]["available_gpu_frequencies"]
    print(gpu_frequencies)
    LUT_T, LUT_P = profile_batch_sizes(model_type= model_type,
                            gpu_frequency= static_gpu_frequency,
                            batch_sizes = batch_sizes,
                            cpu_frequency = static_cpu_frequency,
                            m = 20,
                            s = s,
                            epochs = epochs,
                            mode = "TRAIN",
                            ml_device = ml_device
                            )

    print("TIME LUT:", LUT_T)
    print("POWER LUT:", LUT_P)
    print('Profiling Done')

    with open(f'{model_type}_power.csv', 'a+', encoding='UTF8') as file:
        writer = csv.writer(file)
        batch_power_values = list(LUT_P.values()) 
        writer.writerow([static_gpu_frequency] + batch_power_values)
    
    with open(f'{model_type}_time.csv', 'a+', encoding='UTF8') as file:
        writer = csv.writer(file)
        batch_time_values = list(LUT_T.values()) 
        writer.writerow([static_gpu_frequency] + batch_time_values)
    
    

