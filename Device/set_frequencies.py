import os
import config

def set_static_cpu_frequency(device, frequency, cpus_no = 4):
    print(config.devices_channels[device])
    cpu_path = config.devices_channels[device]['cpu_path']
    print(f'Setting cpu frequencies to {frequency}')
    # set minimum and maximum frequency to same frequency in all cpus
    for i in range(cpus_no): 
        out_min = os.system(f'echo {frequency} > {cpu_path}/cpu{i}/cpufreq/scaling_min_freq')
        out_max = os.system(f'echo {frequency} > {cpu_path}/cpu{i}/cpufreq/scaling_max_freq')
        if out_min != 0 or out_max != 0:
            raise Exception(f'CPU {i} Frequency update failure')


def set_static_gpu_frequency(device, frequency):
    gpu_path = config.devices_channels[device]['gpu_path']
    # set minimum and maximum frequency to same frequency
    out_min = os.system(f'echo {frequency} > {gpu_path}/min_freq')
    out_max = os.system(f'echo {frequency} > {gpu_path}/max_freq')
    if out_min != 0 or out_max != 0:
        raise Exception('GPU Frequency update failure')


