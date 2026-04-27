import os
import threading
import csv
import time


class PowerMonitor(threading.Thread):

    def __init__(self, event, device_config, static_cpu_frequency, static_gpu_frequency, sleep_time=1):
        super(PowerMonitor, self).__init__()
        self.event = event
        self.power_dir_path = device_config['dir_path']
        self.power_channels = device_config['channels']
        self.cpu_path = device_config['cpu_path']
        self.gpu_path = device_config['gpu_path']
        self.sleep_time = sleep_time
        self.static_gpu_frequency = static_gpu_frequency
        self.static_cpu_frequency = static_cpu_frequency
        self.peak_power = 0

    def run(self):
        start_time = time.time()
        while not self.event.is_set():
            # Checking static freq while training
            if not self.check_cpu_frequency():
                print(f'CPU frequency is not {self.static_cpu_frequency}')
                raise Exception('CPU frequency is not {self.static_cpu_frequency}')
                break
            if not self.check_gpu_frequency():
                print(f'GPU frequency is not {self.static_gpu_frequency}')
                raise Exception('GPU frequency is not {self.static_gpu_frequency}')
                break

            # Get power stats
            power_value = self.get_power_stats(self.power_dir_path, self.power_channels)
            self.peak_power = max(self.peak_power, power_value)
            
            time.sleep(self.sleep_time)

    def get_power_stats(self, stats_dir_path: str, channels: list):
        file_path = os.path.join(stats_dir_path, f'in_power0_input')
        with open(file_path, 'r') as f:
            power = float(f.read())
        return power

    def check_cpu_frequency(self, no_cpus = 4):
        for i in range(no_cpus):
            with open(f'{self.cpu_path}/cpu{i}/cpufreq/scaling_cur_freq', 'r') as f:
                if f.read().strip() != self.static_cpu_frequency:
                    return False
        return True

    def check_gpu_frequency(self):
        with open(f'{self.gpu_path}/cur_freq', 'r') as f:
            return f.read().strip() == str(self.static_gpu_frequency)
        
    def get_peak_power(self):
        return self.peak_power
