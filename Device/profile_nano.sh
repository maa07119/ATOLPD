for f in 153600000 307200000 460800000 614400000 768000000 921600000; do
    python3 run_measurements.py --device_name NANO --mode TRAIN --model_type resnet18 --static_gpu_frequency $f --ml_device GPU --static_cpu_frequency 825600 
    sleep 5
    done
done
