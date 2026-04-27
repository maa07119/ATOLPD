from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from helpers import get_optimal_combination, get_max_freq_for_bs
from examples_to_accuracy import ETA_DATASET
import argparse


# MAXIMUM FREQUENCY FOR EACH POWER CONSTRAINT
MAX_FREQ_INDICES = {4.5: 1, 7.0: 3, 10.0: 5}






def get_latency_energy(batch_idx, freq_idx, LUT_T, LUT_E, batches_iterations_actual):
    batch_values = batches_iterations_actual.values()
    batch_values = np.array(list(batch_values))
    batches_mean = np.mean(batch_values, axis = 1)
    batches_std = np.std(batch_values, axis = 1)
    latency = LUT_T[batch_idx][freq_idx] * batches_mean[batch_idx]
    latency_std = LUT_T[batch_idx][freq_idx] * batches_std[batch_idx]

    energy = LUT_E[batch_idx][freq_idx] * batches_mean[batch_idx]

    return latency, latency_std, energy


def configuration_selection(p_max, r, LUT_P, LUT_T, LUT_E, model_type):
    
    if model_type == "mobilenetv2":
        batches = [4, 8, 16, 32, 64]
        frequencies = [153, 307, 460, 614, 768, 912]
        baseline_bs_index = 4
    if model_type == "resnet18":
        batches = [4, 8, 16, 32, 64, 128]
        frequencies = [153, 307, 460, 614, 768, 912]
        baseline_bs_index = 5

    if model_type == "transformers":
        batches = [4, 8, 16, 32, 64, 128]
        frequencies = [153, 307, 460, 614, 768, 912]
        baseline_bs_index = 5
    
    for j, (t_dataset, batches_iterations_gt) in enumerate(ETA_DATASET[model_type].items()):
        print("-"*20)
        print("DATASET:", t_dataset)
        print("-" * 20)
        
        batch_values = batches_iterations_gt.values()
        batch_values = np.array(list(batch_values))
        gt = np.mean(batch_values, axis = 1)

        power_constraint = p / 1000 # convert to W
        
        # prediction
        # Get optimal combination given for the estimated r
        batch_idx, freq_idx, _ = get_optimal_combination(power_constraint, LUT_P,
                                                        LUT_T, r)
        
        pred_l, pred_l_std, pred_e = get_latency_energy(batch_idx, freq_idx, LUT_T, LUT_E, batches_iterations_gt)
        print("Prediction:", f'{int(pred_l)}, (b_idx: {batch_idx}, f_idx: {freq_idx})')

        
        # Largest baseline with recommend frequency
        # baseline_freq_idx = get_max_freq_for_bs(power_constraint, LUT_P, batch_size= baseline_bs_index)
        baseline_freq_idx = MAX_FREQ_INDICES[power_constraint]
        baseline_l, baseline_l_std, baseline_e = get_latency_energy(
            baseline_bs_index, baseline_freq_idx, LUT_T, LUT_E, batches_iterations_gt)
        print("Baseline 1:", f'{int(baseline_l)}, (b_idx: {baseline_bs_index}, f_idx: {baseline_freq_idx})')


        # Select batch size with minimum examples to accuracy 
        data_eff_bs = np.argmin(r)
        data_eff_bs_freq = MAX_FREQ_INDICES[power_constraint]
        baseline2_l, baseline2_l_std, baseline2_e =  get_latency_energy(
            data_eff_bs, data_eff_bs_freq, LUT_T, LUT_E, batches_iterations_gt)
        print("Baseline 2:", f'{int(baseline2_l)}, (b_idx: {data_eff_bs}, f_idx: {data_eff_bs_freq})')



if __name__ == "__main__":
    
    # Add args for power limit to select 
    # Export results in a csv file
    # Split the drawing and export results
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', help = '(resnet18, mobilenetv2, transformers)', default='resnet18')
    args = parser.parse_args()

    # ETA_DATASET = ETA_DATASET_SCHEDULER
    model_type = args.model_type


    df_power = pd.read_csv(f'data/{model_type}/batch_freq_maxpower.csv')
    df_time = pd.read_csv(f'data/{model_type}/batch_freq_time.csv')
    df_energy = pd.read_csv(f'data/{model_type}/batch_freq_energyperepoch.csv')


    """
    Units used:
    - Latency (seconds)
    - Energy (KWh)
    """
    LUT_P = df_power.T.to_numpy()[1:, :] / 1000
    LUT_T = df_time.T.to_numpy()[1:, :]
    LUT_E = df_energy.T.to_numpy()[1:, :] 

    # Converting resuls to from 8000 to evaluation samples
    original_samples = 8000
    evaluation_samples = 4096
    LUT_T *= (int(evaluation_samples) / original_samples)
    LUT_E *= (int(evaluation_samples) / original_samples)



    power_constraints = [4500, 7000, 10000]
    if model_type == "mobilenetv2":
        r = np.array([1., 0.63716814, 0.53982301, 0.51327434, 0.63716814])
    elif model_type == "resnet18":
        r = np.array([0.44915254,  0.34745763, 0.41525424, 0.43220339, 0.77966102, 1.])

    elif model_type == "transformers":
        r = np.array([0.84756098, 0.88414634, 0.82926829, 0.81097561, 0.8597561,  1.])
    
    for p in power_constraints:
        print("POWER CONSTRAINT:", p)
        configuration_selection(p, r, LUT_P, LUT_T, LUT_E, model_type)
        print("*" * 30)
    