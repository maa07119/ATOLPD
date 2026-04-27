import numpy as np

def get_optimal_combination(power_constraint, power_list, latency_list, epochs = None):
    # get batches and freq indices under power constraint
    batches_indices, frequencies_indices = np.where(power_list < power_constraint)

    # select fastest frequecy per batch 
    batches_freq = []
    for batch_idx in np.unique(batches_indices):
        freq_last_idx = np.where(batches_indices == batch_idx)[0][-1]
        freq_idx = frequencies_indices[freq_last_idx]
        batches_freq.append((batch_idx, freq_idx))
    
    # get selected latencies and multiply by epochs
    selected_latencies = [latency_list[batch_idx, freq_idx] * epochs[batch_idx] for batch_idx, freq_idx in batches_freq]

    # get optimal combination
    batch_index, freq_index = batches_freq[np.argmin(selected_latencies)]
    
    return batch_index, freq_index, np.min(selected_latencies)



def get_max_freq_for_bs(power_constraint, power_list, batch_size):
    # get batches and freq indices under power constraint
    freq_indices = np.where(power_list[batch_size] < power_constraint)
    return freq_indices[0][-1]
   

