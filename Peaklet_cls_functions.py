import numpy as np
import straxen

def compute_s1_boundary(parm, area):
    boundary_line = parm[0]*np.exp(-area/parm[1]) + parm[2]
    
    return boundary_line


def data_to_log_decile_log_area_aft(peaklet_data, normalization_factor):
    """
    Converts peaklet data into the current best inputs for the SOM, 
    log10(deciles) + log10(area) + AFT
    Since we are dealing with logs, anything less than 1 will be set to 1
    
    """
    
    # turn deciles into approriate 'normalized' format (maybe also consider L1 normalization of these inputs)
    _,decile_data = compute_wf_and_quantiles(peaklet_data, 10)
    decile_data[decile_data < 1] = 1
    #decile_L1 = np.log10(decile_data)
    decile_log = np.log10(decile_data)
    decile_log_over_max = np.divide(decile_log, normalization_factor[:10])
    
    # Now lets deal with area
    if np.min(peaklet_data['area']) < 0:
        # this might be an issue with the recall function
        # I should also save this value to use
        peaklet_data['area'] = peaklet_data['area']+normalization_factor[11]+1
    elif np.min(peaklet_data['area']) == 1:
        pass # area data is already shifted 
    peaklet_log_area = np.log10(peaklet_data['area'])
    
    peaklet_aft = np.sum(peaklet_data['area_per_channel'][:,:straxen.n_top_pmts], axis = 1) / peaklet_data['area']
    peaklet_aft = np.where(peaklet_aft > 0, peaklet_aft, 0)
    peaklet_aft = np.where(peaklet_aft < 1, peaklet_aft, 1)
    
    print(decile_log.shape)
    print((decile_log / normalization_factor[:10]).shape)
    deciles_area_aft = np.concatenate((decile_log_over_max, 
                                       np.reshape(peaklet_log_area, (len(peaklet_log_area),1))/ normalization_factor[10],
                                       np.reshape(peaklet_aft, (len(peaklet_log_area),1))), axis = 1)
    
    return deciles_area_aft


def compute_wf_and_quantiles(peaks: np.ndarray, bayes_n_nodes: int):
    """
    Compute waveforms and quantiles for a given number of nodes(atributes)
    :param peaks:
    :param bayes_n_nodes: number of nodes or atributes
    :return: waveforms and quantiles
    """
    waveforms = np.zeros((len(peaks), bayes_n_nodes))
    quantiles = np.zeros((len(peaks), bayes_n_nodes))

    num_samples = peaks['data'].shape[1] 
    #modified line, original num_samples = peaks['data'].shape[1] 
    step_size = int(num_samples/bayes_n_nodes)
    steps = np.arange(0, num_samples+1, step_size)

    data = peaks['data'].copy() #data = peaks['data'].copy() 
    data[data < 0.0] = 0.0
    for i, p in enumerate(peaks):
        sample_number = np.arange(0, num_samples+1, 1)*p['dt']
        frac_of_cumsum = np.append([0.0], np.cumsum(data[i, :]) / np.sum(data[i, :]))
        cumsum_steps = np.interp(np.linspace(0., 1., bayes_n_nodes, endpoint=False), frac_of_cumsum, sample_number)
        cumsum_steps = np.append(cumsum_steps, sample_number[-1])
        quantiles[i, :] = cumsum_steps[1:] - cumsum_steps[:-1]

    for j in range(bayes_n_nodes):
        waveforms[:, j] = np.sum(data[:, steps[j]:steps[j+1]], axis=1)
    waveforms = waveforms/(peaks['dt']*step_size)[:, np.newaxis]

    del data
    return waveforms, quantiles

def compute_AFT(data):
    peaklets_aft = np.sum(data['area_per_channel'][:,:straxen.n_top_pmts], axis = 1) / np.sum(data['area_per_channel'], axis = 1) 
    return peaklets_aft