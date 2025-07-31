from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

def split_data_into_groups(input_pattern, ca1_output, d, m):
    d_input_history = np.zeros((np.size(input_pattern), d))
    for i in range(d-1, np.size(input_pattern)):
        d_input_history[i, :] = input_pattern[i-d+1:i+1]

    num_groups = m**(d-1)
    group = np.zeros(np.size(input_pattern))
    for i in range(d-1):
        group[i] = None

    for i in range(d-1, np.size(input_pattern)):
        for j in range(1, d):
            group[i] += d_input_history[i, j] * m**(j-1)

    return group


def split_data_into_segments(rng, num_data, num_segments):
    data_to_segment = rng.integers(0, num_segments, size=num_data)
    return data_to_segment

def rename_input_pattern(input_seq):
    new_input_pattern = np.zeros(np.size(input_seq))
    No_input_pattern = np.unique(input_seq)
    for idx, input in enumerate(input_seq):
        for new_input, old_input in enumerate(No_input_pattern):
            if input == old_input:
                new_input_pattern[idx] = new_input

    return new_input_pattern

def shift_array(arr, d):
    shift_amount = d-1
    if shift_amount == 0:
        return arr
    else:
        none_padding = np.array([None] * shift_amount, dtype=object)
        shifted_elements = arr[:-shift_amount]
        return np.concatenate((none_padding, shifted_elements))

def _get_data_within_group(group, data_to_group, input_seq, ca1_output, m, d):
    new_data = {}
    new_data['latest_input'] = input_seq[np.where(data_to_group == group)]
    new_data['oldest_input'] = shift_array(input_seq, d)
    new_data['oldest_input']  = new_data['oldest_input'][np.where(data_to_group == group)]
    new_data['state_vars'] = ca1_output[np.where(data_to_group == group)]
    new_data['group'] = data_to_group[np.where(data_to_group == group)]
    new_data['num_data'] = np.size(new_data['latest_input'])

    return new_data

def _get_data_within_segments(segment, data_to_segment, input_seq, ca1_output):
    new_data = {}
    new_data['input_seq'] = input_seq[np.where(data_to_segment == segment)]
    new_data['state_vars'] = ca1_output[np.where(data_to_segment == segment)]
    new_data['segment'] = data_to_segment[np.where(data_to_segment == segment)]
    new_data['num_data'] = np.size(new_data['input_seq'])

    return new_data

def calculate_centroid(z, label, m):
    centroid = []
    for i in range(m):
        centroid.append(np.mean(z[label==i]))
    return centroid

def classify_validate_data(centroid, z_validate):
    estimation = np.zeros(np.size(z_validate))
    for z_idx, z in enumerate(z_validate):
        minimum_distance = np.max(z_validate) - np.min(z_validate)
        for cls, centroid_of_class in enumerate(centroid):
            if abs(centroid_of_class - z) < minimum_distance:
                minimum_distance = abs(centroid_of_class - z)
                estimation[z_idx] = cls
    return estimation
            
def calculate_ER(estimation, ground_truth):
    num_error = 0
    for est, truth in zip(estimation, ground_truth):
        if est != truth:
            num_error += 1
    
    return num_error / np.size(estimation)


def conduct_lda(label, ca1_output, data_to_segment, segment, m, d, group):
    print(f'conduct lda for segment {segment}')
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=1)
    z = lda.fit(ca1_output[data_to_segment!=segment], label[data_to_segment!=segment].tolist()).transform(ca1_output[data_to_segment!=segment])
    z_validate = lda.transform(ca1_output[data_to_segment==segment])
    estimated_class = classify_validate_data(calculate_centroid(z, label[data_to_segment!=segment], m), z_validate)
    #from matplotlib import pyplot as plt
    #colors = ['blue', 'red']
    #for color, i, target_name in zip(colors, [0, 1], [0, 1]):
    #    plt.hist(z_validate[estimated_class==i],
    #                label=target_name)
    #plt.xlabel('LDA1')
    #plt.ylabel('num data point')
    #plt.legend(loc='best', shadow=False, scatterpoints=1)
    #plt.title('LDA')
    #output_filename = 'output_' + str(d) + '_' + str(group) + '_' + str(segment) + 'png'
    #plt.savefig(output_filename)
    #plt.close()
    ER = calculate_ER(estimated_class, label[data_to_segment==segment])
    return ER

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parameters setting for the simulation')
    parser.add_argument(
        '--w_tilde',
        type=float,
        default=1.6, # デフォルト値
        help='Scaling factor of synaptic weight'
    )
    parser.add_argument(
        '--num_ca3_neurons',
        type=int,
        default=100, # デフォルト値
        help='Number of neurons in CA3'
    )
    parser.add_argument(
        '--num_ca1_neurons',
        type=int,
        default=100, # デフォルト値
        help='Number of neurons in CA1'
    )
    parser.add_argument(
        '--depth',
        type=int,
        default=1, # デフォルト値
        help='length of past input to analyze'
    )

    args = parser.parse_args()
    w_tilde = args.w_tilde
    num_ca3_neurons = args.num_ca3_neurons
    num_ca1_neurons = args.num_ca1_neurons

    filename_parts_list = []
    filename_parts_list.append(f"WT{w_tilde:.4f}".replace('.', 'p')) # 小数点を'p'に変換してファイル名に含める
    filename_parts_list.append(f"NC3N{num_ca3_neurons:04d}") # 小数点を'p'に変換してファイル名に含める
    filename_parts_list.append(f"NC1N{num_ca1_neurons:04d}") # 小数点を'p'に変換してファイル名に含める
    filename = ""
    if filename_parts_list:
        filename_parts = f"{'_'.join(filename_parts_list)}"
        print(filename_parts)
    else:
        filename_parts = f""
        print(filename_parts)


    data = np.load('../data/spike_counts' + filename_parts + '.npz')
    rng = np.random.default_rng(100)

    # remove transient period
    data_without_transient = {}
    data_without_transient['time'] = data['time'][100:]
    data_without_transient['state_vars'] = data['state_vars'][100:]
    data_without_transient['input_seq'] = data['input_seq'][100:]     
    data_without_transient['input_seq'] = rename_input_pattern(data_without_transient['input_seq'])

    num_segments = 10
    d = args.depth
    m = 2
    num_groups = m**(d-1)
    num_all_data = 0
    data_without_transient['group'] = split_data_into_groups(data_without_transient['input_seq'], data_without_transient['state_vars'], d, m)
    for i in range(num_groups):
        data_in_group = _get_data_within_group(i,  data_without_transient['group'], data_without_transient['input_seq'], data_without_transient['state_vars'], m, d)
        print(f'LDA for group {i}')
        print(f'num data in group {i} = {data_in_group['num_data']}')
        error_rate = np.zeros(num_segments)
        data_to_segments = split_data_into_segments(rng, data_in_group['num_data'], num_segments)
        for j in range(num_segments):
            error_rate[j] = conduct_lda(data_in_group['oldest_input'], data_in_group['state_vars'], data_to_segments, j, m, d, i)
            print(error_rate[j])
        print(np.mean(error_rate))
        num_all_data += data_in_group['num_data']
    print(f'number of all data is {num_all_data}')
