import linear_discriminant_analysis
import numpy as np

#if __name__ == '__main__':
#    import argparse
#    parser = argparse.ArgumentParser(description='Parameters setting for the simulation')
#    parser.add_argument(
#        '--w_tilde',
#        type=float,
#        default=1.6, # デフォルト値
#        help='Scaling factor of synaptic weight'
#    )
#    parser.add_argument(
#        '--num_ca3_neurons',
#        type=int,
#        default=100, # デフォルト値
#        help='Number of neurons in CA3'
#    )
#    parser.add_argument(
#        '--num_ca1_neurons',
#        type=int,
#        default=100, # デフォルト値
#        help='Number of neurons in CA1'
#    )
#    parser.add_argument(
#        '--depth',
#        type=int,
#        default=1, # デフォルト値
#        help='length of past input to analyze'
#    )
#
#    args = parser.parse_args()
#    w_tilde = args.w_tilde
#    num_ca3_neurons = args.num_ca3_neurons
#    num_ca1_neurons = args.num_ca1_neurons
#
#    filename_parts_list = []
#    filename_parts_list.append(f"WT{w_tilde:.4f}".replace('.', 'p')) # 小数点を'p'に変換してファイル名に含める
#    filename_parts_list.append(f"NC3N{num_ca3_neurons:04d}") # 小数点を'p'に変換してファイル名に含める
#    filename_parts_list.append(f"NC1N{num_ca1_neurons:04d}") # 小数点を'p'に変換してファイル名に含める
#    filename = ""
#    if filename_parts_list:
#        filename_parts = f"{'_'.join(filename_parts_list)}"
#        print(filename_parts)
#    else:
#        filename_parts = f""
#        print(filename_parts)
#
#    data = np.load('../data/spike_counts' + filename_parts + '.npz')
#    rng = np.random.default_rng(100)
#
#    # remove transient period
#    data_without_transient = {}
#    data_without_transient['time'] = data['time'][100:]
#    data_without_transient['state_vars'] = data['state_vars'][100:]
#    data_without_transient['input_seq'] = data['input_seq'][100:]     
#    data_without_transient['input_seq'] = linear_discriminant_analysis.rename_input_pattern(data_without_transient['input_seq'])
#
#    num_segments = 10
#    d = args.depth
#    m = 2
#    num_groups = m**(d-1)
#    num_all_data = 0
#    data_without_transient['group'] = linear_discriminant_analysis.split_data_into_groups(data_without_transient['input_seq'], data_without_transient['state_vars'], d, m)
#    for i in range(num_groups):
#        data_in_group = linear_discriminant_analysis._get_data_within_group(i,  data_without_transient['group'], data_without_transient['input_seq'], data_without_transient['state_vars'], m, d)
#        print(f'LDA for group {i}')
#        print(f'num data in group {i} = {data_in_group['num_data']}')
#        error_rate = np.zeros(num_segments)
#        data_to_segments = linear_discriminant_analysis.split_data_into_segments(rng, data_in_group['num_data'], num_segments)
#        for j in range(num_segments):
#            error_rate[j] = linear_discriminant_analysis.conduct_lda(data_in_group['oldest_input'], data_in_group['state_vars'], data_to_segments, j, m, d, i)
#            print(error_rate[j])
#        print(np.mean(error_rate))
#        num_all_data += data_in_group['num_data']
#    print(f'number of all data is {num_all_data}')

import linear_discriminant_analysis
import numpy as np

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
        '--num_ca1_neurons_list',
        nargs='+',
        type=int,
        default=[10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 65, 70, 75, 80, 85, 90, 100], # ニューロン数のリスト
        help='List of number of neurons in CA1 to iterate over'
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
    num_ca1_neurons_list = args.num_ca1_neurons_list

    MER = []
    for num_ca1_neurons in num_ca1_neurons_list:
        print(f"--- Processing with num_ca1_neurons = {num_ca1_neurons} ---")

        filename_parts_list = []
        filename_parts_list.append(f"WT{w_tilde:.4f}".replace('.', 'p')) # 小数点を'p'に変換してファイル名に含める
        filename_parts_list.append(f"NC3N{num_ca3_neurons:04d}") # CA3ニューロン数をファイル名に含める
        filename_parts_list.append(f"NC1N{num_ca1_neurons:04d}") # CA1ニューロン数をファイル名に含める
        filename_parts = f"{'_'.join(filename_parts_list)}"
        print(f"Loading data from file: ../data/spike_counts{filename_parts}.npz")

        try:
            data = np.load('../data/spike_counts' + filename_parts + '.npz')
        except FileNotFoundError:
            print(f"Error: File not found for num_ca1_neurons = {num_ca1_neurons}. Skipping this configuration.")
            continue

        rng = np.random.default_rng(100)

        # remove transient period
        data_without_transient = {}
        data_without_transient['time'] = data['time'][100:]
        data_without_transient['state_vars'] = data['state_vars'][100:]
        data_without_transient['input_seq'] = data['input_seq'][100:]      
        data_without_transient['input_seq'] = linear_discriminant_analysis.rename_input_pattern(data_without_transient['input_seq'])

        num_segments = 10
        d = args.depth
        m = 2
        num_groups = m**(d-1)
        num_all_data = 0
        data_without_transient['group'] = linear_discriminant_analysis.split_data_into_groups(data_without_transient['input_seq'], data_without_transient['state_vars'], d, m)

        error_rate = np.zeros((num_groups, num_segments))
        for i in range(num_groups):
            data_in_group = linear_discriminant_analysis._get_data_within_group(i, data_without_transient['group'], data_without_transient['input_seq'], data_without_transient['state_vars'], m, d)
            print(f'\nLDA for group {i}')
            print(f'num data in group {i} = {data_in_group["num_data"]}')
            data_to_segments = linear_discriminant_analysis.split_data_into_segments(rng, data_in_group['num_data'], num_segments)
            for j in range(num_segments):
                error_rate[i, j] = linear_discriminant_analysis.conduct_lda(data_in_group['oldest_input'], data_in_group['state_vars'], data_to_segments, j, m, d, i)
                print(f"Segment {j} error rate: {error_rate[i, j]}")
            num_all_data += data_in_group['num_data']
        
        print(f"Mean error rate for group {i}: {np.mean(error_rate)}")
        MER.append(np.mean(error_rate))
        print(f'Total number of data for num_ca1_neurons = {num_ca1_neurons} is {num_all_data}\n')

    from matplotlib import pyplot as plt
    plt.plot(num_ca1_neurons_list, MER)
    plt.xlabel('num CA1 neurons')
    plt.ylabel('MER')
    plt.savefig('../figure/MER.png')
