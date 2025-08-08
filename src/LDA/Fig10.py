import linear_discriminant_analysis
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parameters setting for the simulation')
    parser.add_argument(
        '--w_tilde',
        type=float,
        default=[1.25, 1.5, 1.75, 2.0], # デフォルト値
        help='Scaling factor of synaptic weight'
    )
    parser.add_argument(
        '--t_interval_T',
        type=int,
        default=[25, 50, 75, 100, 125, 150, 175, 200], # デフォルト値
        help='Interval of input from CA3'
    )
    parser.add_argument(
        '--seeds',
        type=int,
        default=[0, 1, 2, 3, 4, 5], # デフォルト値
        help='List of random seeds'
    )

    args = parser.parse_args()
    w_tilde_list = args.w_tilde
    t_interval_T_list = args.t_interval_T


    d = 3
    MERs = np.zeros((len(t_interval_T_list), len(w_tilde_list)))
    for t_interval_T_idx, t_interval_T in enumerate(t_interval_T_list):
        for w_tilde_idx, w_tilde in enumerate(w_tilde_list):
            num_segments = 10
            m = 2
            num_groups = m**(d-1)
            error_rate = np.zeros((len(args.seeds), num_groups, num_segments))
            for seed_idx, seed in enumerate(args.seeds):

                filename_parts_list = []
                filename_parts_list.append(f"WT{w_tilde:.4f}".replace('.', 'p')) # 小数点を'p'に変換してファイル名に含める
                filename_parts_list.append(f"INT{t_interval_T:03d}")
                filename_parts_list.append(f"SEED{seed:03d}") # seed
                filename_parts = f"{'_'.join(filename_parts_list)}"
                print(f"Loading data from file: ../data/spike_counts{filename_parts}.npz")

                try:
                    data = np.load('../data/spike_counts' + filename_parts + '.npz')
                except FileNotFoundError:
                    print(f"Error: File not found for t_interval_T = {t_interval_T}, w_tilde = {w_tilde}. Skipping this configuration.")
                    continue

                rng = np.random.default_rng(100)

                # remove transient period
                data_without_transient = {}
                data_without_transient['time'] = data['time'][100:]
                data_without_transient['state_vars'] = data['state_vars'][100:]
                data_without_transient['input_seq'] = data['input_seq'][100:]      
                data_without_transient['input_seq'] = linear_discriminant_analysis.rename_input_pattern(data_without_transient['input_seq'])

                num_all_data = 0
                data_without_transient['group'] = linear_discriminant_analysis.split_data_into_groups(data_without_transient['input_seq'], data_without_transient['state_vars'], d, m)

                for i in range(num_groups):
                    data_in_group = linear_discriminant_analysis._get_data_within_group(i, data_without_transient['group'], data_without_transient['input_seq'], data_without_transient['state_vars'], m, d)
                    print(f'\nLDA for group {i}')
                    print(f'num data in group {i} = {data_in_group["num_data"]}')
                    data_to_segments = linear_discriminant_analysis.split_data_into_segments(rng, data_in_group['num_data'], num_segments)
                    for j in range(num_segments):
                        error_rate[seed_idx, i, j] = linear_discriminant_analysis.conduct_lda(data_in_group['oldest_input'], data_in_group['state_vars'], data_to_segments, j, m, d, i)
                        print(f"Segment {j} error rate: {error_rate[seed_idx, i, j]}")
                    num_all_data += data_in_group['num_data']
                
                print(f"Mean error rate for group {i}: {np.mean(error_rate[seed_idx, i, :])}")
            MERs[t_interval_T_idx, w_tilde_idx] = np.mean(error_rate)



    X, Y = np.meshgrid(t_interval_T_list, w_tilde_list)
    plt.pcolor(X, Y, MERs.T, cmap='viridis')
    plt.colorbar(label='MER') # カラーバーを追加し、ラベルを設定
    plt.xlabel('T') # X軸のラベル
    plt.ylabel('w̃') # Y軸のラベル (チルダはLaTeX形式で記述)
    plt.title('MER Heatmap') # グラフのタイトル
    plt.tight_layout() # レイアウトを自動調整
    plt.savefig('../figure/MER.png')
