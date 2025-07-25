import numpy as np
from scipy.integrate import odeint, solve_ivp
from pinsky_rinzel_model import PinskyRinzelModel 

class CA1Network:
    def __init__(self, num_ca1_neurons=100, num_ca3_patterns=100, num_ca3_patterns_input=3,
                 neuron_type="bursting", synapse_type="BOTH", w_tilde=0.01, dt=0.05, seed=None):
        self.num_ca1_neurons        = num_ca1_neurons
        self.num_ca3_patterns       = num_ca3_patterns       # M (論文の Section 3.1)
        self.num_ca3_patterns_input = num_ca3_patterns_input # m (論文の Section 3.1)
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        self.ca1_neurons = []
        for _ in range(self.num_ca1_neurons):
            neuron = PinskyRinzelModel(neuron_type=neuron_type, synapse_type=synapse_type, dt=dt)
            self.ca1_neurons.append(neuron)
        self.num_ca3_neurons = num_ca1_neurons # 仮にCA3ニューロン数とCA1ニューロン数を同じにする
        self.p_fi = 0.1 # 論文 Section 3.1, probability of taking value 1 is p_fi=0.1
        self.ca3_elementary_patterns = self._generate_ca3_patterns()
        self.tilde_w = w_tilde # scaling factor for synaptic weight
        self.ca3_ca1_weights = self._initialize_ca3_ca1_weights()
        self.initial_network_state = self._get_initial_network_state()

    def _generate_ca3_patterns(self):
        patterns = self.rng.binomial(1, self.p_fi, size=(self.num_ca3_patterns, self.num_ca3_neurons))
        return patterns


    def _initialize_ca3_ca1_weights(self):
        ca1_response_to_ca3_patterns = self.rng.uniform(0, 1, size=(self.num_ca1_neurons, self.num_ca3_patterns))

        weights = np.zeros((self.num_ca3_neurons, self.num_ca1_neurons))

        for pre_idx in range(self.num_ca3_neurons):
            for pst_idx in range(self.num_ca1_neurons):
                weights[pre_idx, pst_idx] = np.sum(
                    self.ca3_elementary_patterns[:, pre_idx] * ca1_response_to_ca3_patterns[pst_idx, :]
                )
        
        return self.tilde_w * weights # multiply scaling factor

    def _get_initial_network_state(self):
        all_y0 = []
        for neuron in self.ca1_neurons:
            y0_list = [
                neuron.initial_conditions['Vs'],
                neuron.initial_conditions['Vd'],
                neuron.initial_conditions['Ca'],
                neuron.initial_conditions['m'],
                neuron.initial_conditions['h'],
                neuron.initial_conditions['n'],
                neuron.initial_conditions['s'],
                neuron.initial_conditions['c'],
                neuron.initial_conditions['q'],
                neuron.initial_conditions['Ga'],
                neuron.initial_conditions['Gn']
            ]
            all_y0.extend(y0_list)
        return np.array(all_y0)

    def calculate_ca1_input_spike(self, current_ca3_pattern_idx):
        if not (0 <= current_ca3_pattern_idx < self.num_ca3_patterns):
            raise ValueError(f"Invalid CA3 pattern index: {current_ca3_pattern_idx}")

        current_ca3_pattern = self.ca3_elementary_patterns[current_ca3_pattern_idx, :]
        input_currents = current_ca3_pattern @ self.ca3_ca1_weights # (N_CA3,) @ (N_CA3, N_CA1) -> (N_CA1,)

        return input_currents
    
    def network_equations(self, t, all_state_vars, ca3_input_pattern_idx_at_t):
        d_all_state_vars_dt = []
        
        if ca3_input_pattern_idx_at_t != -1: # while stimuli from CA3 are delivered
            per_neuron_synaptic_input = self.calculate_ca1_input_spike(ca3_input_pattern_idx_at_t)
        else: # while stimuli from CA3 is suppressed
            per_neuron_synaptic_input = np.zeros(self.num_ca1_neurons)


        num_vars_per_neuron = len(self.ca1_neurons[0].initial_conditions) # 11変数
        for i, neuron in enumerate(self.ca1_neurons):
            neuron_state_vars = all_state_vars[i * num_vars_per_neuron : (i + 1) * num_vars_per_neuron]
            
            synaptic_input_for_neuron = per_neuron_synaptic_input[i]
            d_neuron_state_vars_dt = neuron.equations(
                t, 
                neuron_state_vars, 
                input_signal_val=synaptic_input_for_neuron,
            )
            d_all_state_vars_dt.extend(d_neuron_state_vars_dt)
            
        return d_all_state_vars_dt


    def simulate_network(self, t_span, ca3_input_sequence, ca3_input_interval_T,
                         ca3_input_duration_delta):
        def network_input_func(t):
            k = int(t / ca3_input_interval_T) # 現在のTインターバルのインデックス
            
            if k < len(ca3_input_sequence):
                if (t >= k * ca3_input_interval_T) and (t < k * ca3_input_interval_T + ca3_input_duration_delta):
                    return ca3_input_sequence[k] # 現在のCA3パターンインデックスを返す
            return -1 # 入力がない期間は-1を返す (network_equationsでゼロ入力として処理)

        initial_y0 = self.initial_network_state

        t_eval = np.arange(t_span[0], t_span[1], self.dt)

        def ode_wrapper(t, all_state_vars):
            current_ca3_pattern_idx = network_input_func(t)
            return self.network_equations(t, all_state_vars, current_ca3_pattern_idx)

        #def ode_wrapper(t, all_state_vars, network_input_func):
        #   current_ca3_pattern_idx = network_input_func(t)
        #   return self.network_equations(t, all_state_vars, current_ca3_pattern_idx)
       
        try:
            sol = solve_ivp(ode_wrapper, t_span, initial_y0, method='RK45', t_eval=t_eval, rtol=1e-5, atol=1e-8)
            return sol
            #sol = odeint(ode_wrapper, initial_y0, t_eval, rtol=1e-5, atol=1e-8, full_output=False)
            
            #class OdeintResult:
            #    def __init__(self, t, y):
            #        self.t = t
            #        self.y = y.T
            #sol_obj = OdeintResult(t_eval, sol)
            #return sol_obj
            # odeint の呼び出し
            #sol_raw = odeint(
            #    ode_wrapper, 
            #    initial_y0, 
            #    t_eval, 
            #    args=(network_input_func,),
            #    rtol=1e-5, 
            #    atol=1e-8,
            #    full_output=False 
            #)
            
            # odeint の結果を solve_ivp の結果形式に似せてラップ
            # odeint の sol_raw は (n_steps, n_vars) なので、
            # solve_ivp の y 形式 (n_vars, n_steps) にするために転置する
            #class OdeResultMimic:
            #    def __init__(self, t, y_transposed):
            #        self.t = t
            #        self.y = y_transposed 
            #
            #sol_obj = OdeResultMimic(t_eval, sol_raw.T) # sol_raw.T で転置
            #return sol_obj
        except ValueError as e:
            print(f"Error during network simulation: {e}")
            print("This might be due to numerical instability. Consider adjusting initial conditions, dt, or solver tolerances.")
            return None

        return sol

    def extract_neuron_data(self, sol, neuron_idx):
        num_vars_per_neuron = len(self.ca1_neurons[0].initial_conditions)
        start_idx = neuron_idx * num_vars_per_neuron
        end_idx = start_idx + num_vars_per_neuron

        data = {
            'Vs': sol.y[start_idx, :],
            'Vd': sol.y[start_idx + 1, :],
            'Ca': sol.y[start_idx + 2, :],
            'm':  sol.y[start_idx + 3, :],
            'h':  sol.y[start_idx + 4, :],
            'n':  sol.y[start_idx + 5, :],
            's':  sol.y[start_idx + 6, :],
            'c':  sol.y[start_idx + 7, :],
            'q':  sol.y[start_idx + 8, :],
            'Ga': sol.y[start_idx + 9, :],
            'Gn': sol.y[start_idx + 10, :]
        }
        return data

    def get_all_soma_membrane_potentials(self, sol):
        if sol is None:
            print("Simulation result (sol) is None. Cannot extract Vs data.")
            return None

        num_vars_per_neuron = len(self.ca1_neurons[0].initial_conditions)
        
        all_vs_data = []
        for i in range(self.num_ca1_neurons):
            vs_idx = i * num_vars_per_neuron # 各ニューロンのVsのインデックス
            all_vs_data.append(sol.y[vs_idx, :])
        
        return np.array(all_vs_data)

    def get_averaged_soma_potentials(self, sol, t_interval_T):
        if sol is None:
            print("Simulation result (sol) is None. Cannot average Vs data.")
            return None
        
        if all_vs_matrix is None:
            return None

        num_time_steps = sol.t.shape[0]
        num_intervals = int(num_time_steps * self.dt / t_interval_T)
        
        averaged_vs_matrix = np.zeros((self.num_ca1_neurons, num_intervals))
        
        for k in range(num_intervals):
            start_time_idx = int((k * t_interval_T) / self.dt)
            end_time_idx = int(((k + 1) * t_interval_T) / self.dt)
            
            # 各ニューロンについて、この時間区間の膜電位の平均を計算
            # np.meanは axis=1 で時間方向の平均を取る
            # all_vs_matrix[neuron_idx, time_idx]
            averaged_vs_matrix[:, k] = np.mean(all_vs_matrix[:, start_time_idx:end_time_idx], axis=1)
            
        return averaged_vs_matrix

    def get_spike_counts(self, sol, t_interval_T):
        if sol is None:
            print("Simulation result (sol) is None. Cannot count spikes.")
            return None
        
        if all_vs_matrix is None:
            return None

        num_time_steps = sol.t.shape[0]
        num_intervals = int(num_time_steps * self.dt / t_interval_T) 
        
        spike_counts_matrix = np.zeros((self.num_ca1_neurons, num_intervals))
        
        for k in range(num_intervals): 
            start_time_idx = int((k * t_interval_T) / self.dt)
            end_time_idx = int(((k + 1) * t_interval_T) / self.dt)
            
            # 各ニューロンについて、この時間区間のスパイク数をカウント
            for neuron_idx in range(self.num_ca1_neurons):
                vs_trace_in_interval = all_vs_matrix[neuron_idx, start_time_idx:end_time_idx]
                # PinskyRinzelModel インスタンスの count_spikes_in_trace メソッドを使用
                spike_counts_matrix[neuron_idx, k] = self.ca1_neurons[neuron_idx].count_spikes_in_trace(vs_trace_in_interval)
            
        return spike_counts_matrix


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parameters setting for the simulation')
    parser.add_argument(
        '--w_tilde',
        type=float,
        default=0.01, # デフォルト値
        help='Scaling factor of synaptic weight'
    )
    args = parser.parse_args()
    w_tilde = args.w_tilde

    print(f"Scaling fact or of synaptic weight : {w_tilde}")
    filename_parts_list = []
    filename_parts_list.append(f"WT{w_tilde:.3f}".replace('.', 'p')) # 小数点を'p'に変換してファイル名に含める
    filename = ""
    if filename_parts_list:
        filename_parts = f"{'_'.join(filename_parts_list)}"
        print(filename_parts)
    else:
        filename_parts = f""
        print(filename_parts)


    # Parameters
    num_ca1 = 100              # CA1ニューロン数 (論文 Figure 9で100まで)
    num_ca3_patterns = 20      # M=100 (論文では具体的な値が指定されていない)
    num_ca3_patterns_input = 3 # m=3 (主成分空間の次元と一致させる)
    t_interval_T = 100.0       # T=100ms (論文 Figure 4) 
    duration_delta = 5.0       # delta=5ms (論文 Section 3.1) 
    sim_dt = 0.05              # シミュレーションタイムステップ 
    neuron_type = "bursting"
    synapse_type = "BOTH"
    sequence_length = 1000     # T間隔の数
    t_span_network = (0, t_interval_T * sequence_length) # 0ms から 1000ms
    rng = np.random.default_rng(42)
    selected_numbers = rng.choice(range(num_ca3_patterns), size=num_ca3_patterns_input, replace=False)
    #print(selected_numbers)
    ca3_input_sequence = rng.integers(0, num_ca3_patterns_input, size=sequence_length).tolist()
    for input_idx in range(sequence_length):
        ca3_input_sequence[input_idx] = selected_numbers[ca3_input_sequence[input_idx]]
    #print(f"Generated CA3 input sequence: {ca3_input_sequence}")

    print(f"Initializing CA1 Network with {num_ca1} neurons...")
    ca1_network = CA1Network(
        num_ca1_neurons=num_ca1,
        num_ca3_patterns=num_ca3_patterns,
        num_ca3_patterns_input=num_ca3_patterns_input,
        neuron_type=neuron_type,
        synapse_type=synapse_type,
        w_tilde = w_tilde,
        dt=sim_dt,
        seed=42
    )
    print("CA1 Network Initialized.")

    print(f"Starting network simulation for {t_span_network[1]} ms...")
    import time
    start = time.time()
    network_sol = ca1_network.simulate_network(
        t_span=t_span_network,
        ca3_input_sequence=ca3_input_sequence,
        ca3_input_interval_T=t_interval_T,
        ca3_input_duration_delta=duration_delta,
    )
    end = time.time()
    print("Network simulation completed:end - start")

    if network_sol is not None:
        from matplotlib import pyplot as plt
        plt.figure(figsize=(12, 8))
        num_neurons_to_plot = min(5, num_ca1)

        # transform data
        all_vs_matrix = ca1_network.get_all_soma_membrane_potentials(network_sol)
        averaged_vs_matrix = ca1_network.get_averaged_soma_potentials(network_sol, t_interval_T)
        spike_counts_matrix = ca1_network.get_spike_counts(network_sol, t_interval_T)
        print(f"Shape of continuous Vs matrix: {all_vs_matrix.shape}")
        print(f"Shape of averaged Vs matrix: {averaged_vs_matrix.shape}")
        print(f"Shape of spike counts matrix: {spike_counts_matrix.shape}")


        # figure 1
        for i in range(num_neurons_to_plot):
            neuron_data = ca1_network.extract_neuron_data(network_sol, i)
            plt.plot(network_sol.t, neuron_data['Vs'], label=f'Neuron {i} Vs')

        y_min, y_max = plt.ylim()
        for k_idx, pattern_idx in enumerate(ca3_input_sequence):
            start_time = k_idx * t_interval_T
            end_time = start_time + duration_delta
            plt.axvspan(start_time, end_time, color=f'C{pattern_idx}', alpha=0.1, label=f'Input {pattern_idx}' if k_idx == 0 else "")
            plt.text(start_time + duration_delta/2, y_max * 0.9, str(pattern_idx), 
                     horizontalalignment='center', verticalalignment='top', fontsize=10, color='gray')

        plt.title(f'CA1 Network Simulation (N={num_ca1}, m={num_ca3_patterns}, T={t_interval_T}ms, delta={duration_delta}ms)')
        plt.xlabel('Time (msec)')
        plt.ylabel('Soma Membrane Potential (mV)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        #plt.savefig("../figure/membranepotential.png")
        plt.savefig("../figure/membranepotential_" + filename_parts + ".png")
        plt.close()


        # figure 2
        # --- 3つのヒートマップを並べてプロット ---
        fig, axes = plt.subplots(3, 1, figsize=(15, 18), sharex=False) # sharex=False for different x-axis scales

        # Plot 1: Continuous Time Soma Potential Heatmap
        ax0 = axes[0]
        im0 = ax0.imshow(all_vs_matrix, aspect='auto', cmap='hot',
                         extent=[network_sol.t.min(), network_sol.t.max(), ca1_network.num_ca1_neurons - 0.5, -0.5],
                         origin='upper', vmin=-80, vmax=40)
        fig.colorbar(im0, ax=ax0, label='Soma Membrane Potential (mV)')
        ax0.set_title('1. Continuous Time Soma Potential Heatmap')
        ax0.set_xlabel('Time (msec)')
        ax0.set_ylabel('Neuron Index')

        # Input pattern overlay for continuous plot
        y_min_ax0, y_max_ax0 = ax0.get_ylim()
        for k_idx, pattern_idx in enumerate(ca3_input_sequence):
            start_time = k_idx * t_interval_T
            end_time = start_time + duration_delta
            ax0.axvspan(start_time, end_time, color=f'C{pattern_idx}', alpha=0.05)
            ax0.text(start_time + duration_delta/2, y_max_ax0 * 0.95, str(pattern_idx), 
                     horizontalalignment='center', verticalalignment='top', fontsize=8, color='white',
                     bbox=dict(facecolor=f'C{pattern_idx}', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))


        # Plot 2: Averaged Soma Potential Heatmap
        ax1 = axes[1]
        interval_times = np.arange(averaged_vs_matrix.shape[1]) * t_interval_T + t_interval_T / 2
        im1 = ax1.imshow(averaged_vs_matrix, aspect='auto', cmap='hot',
                         extent=[interval_times.min() - t_interval_T/2, interval_times.max() + t_interval_T/2, 
                                 ca1_network.num_ca1_neurons - 0.5, -0.5],
                         origin='upper', vmin=-80, vmax=40)
        fig.colorbar(im1, ax=ax1, label='Averaged Soma Membrane Potential (mV)')
        ax1.set_title('2. Averaged Soma Potential Heatmap (Discrete Intervals)')
        ax1.set_xlabel('Interval Time (msec)')
        ax1.set_ylabel('Neuron Index')

        # Input pattern overlay for averaged plot
        y_min_ax1, y_max_ax1 = ax1.get_ylim()
        for k_idx, pattern_idx in enumerate(ca3_input_sequence):
            start_interval_time = k_idx * t_interval_T
            end_interval_time = (k_idx + 1) * t_interval_T
            ax1.axvspan(start_interval_time, end_interval_time, color=f'C{pattern_idx}', alpha=0.05)
            ax1.text(start_interval_time + t_interval_T/2, y_max_ax1 * 0.95, str(pattern_idx),
                     horizontalalignment='center', verticalalignment='top', fontsize=8, color='white',
                     bbox=dict(facecolor=f'C{pattern_idx}', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))


        # Plot 3: Spike Count Heatmap
        ax2 = axes[2]
        # vmax はスパイク数の最大値に基づいて調整、0の場合は1とする
        max_spikes = np.max(spike_counts_matrix) if np.max(spike_counts_matrix) > 0 else 1 
        im2 = ax2.imshow(spike_counts_matrix, aspect='auto', cmap='hot', # スパイク数は整数なのでviridisなどが見やすい
                         extent=[interval_times.min() - t_interval_T/2, interval_times.max() + t_interval_T/2, 
                                 ca1_network.num_ca1_neurons - 0.5, -0.5],
                         origin='upper', vmin=0, vmax=max_spikes)
        fig.colorbar(im2, ax=ax2, label='Number of Spikes')
        ax2.set_title('3. Spike Count Heatmap (Discrete Intervals)')
        ax2.set_xlabel('Interval Time (msec)')
        ax2.set_ylabel('Neuron Index')

        # Input pattern overlay for spike count plot
        y_min_ax2, y_max_ax2 = ax2.get_ylim()
        for k_idx, pattern_idx in enumerate(ca3_input_sequence):
            start_interval_time = k_idx * t_interval_T
            end_interval_time = (k_idx + 1) * t_interval_T
            ax2.axvspan(start_interval_time, end_interval_time, color=f'C{pattern_idx}', alpha=0.05)
            ax2.text(start_interval_time + t_interval_T/2, y_max_ax2 * 0.95, str(pattern_idx),
                     horizontalalignment='center', verticalalignment='top', fontsize=8, color='white',
                     bbox=dict(facecolor=f'C{pattern_idx}', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

        plt.tight_layout()
        #plt.savefig("../figure/statevars.png")
        plt.savefig("../figure/statevars_" + filename_parts + ".png")

        # figure 3
        from sklearn.decomposition import PCA
        print("\nStarting PCA and 3D plotting (no color coding)...")
        fig_pca = plt.figure(figsize=(10, 8))
        ax_pca = fig_pca.add_subplot(111, projection='3d')

        X_pca = averaged_vs_matrix.T 
        print(f"Shape of data for PCA (intervals x neurons): {X_pca.shape}")
        pca = PCA(n_components=3)
        principal_components = pca.fit_transform(X_pca)
        print(f"Shape of principal components: {principal_components.shape}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Cumulative explained variance: {np.sum(pca.explained_variance_ratio_)}")
        for i in range(sequence_length):
            if i > 150:
                if ca3_input_sequence[i] == selected_numbers[0]:
                    ax_pca.scatter(principal_components[i, 0], principal_components[i, 1], principal_components[i, 2], color="r", s=20, alpha=0.7)
                elif ca3_input_sequence[i] == selected_numbers[1]:
                    ax_pca.scatter(principal_components[i, 0], principal_components[i, 1], principal_components[i, 2], color="b", s=20, alpha=0.7)
                elif ca3_input_sequence[i] == selected_numbers[2]:
                    ax_pca.scatter(principal_components[i, 0], principal_components[i, 1], principal_components[i, 2], color="y", s=20, alpha=0.7)
        ax_pca.set_box_aspect((1, 1, 0.5)) 
        ax_pca.set_xlabel('Principal Component 1 (u1)')
        ax_pca.set_ylabel('Principal Component 2 (u2)')
        ax_pca.set_zlabel('Principal Component 3 (u3)')
        ax_pca.set_title(f'PCA of Averaged Soma Potentials (m={ca1_network.num_ca3_patterns_input})')
        ax_pca.grid(True)
        plt.tight_layout()
        #plt.savefig("../figure/PCA_depth1.png")
        plt.savefig("../figure/PCA_vs_depth1_" + filename_parts + ".png")
        print("PCA 3D plot saved successfully.")

        
        # figure 4
        print("\nStarting PCA and 3D plotting (no color coding)...")
        my_colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'orange', 'brown']
        fig_pca = plt.figure(figsize=(10, 8))
        ax_pca = fig_pca.add_subplot(111, projection='3d')

        X_pca = averaged_vs_matrix.T 
        print(f"Shape of data for PCA (intervals x neurons): {X_pca.shape}")
        pca = PCA(n_components=3)
        principal_components = pca.fit_transform(X_pca)
        print(f"Shape of principal components: {principal_components.shape}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Cumulative explained variance: {np.sum(pca.explained_variance_ratio_)}")
        for i in range(sequence_length):
            if i > 150:
                for last_input_idx1 in range(3):
                    for last_input_idx2 in range(3):
                        if ca3_input_sequence[i] == selected_numbers[last_input_idx1]:
                            if ca3_input_sequence[i-1] == selected_numbers[last_input_idx2]:
                                ax_pca.scatter(principal_components[i, 0], 
                                               principal_components[i, 1], 
                                               principal_components[i, 2], 
                                               color=my_colors[last_input_idx1 * 3 + last_input_idx2],
                                               s=20, alpha=0.7)
        ax_pca.set_box_aspect((1, 1, 0.5)) 
        ax_pca.set_xlabel('Principal Component 1 (u1)')
        ax_pca.set_ylabel('Principal Component 2 (u2)')
        ax_pca.set_zlabel('Principal Component 3 (u3)')
        ax_pca.set_title(f'PCA of Averaged Soma Potentials (m={ca1_network.num_ca3_patterns_input})')
        ax_pca.grid(True)
        plt.tight_layout()
        plt.savefig("../figure/PCA_vs_depth2_" + filename_parts + ".png")
        print("PCA 3D plot saved successfully.")

        # figure 5
        from sklearn.decomposition import PCA
        print("\nStarting PCA and 3D plotting (no color coding)...")
        fig_pca = plt.figure(figsize=(10, 8))
        ax_pca = fig_pca.add_subplot(111, projection='3d')

        X_pca = spike_counts_matrix.T 
        print(f"Shape of data for PCA (intervals x neurons): {X_pca.shape}")
        pca = PCA(n_components=3)
        principal_components = pca.fit_transform(X_pca)
        print(f"Shape of principal components: {principal_components.shape}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Cumulative explained variance: {np.sum(pca.explained_variance_ratio_)}")
        for i in range(sequence_length):
            if i > 150:
                if ca3_input_sequence[i] == selected_numbers[0]:
                    ax_pca.scatter(principal_components[i, 0], principal_components[i, 1], principal_components[i, 2], color="r", s=20, alpha=0.7)
                elif ca3_input_sequence[i] == selected_numbers[1]:
                    ax_pca.scatter(principal_components[i, 0], principal_components[i, 1], principal_components[i, 2], color="b", s=20, alpha=0.7)
                elif ca3_input_sequence[i] == selected_numbers[2]:
                    ax_pca.scatter(principal_components[i, 0], principal_components[i, 1], principal_components[i, 2], color="y", s=20, alpha=0.7)
        ax_pca.set_box_aspect((1, 1, 0.5)) 
        ax_pca.set_xlabel('Principal Component 1 (u1)')
        ax_pca.set_ylabel('Principal Component 2 (u2)')
        ax_pca.set_zlabel('Principal Component 3 (u3)')
        ax_pca.set_title(f'PCA of Averaged Soma Potentials (m={ca1_network.num_ca3_patterns_input})')
        ax_pca.grid(True)
        plt.tight_layout()
        #plt.savefig("../figure/PCA_depth1.png")
        plt.savefig("../figure/PCA_spikes_depth1_" + filename_parts + ".png")
        print("PCA 3D plot saved successfully.")

        
        # figure 6
        print("\nStarting PCA and 3D plotting (no color coding)...")
        my_colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'orange', 'brown']
        fig_pca = plt.figure(figsize=(10, 8))
        ax_pca = fig_pca.add_subplot(111, projection='3d')

        X_pca = spike_counts_matrix.T 
        print(f"Shape of data for PCA (intervals x neurons): {X_pca.shape}")
        pca = PCA(n_components=3)
        principal_components = pca.fit_transform(X_pca)
        print(f"Shape of principal components: {principal_components.shape}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Cumulative explained variance: {np.sum(pca.explained_variance_ratio_)}")
        for i in range(sequence_length):
            if i > 150:
                for last_input_idx1 in range(3):
                    for last_input_idx2 in range(3):
                        if ca3_input_sequence[i] == selected_numbers[last_input_idx1]:
                            if ca3_input_sequence[i-1] == selected_numbers[last_input_idx2]:
                                ax_pca.scatter(principal_components[i, 0], 
                                               principal_components[i, 1], 
                                               principal_components[i, 2], 
                                               color=my_colors[last_input_idx1 * 3 + last_input_idx2],
                                               s=20, alpha=0.7)
        ax_pca.set_box_aspect((1, 1, 0.5)) 
        ax_pca.set_xlabel('Principal Component 1 (u1)')
        ax_pca.set_ylabel('Principal Component 2 (u2)')
        ax_pca.set_zlabel('Principal Component 3 (u3)')
        ax_pca.set_title(f'PCA of Averaged Soma Potentials (m={ca1_network.num_ca3_patterns_input})')
        ax_pca.grid(True)
        plt.tight_layout()
        plt.savefig("../figure/PCA_spikes_depth2_" + filename_parts + ".png")
        print("PCA 3D plot saved successfully.")
        
        # store data to npy files
        if all_vs_matrix is not None:
            output_filename = "../data/all_vs_matrix" + filename_parts + ".npy"
            np.savez_compressed(output_filename_npz,
                                 time=network_sol.t,
                                 soma_potentials=all_vs_matrix)
            print(f"All Vs matrix saved to {output_filename}")
    else:
        print("Network simulation failed. Plotting skipped.")
