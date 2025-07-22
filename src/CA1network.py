import numpy as np
from scipy.integrate import solve_ivp
from pinsky_rinzel_model import PinskyRinzelModel 

class CA1Network:
    """
    CA1ネットワークモデルを実装するクラス。
    複数のPinsky-Rinzelニューロンと、CA3からの入力、シナプス結合を管理する。
    参照論文: Yamaguti et al. (2011) Neural Networks, 24, 43-53.
    """
    def __init__(self, num_ca1_neurons=100, num_ca3_patterns=3,
                 neuron_type="bursting", synapse_type="NMDA", dt=0.05, seed=None):
        """
        CA1ネットワークを初期化する。
        Args:
            num_ca1_neurons (int): CA1ネットワーク内の錐体ニューロンの数 (N).
            num_ca3_patterns (int): CA3からの基本入力パターンの種類数 (M, 論文ではm).
                                     論文ではM個のパターンを保存し、m (<M) 個のパターンを使用する。
                                     ここでは簡略化のため、M=mとする。
            neuron_type (str): 各Pinsky-Rinzelニューロンのタイプ ('bursting' または 'spiking').
            synapse_type (str): シナプスタイプ ('AMPA' または 'NMDA').
            dt (float): シミュレーションのタイムステップ (ms).
            seed (int, optional): 乱数シード。再現性のために設定。
        """
        self.num_ca1_neurons = num_ca1_neurons
        self.num_ca3_patterns = num_ca3_patterns # m (論文の Section 3.1)
        self.dt = dt
        self.rng = np.random.default_rng(seed) # 乱数生成器

        # 各CA1ニューロンのインスタンスを作成
        self.ca1_neurons = []
        for _ in range(self.num_ca1_neurons):
            neuron = PinskyRinzelModel(neuron_type=neuron_type, synapse_type=synapse_type, dt=dt)
            self.ca1_neurons.append(neuron)

        # CA3の基本パターンを初期化 (X^sigma)
        # 論文 Section 3.1: 各要素は1か0をランダムに独立して取る。確率p_fi=0.1で1。
        # N_CA3 は論文に明記されていないが、ここでは簡単のため num_ca1_neurons に合わせるか、
        # あるいは別の値を設定する。ここではnum_ca1_neuronsをN_CA3として使用。
        self.num_ca3_neurons = num_ca1_neurons # 仮にCA3ニューロン数とCA1ニューロン数を同じにする
        self.p_fi = 0.1 # 論文 Section 3.1, probability of taking value 1 is p_fi=0.1
        self.ca3_elementary_patterns = self._generate_ca3_patterns()

        # CA3からCA1へのシナプス結合強度行列 w_ij を初期化
        # 論文 Section 3.1, Eq. (1): w_ij = ~w * sum(x_i^sigma * y_j^sigma)
        # y_j^sigma は CA1ニューロンjが入力パターンsigmaに応答した活動 (uniform dist in [0,1])
        # ~w は adjustable parameter (input strength)
        self.tilde_w = 1.0 # 仮の初期値。シミュレーション時に調整されることが多い
        self.ca3_ca1_weights = self._initialize_ca3_ca1_weights()

        # ネットワーク全体の状態変数をまとめるための初期化
        # 各ニューロンの状態変数を連結した単一の配列として管理する
        self.initial_network_state = self._get_initial_network_state()

    def _generate_ca3_patterns(self):
        """
        CA3の基本空間発火パターン {X^0, ..., X^(M-1)} を生成する。
        X^sigma は N_CA3 個の要素 (0 or 1) からなるベクトル。
        """
        # (num_ca3_patterns, num_ca3_neurons) の形状でバイナリパターンを生成
        patterns = self.rng.binomial(1, self.p_fi, size=(self.num_ca3_patterns, self.num_ca3_neurons))
        return patterns


    def _initialize_ca3_ca1_weights(self):
        """
        CA3からCA1へのシナプス結合強度行列 w_ij を初期化する。
        論文 Eq. (1): w_ij = ~w * sum_{sigma=0}^{M-1} (x_i^sigma * y_j^sigma)
        y_j^sigma は CA1ニューロンjが入力パターンsigmaに応答した活動 (uniform dist in [0,1])。
        このy_j^sigmaは、実際にシミュレーションしないと分からない値なので、
        ここでは論文の記述通りランダムに生成する (See Section 3.1: Each y_j^sigma is a real number randomly chosen from a uniform distribution in [0, 1]).
        """
        # y_j^sigma: (num_ca1_neurons, num_ca3_patterns) の形状
        # 論文の表記に合わせて、y_j^sigmaはニューロンjがパターンsigmaに応答した活動。
        # 通常、i -> j なので、x_i -> y_j。
        # したがって、y_j^sigma のインデックスは (CA1_neuron_idx, CA3_pattern_idx)
        ca1_response_to_ca3_patterns = self.rng.uniform(0, 1, size=(self.num_ca1_neurons, self.num_ca3_patterns))

        # w_ij: (num_ca3_neurons, num_ca1_neurons) の形状
        # i は CA3 ニューロンのインデックス、j は CA1 ニューロンのインデックス
        weights = np.zeros((self.num_ca3_neurons, self.num_ca1_neurons))

        for i in range(self.num_ca3_neurons):
            for j in range(self.num_ca1_neurons):
                # 論文 Eq. (1) に従う
                # x_i^sigma: (num_ca3_patterns,) のベクトルから、特定のsigmaにおけるx_iを取得
                # y_j^sigma: (num_ca3_patterns,) のベクトルから、特定のsigmaにおけるy_jを取得
                # sum_{sigma=0}^{M-1} (x_i^sigma * y_j^sigma)
                weights[i, j] = np.sum(
                    self.ca3_elementary_patterns[:, i] * ca1_response_to_ca3_patterns[j, :]
                )
        
        return self.tilde_w * weights # ~w を乗算

    def _get_initial_network_state(self):
        """
        全てのニューロンの初期状態変数を連結した配列を生成する。
        """
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

    def calculate_ca1_input_current(self, current_ca3_pattern_idx):
        """
        現在のCA3入力パターンに基づいて、各CA1ニューロンへのシナプス入力電流 xi_j^sigma を計算する。
        論文 Section 3.1, Eq. (2): xi_j^sigma = sum_{i=1}^{N_CA3} (w_ij * x_i^sigma)
        Args:
            current_ca3_pattern_idx (int): 現在選択されたCA3入力パターンのインデックス。
        Returns:
            np.array: 各CA1ニューロンへの入力電流 (shape: num_ca1_neurons,).
        """
        if not (0 <= current_ca3_pattern_idx < self.num_ca3_patterns):
            raise ValueError(f"Invalid CA3 pattern index: {current_ca3_pattern_idx}")

        # 現在のCA3パターン (X^sigma)
        current_ca3_pattern = self.ca3_elementary_patterns[current_ca3_pattern_idx, :]

        # 各CA1ニューロン j への入力 xi_j を計算
        # w_ij の形状は (N_CA3, N_CA1)
        # current_ca3_pattern の形状は (N_CA3,)
        # 結果の形状は (N_CA1,) になるように行列積 (転置に注意)
        # sum_{i} (w_ij * x_i^sigma) は、w_ij の j-th 列と x_i^sigma の内積に相当
        # あるいは、(x_i^sigma)^T @ w_ij とすると (1, N_CA3) @ (N_CA3, N_CA1) = (1, N_CA1)
        input_currents = current_ca3_pattern @ self.ca3_ca1_weights # (N_CA3,) @ (N_CA3, N_CA1) -> (N_CA1,)

        return input_currents
    
    def network_equations(self, t, all_state_vars, ca3_input_pattern_idx_at_t):
        """
        ネットワーク全体の微分方程式系。
        Args:
            t (float): 現在の時刻。
            all_state_vars (array): 全てのニューロンの状態変数を連結した配列。
            ca3_input_pattern_idx_at_t (int): 現在の時刻 t でアクティブになるCA3入力パターンのインデックス。
                                            -1 の場合、シナプス入力はゼロとする。
            soma_bias_current_func (callable, optional): 時刻 `t` を引数にとり、その時刻のバイアス電流値を返す関数。
                                                          全ニューロンに同じバイアス電流が印加されると仮定。
        Returns:
            list: 全てのニューロンの各状態変数の時間微分を連結したリスト。
        """
        d_all_state_vars_dt = []
        
        # 現在時刻での各CA1ニューロンへのシナプス入力強度を計算
        if ca3_input_pattern_idx_at_t != -1:
            # calculate_ca1_input_current は xi_j^sigma (N_CA1,) を返す
            per_neuron_synaptic_input = self.calculate_ca1_input_current(ca3_input_pattern_idx_at_t)
        else:
            per_neuron_synaptic_input = np.zeros(self.num_ca1_neurons)


        # 各ニューロンの微分方程式を計算
        num_vars_per_neuron = len(self.ca1_neurons[0].initial_conditions) # 10変数
        for i, neuron in enumerate(self.ca1_neurons):
            # 各ニューロンの状態変数をall_state_varsから抽出
            neuron_state_vars = all_state_vars[i * num_vars_per_neuron : (i + 1) * num_vars_per_neuron]
            
            # 各ニューロンへのシナプス入力
            synaptic_input_for_neuron = per_neuron_synaptic_input[i]

            # PinskyRinzelModelのequationsメソッドを呼び出す
            # Is には common_bias_current を渡す
            d_neuron_state_vars_dt = neuron.equations(
                t, 
                neuron_state_vars, 
                input_signal_val=synaptic_input_for_neuron,
            )
            d_all_state_vars_dt.extend(d_neuron_state_vars_dt)
            
        return d_all_state_vars_dt


    def simulate_network(self, t_span, ca3_input_sequence, ca3_input_interval_T,
                         ca3_input_duration_delta):
        """
        CA1ネットワーク全体のシミュレーションを実行する。
        Args:
            t_span (tuple): (t_start, t_end) シミュレーション期間 (ms)。
            ca3_input_sequence (list): 時刻 k*T で印加されるCA3入力パターンのインデックスのシーケンス。
                                       例: [0, 1, 0, 2, ...]
            ca3_input_interval_T (float): CA3入力パターンが切り替わる時間間隔 T (ms)。
            ca3_input_duration_delta (float): CA3入力がアクティブな持続時間 delta (ms)。
            soma_bias_current_func (callable, optional): 時刻 `t` を引数にとり、その時刻のバイアス電流値を返す関数。
                                                          全ニューロンに同じバイアス電流が印加されると仮定。
        Returns:
            scipy.integrate.OdeResult: シミュレーション結果。
        """
        # 論文の入力ロジック (Section 3.1, Eq. (3)) に基づく外部入力関数
        def network_input_func(t):
            k = int(t / ca3_input_interval_T) # 現在のTインターバルのインデックス
            
            if k < len(ca3_input_sequence):
                # 入力期間中のみアクティブ
                if (t >= k * ca3_input_interval_T) and (t < k * ca3_input_interval_T + ca3_input_duration_delta):
                    return ca3_input_sequence[k] # 現在のCA3パターンインデックスを返す
            return -1 # 入力がない期間は-1を返す (network_equationsでゼロ入力として処理)

        # ネットワーク全体の状態変数の初期値
        initial_y0 = self.initial_network_state

        t_eval = np.arange(t_span[0], t_span[1], self.dt)

        # solve_ivp に渡すラッパー関数
        def ode_wrapper(t, all_state_vars):
            current_ca3_pattern_idx = network_input_func(t)
            return self.network_equations(t, all_state_vars, current_ca3_pattern_idx)

        try:
            sol = solve_ivp(ode_wrapper, t_span, initial_y0, method='RK45', t_eval=t_eval, rtol=1e-5, atol=1e-8)
        except ValueError as e:
            print(f"Error during network simulation: {e}")
            print("This might be due to numerical instability. Consider adjusting initial conditions, dt, or solver tolerances.")
            return None

        return sol

    def extract_neuron_data(self, sol, neuron_idx):
        """
        シミュレーション結果から特定のニューロンのデータを抽出する。
        Args:
            sol (scipy.integrate.OdeResult): simulate_networkの戻り値。
            neuron_idx (int): 抽出したいニューロンのインデックス。
        Returns:
            dict: そのニューロンの Vs, Vd, Ca, h, n, s, c, q, Ga, Gn の時系列データ。
        """
        num_vars_per_neuron = len(self.ca1_neurons[0].initial_conditions)
        start_idx = neuron_idx * num_vars_per_neuron
        end_idx = start_idx + num_vars_per_neuron

        # 状態変数の順番: Vs, Vd, Ca, m, h, n, s, c, q, Ga, Gn
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


if __name__ == '__main__':
    # 論文のFigure 4の設定に合わせたパラメータ
    num_ca1 = 100 # CA1ニューロン数 (論文 Figure 9で100まで)
    num_ca3_patterns = 3 # m=3 (論文 Figure 3, 4, 6)
    t_interval_T = 100.0 # T=100ms (論文 Figure 4) 
    duration_delta = 5.0 # delta=5ms (論文 Section 3.1) 
    sim_dt = 0.05 # シミュレーションタイムステップ 
    
    # 論文 Figure 4 は bursting type neurons with NMDA synapses を使用 
    neuron_type = "bursting"
    synapse_type = "BOTH"

    # シミュレーション時間: 短めのシーケンスで動作確認
    # 例えば、3つのパターンを5回繰り返すような入力シーケンス
    # 論文 Section 4.1 には L=10000 (10000 T intervals) のデータが記録されているが、まずは短く
    sequence_length = 10 # T間隔の数 (100ms * 10 = 1000ms = 1秒)
    t_span_network = (0, t_interval_T * sequence_length) # 0ms から 1000ms

    # CA3入力シーケンスの例 (ランダムに生成)
    # 論文 Section 3.1: "One of the m(<M) elementary input patterns defined above is randomly chosen every T ms." 
    rng = np.random.default_rng(42) # シーケンス再現のための乱数シード
    ca3_input_sequence = rng.integers(0, num_ca3_patterns, size=sequence_length).tolist()
    print(f"Generated CA3 input sequence: {ca3_input_sequence}")


    print(f"Initializing CA1 Network with {num_ca1} neurons...")
    ca1_network = CA1Network(
        num_ca1_neurons=num_ca1,
        num_ca3_patterns=num_ca3_patterns,
        neuron_type=neuron_type,
        synapse_type=synapse_type,
        dt=sim_dt,
        seed=42 # ネットワークの乱数シード
    )
    print("CA1 Network Initialized.")

    print(f"Starting network simulation for {t_span_network[1]} ms...")
    network_sol = ca1_network.simulate_network(
        t_span=t_span_network,
        ca3_input_sequence=ca3_input_sequence,
        ca3_input_interval_T=t_interval_T,
        ca3_input_duration_delta=duration_delta,
    )
    print("Network simulation completed.")

    if network_sol is not None:
        # 結果のプロット (最初の数ニューロンのVs)
        plt.figure(figsize=(12, 8))
        num_neurons_to_plot = min(5, num_ca1) # 最初の5ニューロンをプロット

        for i in range(num_neurons_to_plot):
            neuron_data = ca1_network.extract_neuron_data(network_sol, i)
            plt.plot(network_sol.t, neuron_data['Vs'], label=f'Neuron {i} Vs')

        # 入力パターンの表示
        y_min, y_max = plt.ylim() # 現在のy軸の範囲を取得
        for k_idx, pattern_idx in enumerate(ca3_input_sequence):
            start_time = k_idx * t_interval_T
            end_time = start_time + duration_delta
            # 入力期間を帯で表示
            plt.axvspan(start_time, end_time, color=f'C{pattern_idx}', alpha=0.1, label=f'Input {pattern_idx}' if k_idx == 0 else "")
            # 入力パターン番号をテキストで表示
            plt.text(start_time + duration_delta/2, y_max * 0.9, str(pattern_idx), 
                     horizontalalignment='center', verticalalignment='top', fontsize=10, color='gray')


        plt.title(f'CA1 Network Simulation (N={num_ca1}, m={num_ca3_patterns}, T={t_interval_T}ms, delta={duration_delta}ms)')
        plt.xlabel('Time (msec)')
        plt.ylabel('Soma Membrane Potential (mV)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("ca1_network_simulation.png")
        # plt.show()
    else:
        print("Network simulation failed. Plotting skipped.")
