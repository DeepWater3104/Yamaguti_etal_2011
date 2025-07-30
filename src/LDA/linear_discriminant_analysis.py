from sklearn import LinearDiscriminantAnalysis

class  Linear_Discriminant_Analysis:
    def __init__(self, ca3_input_sequence, sequence_transient_length, sequence_analysis_length, ca1_network, network_sol):
        self.ca3_input_sequence = ca3_input_sequence
        self.sequence_analysis_length = sequence_analysis_length
        self.sequence_transient_length = sequence_transient_length
        self.ca3_elementary_pattern = ca1_newtork.ca3_elemehntary_pattern
        self.state_vector = newtork_sol.spike_counts_matrix
        self.d_input_history = self._get_past_d_input()
        self.lda = LinearDiscriminantAnalysis(n_component=1)
        self.z = self._get_z_value()

    def _get_past_d_input(self):
        d_input_history = np.zeros((self.sequence_analysis_length, self.d))
        for i in range(self.sequence_analysis_length):
            d_input_history[i, :] = ca3_input_sequence[sequence_transient_length+i-1:sequence_transient_length+i]

    def separate_dataset(self):
        '''
        Separate data into 10 segments
        '''
        self.data_segments = 
        self.training_data =
        self.test_data     =

    def _get_z_value(self):
        z = self.lda.fit(self.state_veector, self.d_input_history[:, -1]).transform(X) # only applicable for cases d=1

    def _inference_class(self):

if __name__ == '__main__':
    '''
    Parameters
    '''
    num_ca1_neurons = 100      # CA1ニューロン数 (論文 Figure 9で100まで)
    num_ca3_patterns = 20      # M=100 (論文では具体的な値が指定されていない)
    num_ca3_patterns_input = 2 # m=2
    t_interval_T = 100.0       # T=100ms (論文 Figure 4) 
    duration_delta = 5.0       # delta=5ms (論文 Section 3.1) 
    sim_dt = 0.05              # シミュレーションタイムステップ 
    neuron_type  = ["bursting" for _ in range(num_ca1_neurons)]
    synapse_type = ["BOTH"     for _ in range(num_ca1_neurons)]
    sequence_transient_length = 100
    sequence_analysis_length = 10000
    sequence_length = sequence_analysis_length+transient_length    # T間隔の数
    t_span_network = (0, t_interval_T * sequence_length) # 0ms から 1000ms
    rng = np.random.default_rng(42)
    selected_numbers = rng.choice(range(num_ca3_patterns), size=num_ca3_patterns_input, replace=False)
    ca3_input_sequence = rng.integers(0, num_ca3_patterns_input, size=sequence_length).tolist()
    for input_idx in range(sequence_length):
        ca3_input_sequence[input_idx] = selected_numbers[ca3_input_sequence[input_idx]]

    '''
    Initialize CA1-CA3 network and simulate 
    Get data sets to train LDA model
    '''
    print(f"Initializing CA1 Network with {num_ca1_neurons} neurons...")
    ca1_network = CA1Network(
        num_ca1_neurons=num_ca1_neurons,
        num_ca3_neurons=num_ca3_neurons,
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
    print(f"Network simulation completed:{end - start:.4f}")

    '''
    Create correspondance between state vector and input history of length d
    '''
    lda = Linear_Discriminant_Analysis(ca3_input_sequence, sequence_transient_length, sequence_analysis_length, ca1_newtork, network_sol)
    
    

    '''
    Separate data sets into 10 segments
    9 for training, 1 for validating
    '''


    '''
    Train LDA model
    '''


    '''
    Validate LDA model
    '''


    '''
    Plot z-plane vs the number of samples
    '''
   

    '''
    Plot z-plane vs the number of samples
    '''


