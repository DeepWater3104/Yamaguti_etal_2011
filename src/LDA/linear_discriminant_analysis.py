class  Linear_Discriminant_Analysis:
    def __init__(self,):

    def _get_matrix_A(self, ):


if __name__ == '__main__':
    '''
    Parameters
    '''
    num_ca1_neurons = 100      # CA1ニューロン数 (論文 Figure 9で100まで)
    num_ca3_patterns = 20      # M=100 (論文では具体的な値が指定されていない)
    num_ca3_patterns_input = 3 # m=3 (主成分空間の次元と一致させる)
    t_interval_T = 100.0       # T=100ms (論文 Figure 4) 
    duration_delta = 5.0       # delta=5ms (論文 Section 3.1) 
    sim_dt = 0.05              # シミュレーションタイムステップ 
    neuron_type  = ["bursting" for _ in range(num_ca1_neurons)]
    synapse_type = ["BOTH"     for _ in range(num_ca1_neurons)]
    sequence_length = 1000     # T間隔の数
    t_span_network = (0, t_interval_T * sequence_length) # 0ms から 1000ms
    rng = np.random.default_rng(42)
    selected_numbers = rng.choice(range(num_ca3_patterns), size=num_ca3_patterns_input, replace=False)
    ca3_input_sequence = rng.integers(0, num_ca3_patterns_input, size=sequence_length).tolist()
    for input_idx in range(sequence_length):
        ca3_input_sequence[input_idx] = selected_numbers[ca3_input_sequence[input_idx]]

    '''
    Generate CA3 spatial patterns according to rule described in the original paper
    '''


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


