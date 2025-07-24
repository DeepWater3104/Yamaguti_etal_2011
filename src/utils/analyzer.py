import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

def get_averaged_soma_potentials(all_vs_matrix):
    num_time_steps = sol.t.shape[0]
    num_intervals = int(num_time_steps * self.dt / t_interval_T)
    averaged_vs_matrix = np.zeros((self.num_ca1_neurons, num_intervals))
    
    for k in range(num_intervals):
        start_time_idx = int((k * t_interval_T) / self.dt)
        end_time_idx = int(((k + 1) * t_interval_T) / self.dt)
        averaged_vs_matrix[:, k] = np.mean(all_vs_matrix[:, start_time_idx:end_time_idx], axis=1)

    return averaged_vs_matrix

def PCA():
    # transform data into discrete-time data
    all_vs_matrix = np.load("../data/all_vs_matrix.npy")
    averaged_vs_matrix = ca1_network.get_averaged_soma_potentials(network_sol, t_interval_T)
    spike_counts_matrix = ca1_network.get_spike_counts(network_sol, t_interval_T)
    fig_pca = plt.figure(figsize=(10, 8))
    ax_pca = fig_pca.add_subplot(111, projection='3d')

    print("\nStarting PCA and 3D plotting (no color coding)...")
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
    plt.savefig("../figure/PCA_depth1_" + filename_parts + ".png")
    print("PCA 3D plot saved successfully.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parameters setting for the simulation')
    parser.add_argument(
        '--w_tilde',
        type=float,
        default=0.01,
        help='Scaling factor of synaptic weight'
    )
    args = parser.parse_args()
    w_tilde = args.w_tilde

    print(f"Scaling fact or of synaptic weight : {w_tilde}")
    filename_parts_list = []
    filename_parts_list.append(f"WT{w_tilde:.3f}".replace('.', 'p'))
    filename = ""
    if filename_parts_list:
        filename_parts = f"{'_'.join(filename_parts_list)}"
        print(filename_parts)
    else:
        filename_parts = f""
        print(filename_parts)
