import os
import shutil

def rename_images(source_dir='../figure', target_dir='../figure2'):
    """
    Renames image files in the format PCA_vs_depth2_WT{synaptic_weight}_NC3N{number_of_neurons}.png
    to ensure synaptic_weight increases continuously, and moves them to a new directory.

    Args:
        source_dir (str): The directory containing the original image files.
        target_dir (str): The directory where the renamed files will be saved.
    """
    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    file_data = []
    # Collect data from existing files
    for filename in os.listdir(source_dir):
        if filename.startswith('PCA_vs_depth2_WT') and filename.endswith('.png'):
            parts = filename.replace('.png', '').split('_')
            
            # Extract synaptic_weight
            wt_part = [p for p in parts if p.startswith('WT')]
            if wt_part:
                synaptic_weight_str = wt_part[0].replace('WT', '')
                # Convert 'p' to '.' and then to float
                try:
                    synaptic_weight_float = float(synaptic_weight_str.replace('p', '.'))
                except ValueError:
                    print(f"Warning: Could not parse synaptic weight from {filename}. Skipping.")
                    continue
            else:
                print(f"Warning: Could not find WT part in {filename}. Skipping.")
                continue

            # Extract number_of_neurons
            nc3n_part = [p for p in parts if p.startswith('NC3N')]
            if nc3n_part:
                number_of_neurons_str = nc3n_part[0].replace('NC3N', '')
                try:
                    number_of_neurons_int = int(number_of_neurons_str)
                except ValueError:
                    print(f"Warning: Could not parse neuron count from {filename}. Skipping.")
                    continue
            else:
                print(f"Warning: Could not find NC3N part in {filename}. Skipping.")
                continue

            file_data.append({
                'original_filename': filename,
                'synaptic_weight_float': synaptic_weight_float,
                'number_of_neurons_int': number_of_neurons_int
            })

    # Sort files by synaptic weight first, then by neuron count
    file_data.sort(key=lambda x: (x['synaptic_weight_float'], x['number_of_neurons_int']))

    # Define new synaptic weight parameters
    start_synaptic_weight = 0.01
    increment_synaptic_weight = 0.01

    # Rename and move files
    for i, data in enumerate(file_data):
        original_filename = data['original_filename']
        number_of_neurons = data['number_of_neurons_int']

        # Calculate new synaptic weight
        new_synaptic_weight = start_synaptic_weight + (i * increment_synaptic_weight)
        
        # Format new synaptic weight to '0p0000' format
        # This assumes 4 decimal places for consistency with '0p0300'
        new_synaptic_weight_str = f"{new_synaptic_weight:.4f}".replace('.', 'p')

        # Construct new filename
        new_filename = f"PCA_vs_depth2_WT{new_synaptic_weight_str}_NC3N{number_of_neurons:04d}.png" # 04d for 4 digits padding for neuron count

        original_path = os.path.join(source_dir, original_filename)
        new_path = os.path.join(target_dir, new_filename)

        print(f"Renaming and moving: {original_path} -> {new_path}")
        shutil.copy(original_path, new_path)

    print(f"Successfully processed {len(file_data)} files.")

# To run this, place your image files in the same directory as this script.
# The renamed files will be saved in a new directory named 'figure2' one level up from the current directory.
# rename_images() # This line was executed by the interpreter.

