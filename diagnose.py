import os


def check_model_structure():
    """
    Check the model directory structure and existing files
    """
    # Get current working directory
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")

    # Check model directory structure
    model_dir = os.path.join('model', 'sites_models')
    abs_model_dir = os.path.join(current_dir, 'model', 'sites_models')

    print("\nChecking directory structure:")
    print(f"Looking for model directory at: {abs_model_dir}")

    # Check if directories exist
    if not os.path.exists(os.path.join(current_dir, 'model')):
        print("'model' directory not found!")
        return

    if not os.path.exists(abs_model_dir):
        print("'sites_models' directory not found!")
        return

    # List all files in the model directory
    print("\nExisting model files:")
    model_files = os.listdir(abs_model_dir)
    if model_files:
        for file in model_files:
            file_path = os.path.join(abs_model_dir, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            print(f"- {file} ({file_size:.2f} MB)")
    else:
        print("No model files found in the directory!")

    # Check if specific models exist
    print("\nChecking for specific models:")
    model_types = ['lstm', 'gru', 'saes']
    test_sites = ['970', '3001']  # Add more sites as needed

    for site in test_sites:
        for model_type in model_types:
            model_name = f"{model_type}_{site}.h5"
            model_path = os.path.join(abs_model_dir, model_name)
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
                print(f"✓ Found {model_name} ({file_size:.2f} MB)")
            else:
                print(f"✗ Missing {model_name}")


if __name__ == "__main__":
    check_model_structure()