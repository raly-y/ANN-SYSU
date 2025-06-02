import os
from src.utils import load_config
from src.train import train
from src.inference import inference


def main():
    # Define project root
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(PROJECT_ROOT, 'config.yaml')

    # Load config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = load_config(config_path)

    # Get mode and validate
    mode = config['main']['mode']
    if mode not in ['train', 'inference', 'both']:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'inference'")

    # Run based on mode
    if mode == 'train':
        print("Starting training...")
        train(config)
        print("Training completed successfully.")
    elif mode == 'inference':
        print("Starting inference...")
        inference(config)
        print("Inference completed successfully.")
    else:   # mode == 'both'
        print("Starting training...")
        train(config)
        print("Training completed successfully.")
        print("Starting inference...")
        inference(config)
        print("Inference completed successfully.")


if __name__ == "__main__":
    main()
