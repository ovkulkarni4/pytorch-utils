import json
import os
from google.colab import drive

def setup_environment_from_json(json_path):
    """
    Reads a JSON file containing key-value pairs and sets them as environment variables.
    
    Parameters:
    json_path (str): Path to the JSON file in Google Drive
    
    Returns:
    dict: Dictionary of successfully set environment variables
    """
    # First, mount Google Drive if not already mounted
    drive.mount('/content/drive', force_remount=False)
    
    try:
        # Read the JSON file
        with open(json_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
            
        # Validate that we have a dictionary
        if not isinstance(config, dict):
            raise ValueError("JSON file must contain a key-value object")
            
        # Set each key-value pair as an environment variable
        for key, value in config.items():
            # Convert value to string since environment variables must be strings
            os.environ[str(key)] = str(value)
            print(f"Set environment variable: {key}")
            
        return config
        
    except FileNotFoundError:
        print(f"Error: Could not find config file at {json_path}")
        print("Tip: Double-check your file path and make sure the file exists in Drive")
        return None
        
    except json.JSONDecodeError as e:
        print(f"Error: Your JSON file is not properly formatted")
        print(f"Details: {str(e)}")
        return None
        
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None

# Let's test if our environment variables were set correctly
def verify_environment_variables(config):
    """
    Verifies that environment variables were set correctly.
    
    Parameters:
    config (dict): Dictionary of key-value pairs that should be in environment
    """
    if config is None:
        return
        
    print("\nVerifying environment variables:")
    for key in config:
        value = os.getenv(key)
        if value is not None:
            # Show first few characters of the value for security
            masked_value = value[:3] + "..." if len(value) > 3 else value
            print(f"✓ {key}: {masked_value}")
        else:
            print(f"✗ {key}: Not set!")

# Getting environment variables with default values
def get_env_variable(key, default=None, var_type=str):
    """
    Safely get and convert environment variables to the specified type.
    """
    value = os.getenv(key, default)
    if value is None:
        return None
    
    try:
        return var_type(value)
    except ValueError:
        print(f"Warning: Could not convert {key} to {var_type.__name__}")
        return default
        
# Usage example
# json_path = '/content/drive/My Drive/config.json'
# config = setup_environment_from_json(json_path)
# verify_environment_variables(config)
