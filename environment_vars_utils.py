def read_json_from_drive(file_path):
    """
    Reads a JSON file from Google Drive and returns its contents
    
    Parameters:
    file_path (str): Full path to the JSON file in Google Drive
    
    Returns:
    dict: The contents of the JSON file
    """
    # Mount Google Drive if not already mounted
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    
    try:
        # Read and parse the JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    
    except FileNotFoundError:
        print(f"Error: Could not find file at {file_path}")
        print("Tip: Make sure you've provided the correct path to your file")
        return None
    
    except json.JSONDecodeError as e:
        print(f"Error: The file is not valid JSON")
        print(f"Details: {str(e)}")
        return None
    
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None

# Usage example:
# file_path = '/content/drive/My Drive/example.json'
# data = read_json_from_drive(file_path)

# if data is not None:
#     print("Successfully loaded JSON data:")
#     print(data)
