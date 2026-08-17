import json

def load_data(filepath):
    """Load processed document data from a JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Data file not found: {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {filepath}")
        return []
    except Exception as exc:
        print(f"An error occurred loading data: {exc}")
        return []
