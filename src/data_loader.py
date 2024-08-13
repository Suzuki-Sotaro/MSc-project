import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """
        Load the data from the CSV file.
        """
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"Data loaded successfully from {self.file_path}.")
        except Exception as e:
            print(f"An error occurred while loading the data: {e}")
            raise

    def preprocess_data(self):
        """
        Preprocess the data.
        - Extract the relevant buses specified in the professor's guidance.
        - Filter the last 855 data points as required.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() before preprocessing.")

        # Extract the specified buses
        buses = ['Bus115', 'Bus116', 'Bus117', 'Bus118', 'Bus119', 'Bus121', 'Bus135', 'Bus139']
        columns_to_keep = ['Week', 'Label'] + buses

        if not all(bus in self.data.columns for bus in buses):
            raise ValueError(f"Some of the specified buses are not in the dataset columns: {buses}")

        # Filter the relevant columns
        self.data = self.data[columns_to_keep]

        # Keep the last 855 data points
        self.data = self.data.tail(855)

        print(f"Data preprocessed successfully. {len(self.data)} rows remaining after filtering.")

    def get_data(self):
        """
        Returns the preprocessed data.
        """
        if self.data is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() before getting the data.")
        return self.data

# Example usage:
if __name__ == "__main__":
    file_path = './data/LMP.csv'
    data_loader = DataLoader(file_path)
    
    data_loader.load_data()        # Load the data from the file
    data_loader.preprocess_data()  # Preprocess the data as needed
    processed_data = data_loader.get_data()  # Get the processed data
    
    # Print out a preview of the processed data
    print(processed_data.head())
