import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, file_path='./data/LMP.csv'):
        self.file_path = file_path
        self.data = None
        self.selected_buses = [115, 116, 117, 118, 119, 121, 135, 139]
        self.window_size = 855  # Last 855 values as suggested

    def load_data(self):
        """Load the CSV file and perform initial preprocessing."""
        self.data = pd.read_csv(self.file_path)
        print(f"Data loaded. Shape: {self.data.shape}")

    def preprocess_data(self):
        """Preprocess the data by selecting specific buses and time window."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Select specific buses
        bus_columns = [f'Bus{i}' for i in self.selected_buses]
        selected_data = self.data[['Week', 'Label'] + bus_columns]

        # Select the last 855 rows
        selected_data = selected_data.tail(self.window_size)

        # Reset index
        selected_data = selected_data.reset_index(drop=True)

        print(f"Preprocessed data shape: {selected_data.shape}")
        return selected_data

    def get_bus_data(self, bus_number):
        """Get data for a specific bus."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        column_name = f'Bus{bus_number}'
        if column_name not in self.data.columns:
            raise ValueError(f"Bus {bus_number} not found in the dataset.")

        return self.data[['Week', 'Label', column_name]].tail(self.window_size)

    def get_all_selected_bus_data(self):
        """Get data for all selected buses."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        return self.preprocess_data()

if __name__ == "__main__":
    # Test the DataLoader
    loader = DataLoader()
    loader.load_data()
    processed_data = loader.preprocess_data()
    print(processed_data.head())

    # Example: Get data for Bus 115
    bus_115_data = loader.get_bus_data(115)
    print("\nBus 115 data:")
    print(bus_115_data.head())

    # Example: Get data for all selected buses
    all_bus_data = loader.get_all_selected_bus_data()
    print("\nAll selected bus data:")
    print(all_bus_data.head())