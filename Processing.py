import os
import pandas as pd

# Function to preprocess a CSV file (e.g., changing datetime format)
def preprocess_csv(file_path):
    try:
        # Load the CSV
        df = pd.read_csv(file_path)
        
        # Example preprocessing: change datetime format in a column (modify as needed)
        # if 'Date Time' in df.columns:
        #     df['Date Time'] = pd.to_datetime(df['Date Time']).dt.strftime('%d/%m/%y %H:%M')
        
        
        # Remove rows where 'Slave_Device1_CH1_FLOW m3/h', 'Slave_Device1_CH2_VELOCITY m/s', and 'Slave_Device1_CH3_Pressure BAR' are all zero
        df_cleaned = df[
            ~((df['Slave_Device1_CH1_FLOW m3/h'] == 0) &
            (df['Slave_Device1_CH2_VELOCITY m/s'] == 0) &
            (df['Slave_Device1_CH3_Pressure BAR'] == 0) &
            (df['Slave_Device1_CH4_Voltage V']))
        ]

        # Remove rows with negative values in any of the relevant columns
        df_cleaned = df_cleaned[
            (df_cleaned['Slave_Device1_CH1_FLOW m3/h'] > 0) &
            (df_cleaned['Slave_Device1_CH2_VELOCITY m/s'] > 0) &
            (df_cleaned['Slave_Device1_CH3_Pressure BAR'] > 0)
        ]

        # Save the cleaned dataset
        # df_cleaned.to_csv(output_file, index=False)
        # print(f"Cleaned dataset saved to {output_file}")
        output_file = os.path.join(output_path, os.path.basename(file_path).replace('.csv', '_corrected.csv'))
        # Save the processed CSV (you can overwrite or save to another directory)
        df_cleaned.to_csv(file_path, index=False)
        
    except Exception as e:
        print(f'Error processing {file_path}: {e}')

# Function to process all CSV files in the subfolders
def process_all_csv_in_folders(main_folder_path):
    # Loop through all subfolders and files
    for root, dirs, files in os.walk(main_folder_path):
        for file in files:
            if file.endswith(".csv"):
                # Get full file path
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                
                # Preprocess the CSV
                preprocess_csv(file_path)

# Main folder path containing 9 subfolders
main_folder_path = 'Raw Data_2023'  # Replace with the path to your main folder
output_path = 'Pre-Processed Data'
# Run the processing function on all CSV files
process_all_csv_in_folders(main_folder_path)
