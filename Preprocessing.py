import os
import pandas as pd

def preprocess_csv(file_path, output_path, log_file):
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        original_rows = len(df)

        # Remove 'Slave_Device1_CH4_Voltage V' column
        df_cleaned = df.drop(columns=['Slave_Device1_CH4_Voltage V'])

        # Remove rows where 'Slave_Device1_CH1_FLOW m3/h', 'Slave_Device1_CH2_VELOCITY m/s', and 'Slave_Device1_CH3_Pressure BAR' are all zero
        df_cleaned = df_cleaned[
            ~((df_cleaned['Slave_Device1_CH1_FLOW m3/h'] == 0) &
            (df_cleaned['Slave_Device1_CH2_VELOCITY m/s'] == 0) &
            (df_cleaned['Slave_Device1_CH3_Pressure BAR'] < 0.2))
        ]

        # Remove rows with negative values in ay of the relevant columns
        df_cleaned = df_cleaned[
            (df_cleaned['Slave_Device1_CH1_FLOW m3/h'] >= 0) &
            (df_cleaned['Slave_Device1_CH2_VELOCITY m/s'] >= 0) &
            (df_cleaned['Slave_Device1_CH3_Pressure BAR'] >= 0)
        ]

        # Count how many rows were removed
        cleaned_rows = len(df_cleaned)
        rows_removed = original_rows - cleaned_rows

        # Save the cleaned CSV with 'corrected' suffix in the output path
        output_file = os.path.join(output_path, os.path.basename(file_path).replace('.csv', '_corrected.csv'))
        df_cleaned.to_csv(output_file, index=False)

        # Log the number of rows removed
        with open(log_file, 'a') as log:
            log.write(f'File: {file_path} - Rows Removed: {rows_removed}\n')

        print(f'Processed and saved: {output_file} - Rows removed: {rows_removed}')
        
    except Exception as e:
        print(f'Error processing file: {file_path} - {str(e)}')

def preprocess_folder(input_folder, output_folder, log_file):
    # Clear log file if exists
    with open(log_file, 'w') as log:
        log.write('Preprocessing Log\n')
        log.write('=================\n')

    # Walk through each folder and process CSV files
    for root, dirs, files in os.walk(input_folder):
        
        # Create corresponding output directories
        relative_path = os.path.relpath(root, input_folder)
        output_path = os.path.join(output_folder, relative_path)
        os.makedirs(output_path, exist_ok=True)

        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                preprocess_csv(file_path, output_path, log_file)

if __name__ == "__main__":
    input_folder = 'Raw Data_2023'  
    output_folder = 'Preprocessed_Data'  # Path to save preprocessed files
    log_file = 'preprocessing_log.txt'  # Log file to store the number of rows removed for each file
    preprocess_folder(input_folder, output_folder, log_file)
