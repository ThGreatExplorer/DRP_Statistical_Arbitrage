import pandas as pd
import os

# Directory containing the Excel files
directory = 'BBG_Excels'

# directory to place the Csv files
directoryTo = 'csv'

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(directory, filename)
        # Remove the extension for naming the CSV files
        file_base_name = os.path.splitext(filename)[0]
        
        if (file_base_name.__contains__("_15Min")):

            # Read the Excel file
            xls = pd.ExcelFile(file_path)

            # Save each sheet as a CSV
            for sheet_name in xls.sheet_names:
                df = xls.parse(sheet_name)
                csv_file_name = f'{file_base_name}_{sheet_name}.csv'
                print(csv_file_name)
                csv_file_path = os.path.join(directoryTo, csv_file_name)
                df.to_csv(csv_file_path, index=False)

print("Conversion complete.")
