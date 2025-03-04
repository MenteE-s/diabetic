import pandas as pd

def merge_csv_files(file_list, output_file):
    merged_df = pd.DataFrame()
    
    for file in file_list:
        df = pd.read_csv(file)
        
        # Move last column to first position
        cols = df.columns.tolist()
        df = df[[cols[-1]] + cols[:-1]]
        
        # Append to merged dataframe
        merged_df = pd.concat([merged_df, df], ignore_index=True)
    
    # Save merged file
    merged_df.to_csv(output_file, index=False)
    print(f"Merged file saved as {output_file}")

# Example usage
file_list = [r"H:\Research\MenteE's\Diabetes\Diabetes Mellitus\dataset\clean\Mild_features.csv",
            r"H:\Research\MenteE's\Diabetes\Diabetes Mellitus\dataset\clean\Moderate_features.csv",
            r"H:\Research\MenteE's\Diabetes\Diabetes Mellitus\dataset\clean\No_DR_features.csv", 
            r"H:\Research\MenteE's\Diabetes\Diabetes Mellitus\dataset\clean\Proliferate_DR_features.csv", 
            r"H:\Research\MenteE's\Diabetes\Diabetes Mellitus\dataset\clean\Severe_features.csv"]
merge_csv_files(file_list, "DR_dataset.csv")
