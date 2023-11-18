import os
import pandas as pd
import jsonlines
import csv


# Define a function to change the extension
def change_extension(filepath):
    root, ext = os.path.splitext(filepath)
    return root + '.mp3'



def combineDfs():
    folder_path = r'C:\\Users\\user\\Downloads\\CSV\\test'

    all_files = os.listdir(folder_path)

    csv_files = [f for f in all_files if f.endswith('.csv')]

    df_list=[]

    for csv in csv_files:
        file_path = os.path.join(folder_path,csv)
        try:
            df = pd.read_csv(file_path)
            df_list.append(df)
        except Exception as e:
            print(f'could not read file{csv} because of error: {e}')
            
    big_df = pd.concat(df_list,ignore_index=True)
    file_path= big_df['path']
    big_df['path']=big_df['path'].apply(change_extension)

    big_df.to_csv(os.path.join(folder_path,'Newsample_data.csv'),index=False)


    print(f'file saved succefully')


# combineDfs()
# def  compare(input,output):
#     with open(input,r,newline='') asn infile:
#         reader  = csv.reader

# exit()

JsonPath =r'C:\Users\user\Downloads\\new_valdation\\final_data.json'

folder_path = r'C:\Users\user\Downloads\new_valdation'



csvName = "eval_with_predictions.json"
data_list = {}  # List to store the JSON data

sentence = []  # This list will store the processed data
path = []  # This list will store the processed data

with jsonlines.open(JsonPath) as reader:
    for obj in reader:
        file_path = os.path.basename(obj['audio_filepath'])
        file_name = os.path.splitext(file_path)[0] + ".mp3"
        path.append(file_name)  # the non-empty row to the output_data list
        sentence.append(0 if not obj['pred_text'].strip() else obj['pred_text'])# the non-empty row to the output_data list

# Convert the JSON data to a DataFrame
# print(path)
def read_csv_to_array(csv_path):
    data_array = []

    with open(csv_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:\
            data_array.append(row[0])

    return data_array

# Replace 'your_file.csv' with the actual path to your CSV file
csv_path1 = r'C:\\Users\\user\\Downloads\\new_valdation\\SampleSubmission.csv'
csv_path = r'C:\\Users\\user\\Downloads\\new_valdation\\final_data.csv'
data_array = read_csv_to_array(csv_path)
data_array1 = read_csv_to_array(csv_path1)

# Now, data_array contains the data from the CSV file
print(len(data_array))
print(len(data_array1))
print(len(data_array1) - len(data_array))

# Find elements in array1 that are not in array2
not_in_array2 = list(set(data_array1) - set(data_array))

# Find elements in array2 that are not in array1
not_in_array1 = list(set(data_array) - set(data_array1))
print(len(not_in_array2))
print(len(not_in_array1))
print("Elements in array1 not in array2:", not_in_array2)
print("Elements in array2 not in array1:", not_in_array1)
SampeData = {
    "path":path,
    "sentence":sentence,
    
}
# print(SampeData)
df = pd.DataFrame(SampeData)

# print(df.head(10))

# Removing rows with an empty "sentence" column
# df = df[df['sentence'].str.strip() != '']
folder_path = r'C:\Users\user\Downloads\new_valdation'

# # Now df contains rows with non-empty "sentence" values
# print(df.head())
# # List of IDs to remove
# remove_these_ids = ["common_voice_sw_37664539.mp3", "common_voice_sw_31290838.mp3", "common_voice_sw_28266617.mp3", "common_voice_sw_35087387.mp3", "common_voice_sw_37214113.mp3"]  
# missingrows = [
    # 'common_voice_sw_35557525.mp3', 
    # 'common_voice_sw_30627972.mp3', 
    # 'common_voice_sw_37094180.mp3', 
    # 'common_voice_sw_37209553.mp3',
    # 'common_voice_sw_29955015.mp3',
    # 'common_voice_sw_35087387.mp3', 
    # 'common_voice_sw_37664539.mp3', 
    # 'common_voice_sw_31290838.mp3', 
    # 'common_voice_sw_37214113.mp3',
    # 'common_voice_sw_36450168.mp3', 
    # 'common_voice_sw_31375973.mp3',
    # 'common_voice_sw_30621896.mp3', 
    # 'common_voice_sw_29969526.mp3',
    # 'common_voice_sw_28266617.mp3',
    # 'common_voice_sw_31293094.mp3', 
    # 'common_voice_sw_30034497.mp3', 
# 'common_voice_sw_30479058.mp3'
# ]
# # Remove rows with the specified IDs
# df = df[~df['path'].isin(remove_these_ids)]
# common_voice_sw_28266617.mp3,alipenda  kuimba ndaima alitaka kuimba
# common_voice_sw_37214113,0
# common_voice_sw_37664539.mp3,rangi yake ni ya jano kwa mmonekano  wa ndani 
# common_voice_sw_31290838,alisahau
# common_voice_sw_35087387,0
# df.to_csv(os.path.join(folder_path,'final_data.csv'),index=False)

def find_duplicates(csv_path, column_name):
    # Check if the file exists
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Find duplicates based on the specified column
    duplicates = df[df.duplicated(subset=[column_name], keep=False)]
    print(len(duplicates))
    print(duplicates)
    # Print path and sentence for each duplicate
    # for index, row in duplicates.iterrows():
        # print(f"Path: {csv_path}, Sentence: {row['sentence']}")

# Example usage
column_to_check = 'path'

# find_duplicates(csv_path, column_to_check)
# Function to remove duplicates in a CSV file based on a column
def remove_duplicates(csv_path, column_name):
    # Check if the file exists
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Remove duplicates based on the specified column
    df_no_duplicates = df.drop_duplicates(subset=[column_name])
    
    # Save the DataFrame with duplicates removed
    df_no_duplicates.to_csv(csv_path, index=False)
    
    print(f"Duplicates removed from {csv_path}")

# Example usage

remove_duplicates(csv_path, column_to_check)