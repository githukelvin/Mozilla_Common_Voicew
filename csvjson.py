import csv
import json

def csv_to_jsonline(csv_file_path, jsonline_file_path):
    # Open the CSV file for reading
    with open(csv_file_path, 'r') as csv_file:
        # Open the JSONLines file for writing
        with open(jsonline_file_path, 'w') as jsonline_file:
            # Create a CSV reader
            csv_reader = csv.DictReader(csv_file)
            
            # Iterate through each row in the CSV file
            for row in csv_reader:
                # Convert the row to a JSON object
                json_object = json.dumps(row)
                
                # Write the JSON object as a line in the JSONLines fil
                jsonline_file.write(json_object + '\n')
file=r'C:\\Users\\user\\Downloads\\new_valdation\\test_with_predictions.csv'
savefile=r'C:\\Users\\user\\Downloads\\new_valdation\\test_predictions.json'
csv_to_jsonline(file, savefile)



