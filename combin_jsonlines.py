import jsonlines

# List of input JSONL files to combine
input_files = ["eval_version3_1.json","eval_version3_3.json","test_with_predictions_4.json","eval_version3_2.json"]


# Output file where the combined JSONL will be stored
file=r'C:\\Users\\user\\Downloads\\new_valdation\\final_data.json'
# Function to combine multiple JSONL files
def combine_jsonl(input_files, output_file):
    with jsonlines.open(output_file, 'w') as writer:
        for input_file in input_files:
            file=f'C:\\Users\\user\\Downloads\\new_valdation\\{input_file}'
            with jsonlines.open(file) as reader:
                for line in reader:
                    writer.write(line)

# Combine the JSONL files
combine_jsonl(input_files, file)
