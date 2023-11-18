import jsonlines

# Define the input and output file paths
# input_file =r"C:\\Users\\user\\Downloads\\test_decoded_processed.json"
input_file = r"C:\\Users\\user\\Downloads\\eval_decoded_processed.json"
output_files = ["eval_decoded_processed_1.json", "eval_decoded_processed_2.json", "eval_decoded_processed_3.json"]

def split_jsonl(input_file, output_files, num_chunks):
    with jsonlines.open(input_file) as reader:
        data = list(reader)
        chunk_size = len(data) // num_chunks
        print(chunk_size)
        print(len(data))
        for chunk_num, output_file in enumerate(output_files, start=1):
           chunk_data = data[(chunk_num - 1) * chunk_size : chunk_num * chunk_size]
           file=f'C:\\Users\\user\\Downloads\\new_valdation\\{output_file}'
           with jsonlines.open(file, 'w') as writer:
                writer.write_all(chunk_data)
           
# Split the JSONL file into five chunks
split_jsonl(input_file, output_files, num_chunks=3)
