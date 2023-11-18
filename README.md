# SWAHILI ASR USING MOZILLA COMMON VOICE DATASET

I describe how to train an asr model for swahili model from scratch

using the Mozilla Common Voice dataset. The code is written in python and uses pytorch, torchtext and fastai libraries.
The MVC  dataset  we used  has 126+ hours of audio data

<b>Note you need GPU powered computer to train nemo model use colab. Tried kaggle but faced some issues installing nemo libraries</b>

### 1. Preparing the dataset

Link to the dataset is here <a href="https://drive.google.com/drive/folders/1nesfH2lgKLKYNUn2X_9VOexlHqtQQMap?usp=sharing">Google drive  link to get the dataset</a>
After you have  downloaded the  data the next step is preparing the dataset.
Upload   it  to Mozilla folder  and then extract the files

<code>

!tar -xzvf "/content/drive/Mozilla/eval0.tar.gz" -C "/content/drive/MyDrive/Mozilla/clips/"     #[run this cell to extract tar.gz files]
!tar -xzvf "/content/drive/Mozilla/test0.tar.gz" -C "/content/drive/MyDrive/Mozilla/clips/"     #[run this cell to extract tar.gz files]
!tar -xzvf "/content/drive/Mozilla/train0.tar.gz" -C "/content/drive/MyDrive/Mozilla/clips/"     #[run this cell to extract tar.gz files]

</code>

<code>

from google.colab import drive

drive.mount('/content/drive')

</code>

The  code above is to connect to google drive you  uploaded your  dataset  to.

<code>

# second  step  run this  step  after

!pip install --upgrade pip

!pip install wget text-unidecode

!python -m pip install git+https://github.com/NVIDIA/NeMo.git

!pip install nemo_toolkit['all']

!apt-get update && apt-get install -y libsndfile1 ffmpeg sox   libsox-fmt-mp3

!pip install Cython

!sudo apt-get update

!sudo apt-get upgrade

!sudo apt-get install lame

!pip install sox lameenc jsonlines

!pip install ffmpeg-python wget tensorflow

!pip install text-unidecode pydub

!pip install matplotlib>=3.3.2

# Install NeMo

BRANCH = 'main'

!python -m pip install git+https://github.com/NVIDIA/NeMo.git@$BRANCH#egg=nemo_toolkit[all]

!pip install tensorflow[and-cuda]

</code>

Code  above is to get all  the necessary libraries for nemo

<code>

import tensorflow as tf

print("GPU Available: ", tf.config.list_physical_devices('GPU'))

</code>

This code displays if you have GPU  available.

The following code to follow is to convert tsvs and csv provided  by MVC  to json file

<code>

!python drive/MyDrive/hackathon/tsv_to_json.py \
  --tsv=drive/MyDrive/hackathon/train.tsv \
  --folder=/content/drive/MyDrive/hackathon/clips/clean_train/train \
  --sampling_count=-1

</code>

<code>

!python drive/MyDrive/hackathon/csv_to_json.py \
  --csv=drive/MyDrive/hackathon/eval.csv \
  --folder=/content/drive/MyDrive/hackathon/clips/eval \
  --sampling_count=-1

</code>

<code>

!python drive/MyDrive/hackathon/csv_to_json.py \
  --csv=drive/MyDrive/hackathon/test.csv \
  --folder=/content/drive/MyDrive/hackathon/clips/test \
  --sampling_count=-1

</code>

The  following couple of codes  blocks are used to make directories resample the audio  data from .mp to wav with sample rate of 16000hz and the new resampled files are transferred to the new directory

<code>

!mkdir drive/MyDrive/hackathon/train_data/
!python drive/MyDrive/hackathon/decode_resample.py \
  --manifest=drive/MyDrive/hackathon/train.json \
  --destination_folder=drive/MyDrive/hackathon/train_data

!mkdir drive/MyDrive/hackathon/eval_data/
!python drive/MyDrive/hackathon/csv_resample.py \
  --manifest=drive/MyDrive/hackathon/eval.json \
  --destination_folder=/content/drive/MyDrive/hackathon/eval_data

!mkdir drive/MyDrive/hackathon/test_data/
!python drive/MyDrive/hackathon/csv_resample.py \
  --manifest=drive/MyDrive/hackathon/test.json \
  --destination_folder=drive/MyDrive/hackathon/test_data

</code>

The above code has both code for will  create folders  respectively for test and eval audio data

<code>

# prepare_dataset_kiswahili.py

import json
import os
import re
from collections import defaultdict
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
from tqdm.auto import tqdm

def write_processed_manifest(data, original_path):
    original_manifest_name = os.path.basename(original_path)
    new_manifest_name = original_manifest_name.replace(".json", "_processed.json")

    manifest_dir = os.path.split(original_path)[0]
    filepath = os.path.join(manifest_dir, new_manifest_name)
    write_manifest(filepath, data)
    print(f"Finished writing manifest: {filepath}")
    return filepath

# calculate the character set

def get_charset(manifest_data):
    charset = defaultdict(int)
    for row in tqdm(manifest_data, desc="Computing character set"):
        text = row['text']
        for character in text:
            charset[character] += 1
    return charset

# Preprocessing steps

def remove_special_characters(data):
    chars_to_ignore_regex = "[\.\,\?\:\-!;()«»…\]\[/\*–‽+&_\\½√>€™$•¼}{~—=“\"”″‟„]"
    apostrophes_regex = "[’'‘`ʽ']"
    # print(data["text"])
    if data is not None and isinstance(data, dict) and "text" in data:
        text = data["text"]

        # Check if text is not empty
        if text:
            # Chain regular expressions for better performance
            cleaned_text = re.sub(chars_to_ignore_regex, " ", text)
            cleaned_text = re.sub(apostrophes_regex, "'", cleaned_text)
            cleaned_text = re.sub(r"'+", "'", cleaned_text)
            cleaned_text = re.sub(r"([b-df-hj-np-tv-z])' ([aeiou])", r"\1'\2", cleaned_text)
            cleaned_text = re.sub(r" '| '", " ", cleaned_text)
            cleaned_text = re.sub(r"' ", " ", cleaned_text)
            cleaned_text = re.sub(r" +", " ", cleaned_text)

            data["text"] = cleaned_text
        else:
            print(f"{data['text']} is empty")
    else:
        print("data is None or not a dictionary")

    return data

def replace_diacritics(data):
    data["text"] = re.sub(r"[éèëēê]", "e", data["text"])
    data["text"] = re.sub(r"[ãâāá]", "a", data["text"])
    data["text"] = re.sub(r"[úūü]", "u", data["text"])
    data["text"] = re.sub(r"[ôōó]", "o", data["text"])
    data["text"] = re.sub(r"[ćç]", "c", data["text"])
    data["text"] = re.sub(r"[ïī]", "i", data["text"])
    data["text"] = re.sub(r"[ñ]", "n", data["text"])
    data["text"] = re.sub(r"[ŧ]", "t", data["text"])
    data["text"] = re.sub(r"[ȳ]", "y", data["text"])
    data["text"] = re.sub(r"[š]", "s", data["text"])
    data["text"] = re.sub(r"[ǩ]", "k", data["text"])
    data["text"] = re.sub(r"[ğ]", "g", data["text"])
    return data

def remove_oov_characters(data):
    oov_regex = "[^ 'aiuenrbomkygwthszdcjfvplxq]"
    data["text"] = re.sub(oov_regex, "", data["text"])  # delete oov characters
    data["text"] = data["text"].strip()
    return data

# Processing pipeline

def apply_preprocessors(manifest, preprocessors):
    for processor in preprocessors:
        for idx in tqdm(range(len(manifest)), desc=f"Applying {processor.__name__}"):
            manifest[idx] = processor(manifest[idx])

    print("Finished processing manifest !")
    return manifest

# List of pre-processing functions

PREPROCESSORS = [
    remove_special_characters,
    replace_diacritics,
    remove_oov_characters,
]

train_manifest = "./drive/MyDrive/hackathon/train_decoded.json"
eval_manifest = "./drive/MyDrive/hackathon/eval_decoded.json"
test_manifest = "./drive/MyDrive/hackathon/test_decoded.json"

train_data = read_manifest(train_manifest)
eval_data = read_manifest(eval_manifest)
test_data = read_manifest(test_manifest)

# #Apply preprocessing

train_data_processed = apply_preprocessors(train_data, PREPROCESSORS)
eval_data_processed = apply_preprocessors(eval_data, PREPROCESSORS)
test_data_processed = apply_preprocessors(test_data, PREPROCESSORS)

## Write new manifests

train_manifest_cleaned = write_processed_manifest(train_data_processed, train_manifest)
eval_manifest_cleaned = write_processed_manifest(eval_data_processed, eval_manifest)
test_manifest_cleaned = write_processed_manifest(test_data_processed, test_manifest)

</code>

The above piece of code will remove special characters that appear  in swahili example

```
a             ā, ã
e             ē
i             ī, i̇
o             ō
u             ū
c             č
g             ğ
k             ǩ
s             š
t             ŧ
y             ȳ
```

The next  code below is tokenizing  our dataset to improve quality and speed for longer units.

<code>

# #Tokenizers

!python  drive/MyDrive/hackathon/scripts/process_asr_text_tokenizer.py \
  --manifest=drive/MyDrive/hackathon/eval_decoded_processed.json,drive/MyDrive/hackathon/train_decoded_processed.json \
  --vocab_size=1024 \
  --data_root=./drive/MyDrive/hackathon/ \
  --tokenizer="spe" \
  --spe_type=bpe \
  --spe_character_coverage=1.0 \
  --spe_max_sentencepiece_length=4 \
  --log

</code>

Now  we will create tarrad datasets  to speed up the training i created one tarred set

<code>

!python drive/MyDrive/hackathon/scripts/convert_to_tarred_audio_dataset.py \
  --manifest_path=drive/MyDrive/hackathon/train_decoded_processed.json \
  --target_dir=drive/MyDrive/hackathon/train_tarred_1bk \
  --num_shards=1024 \
  --max_duration=11.0 \
  --min_duration=1.0 \
  --shuffle \
  --shuffle_seed=1 \
  --sort_in_shards \
  --workers=-1

</code>

with that done what we remaining is training our  model below chunk of code has configuration required to traind an asr model from scratch

<code>

!python drive/MyDrive/hackathon/scripts/speech_to_text_ctc_bpe.py \
--config-path=../conf/conformer/ \
--config-name=conformer_ctc_bpe \
exp_manager.name="SwahiliModelTraining1" \
exp_manager.resume_if_exists=true \
exp_manager.resume_ignore_no_checkpoint=true \
exp_manager.exp_dir=drive/MyDrive/hackathon/ \
model.tokenizer.dir=drive/MyDrive/hackathon/tokenizer_spe_bpe_v1024_max_4 \
model.train_ds.is_tarred=true \
model.train_ds.tarred_audio_filepaths=drive/MyDrive/hackathon/train_tarred_1bk/audio__OP_0..1023_CL_.tar \
model.train_ds.manifest_filepath=drive/MyDrive/hackathon/train_tarred_1bk/tarred_audio_manifest.json \
model.validation_ds.manifest_filepath=/content/drive/MyDrive/hackathon/eval_decoded_processed.json \
model.test_ds.manifest_filepath=drive/MyDrive/hackathon/test_decoded_processed_v2.json

</code>

The above  code has all required configs  required .Now  we just  need to  to train our model after successfull run it will save itself
as swahilimodel1.nemo  this is our model now and we can keep  training the model to perfection.

The following code is for transcribind our  test and eval  data

<code>

##### this  one is for  test  data

!python drive/MyDrive/hackathon/scripts/transcribe_speech.py \
  model_path=/content/drive/MyDrive/hackathon/SwahiliModelTraining1/checkpoints/SwahiliModelTraining1.nemo \
  dataset_manifest=drive/MyDrive/hackathon/test_decoded_processed_v2.json \
  output_filename=drive/MyDrive/hackathon/results/test_with_predictions_4.json \
  batch_size=16 \
  cuda=0 \
  amp=True

!python drive/MyDrive/hackathon/scripts/transcribe_speech.py \
  model_path=/content/drive/MyDrive/hackathon/SwahiliModelTraining1/checkpoints/SwahiliModelTraining1.nemo \
  dataset_manifest=drive/MyDrive/hackathon/Prime_eval.json \
  output_filename=drive/MyDrive/hackathon/results/eval_version1.json \
  batch_size=16 \
  cuda=0 \
  amp=True

</code>

This  will produce two json files  <b>test_with_predictions_4.json</b> and <b>eval_version.json</b> respectively the remain the remaining work is just to combine the json files and extract the path and pred_text using  the following code

<code>

import os
import pandas as pd
import jsonlines
import csv

input_files = ["eval_version1.json","test_with_predictions_4.json",]

file=r'drive/MyDrive/hackathon/results/final_data.json'

def combine_jsonl(input_files, output_file):
    with jsonlines.open(output_file, 'w') as writer:
        for input_file in input_files:
            file=f'drive/MyDrive/hackathon/results/{input_file}'
            with jsonlines.open(file) as reader:
                for line in reader:
                    writer.write(line)

combine_jsonl(input_files, file)

JsonPath =r'drive/MyDrive/hackathon/results/final_data.json'

folder_path = r'drive/MyDrive/hackathon/results/'

csvName = "eval_with_predictions.json"

data_list = {}

sentence = []  # This list will store the processed data

path = []  # This list will store the processed data

with jsonlines.open(JsonPath) as reader:

    for obj in reader:

        file_path = os.path.basename(obj['audio_filepath'])

        file_name = os.path.splitext[file_path](0) + ".mp3"

        path.append(file_name)  # the non-empty row to the output_data list

        sentence.append(0 if not obj['pred_text'].strip() else obj['pred_text'])# the non-empty row to the output_data list

SampeData = {
    "path":path,
    "sentence":sentence,

}

df = pd.DataFrame(SampeData)

folder_path = r'drive/MyDrive/hackathon/results/'

df.to_csv(os.path.join(folder_path,'final_data.csv'),index=False)

def remove_duplicates(csv_path, column_name):

    if not os.path.exists(csv_path):

        print(f"File not found: {csv_path}")

        return

    df = pd.read_csv(csv_path)

    df_no_duplicates = df.drop_duplicates(subset=[column_name])

    df_no_duplicates.to_csv(csv_path, index=False)
    
    print(f"Duplicates removed from {csv_path}")

remove_duplicates(csv_path, column_to_check)
</code>


Lastly  want to thank zindi and mozilla for creating this challenge it was a great challenge  since was my first step  in Machine learning learned a great  deal Also thanks for Africastalking for  host the event.

Gracias.Thank You.