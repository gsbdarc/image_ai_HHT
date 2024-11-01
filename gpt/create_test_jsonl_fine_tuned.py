"""
Fine-Tuned Model Test Data Preparation Script

Author: Natalya Rapstine
Email: nrapstin@stanford.edu
Date: Nov. 1, 2024

Description:
This script prepares test data for evaluating a fine-tuned GPT-based model on car images.
It processes a set of car images, encodes them in base64, constructs JSONL entries
for API requests, and saves ground truth labels for evaluation.
"""

import os
import re
import json
import base64
from tqdm import tqdm
import pandas as pd
import gpt_utils

proj_dir = f'/zfs/projects/darc/nrapstin_hht_image_ai'

# Define file paths and constants
TEST_IMAGE_PATHS_FILE = f'{proj_dir}/stanford-cars/gpt/data/fine-tune/test_image_paths.txt'
TEST_JSONL = f'{proj_dir}/stanford-cars/gpt/data/fine-tune/test_ft_model.jsonl'
TEST_LABELS_FILE = f'{proj_dir}/stanford-cars/gpt/data/fine-tune/test_labels.json'
MODEL = "ft:gpt-4o-2024-08-06:gsb-darc-team::AOAURXbh" 
IMAGE_DIR = f'{proj_dir}/stanford-cars/data/images'
data_file = f"{proj_dir}/stanford-cars/data/train.csv"

def main():
    """
    Main function to prepare test data for evaluating the fine-tuned model.

    Steps:
    - Load the dataset to create a mapping from image filenames to labels.
    - Read test image paths from the specified file.
    - Process each image:
        - Retrieve the label for the image.
        - Encode the image in base64 format.
        - Construct the JSON object for the API request.
    - Write the JSONL file for test inputs.
    - Save the ground truth labels to a JSON file for evaluation.
    """
    # Load the dataset to create a mapping from image filenames to labels
    df = pd.read_csv(data_file, usecols=['image', 'Class', 'Class Name'])
    df['Ground Truth Year'] = df['Class Name'].apply(gpt_utils.extract_ground_truth_year)
    image_to_label = dict(zip(df['image'], df['Ground Truth Year']))

    # Read test image paths
    with open(TEST_IMAGE_PATHS_FILE, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]

    labels = {}
    examples = []

    # Process each image
    for idx, relative_path in enumerate(tqdm(image_paths, desc="Creating test JSONL entries")):
        try:
            full_path = os.path.join(IMAGE_DIR, relative_path)
            filename = os.path.basename(relative_path)

            # Get the label for the image
            label = image_to_label.get(filename)
            if label is None:
                print(f"Label not found for image {filename}. Skipping.")
                continue

            # Encode the image
            encoded_image = gpt_utils.encode_image(full_path)
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                }
            }

            # Save the ground truth label with the index as the key
            labels[str(idx)] = str(label)  # Use string keys for JSON compatibility

            image_id = os.path.splitext(filename)[0]

            # Construct the JSON object for the API request
            example = {
                "custom_id": image_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an assistant that identifies the year of cars in images."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "What is the year of the car in this image?"
                                },
                                image_content
                            ]
                        }
                    ]
                    # Include any additional parameters if necessary
                }
            }

            # Append the example to the list
            examples.append(example)

        except Exception as e:
            print(f"Error processing image {relative_path}: {e}")
            continue

    # Write the JSONL file for test inputs
    with open(TEST_JSONL, 'w') as f:
        for example in examples:
            json_line = json.dumps(example)
            f.write(json_line + '\n')
    print(f"Created {TEST_JSONL} with {len(examples)} examples.")

    # Save the ground truth labels to a JSON file for evaluation
    with open(TEST_LABELS_FILE, 'w') as f:
        json.dump(labels, f)
    print(f"Saved labels to {TEST_LABELS_FILE}")

if __name__ == "__main__":
    main()