"""
GPT Image Processing Utilities

Author: Natalya Rapstine
Email: nrapstin@stanford.edu
Date: Nov. 1, 2024

Description:
This script contains a collection of utility functions for processing images and preparing data for fine-tuning GPT models, specifically for tasks like identifying the year of cars in images. The functions cover various steps such as encoding images, extracting years from class names and responses, computing accuracy, preparing datasets, and more.

"""

import pandas as pd
import os
import re
import ast
import base64
import random
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import PIL
from PIL import Image

random.seed(42)

image_dir = f'/scratch/shared/image_ai_HHT_data/images'
BATCH_FILE = 'data/batches.json'

def encode_image(image_path):
    """
    Encodes an image file to a base64 string.

    Parameters:
    - image_path (str): The file path to the image that needs to be encoded.

    Returns:
    - str: A base64 encoded string representation of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_image_objects(image_paths):
    """
    Generate image objects for the system prompt.
    """
    return [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}} for image_path in image_paths]

def extract_ground_truth_year(class_name):
    """
    Extracts the first four-digit year from the class name string.

    Parameters:
        class_name (str): The class name string containing the year.

    Returns:
        int or None: The extracted year as an integer if found; otherwise, None.
    """
    # Find the first four-digit year in the string
    match = re.search(r'\b((?:19|20)\d{2})\b', class_name)
    
    # If a match is found, return it as an integer; otherwise, return None
    return int(match.group(0)) if match else None

def extract_predicted_years(response):
    """
    Extracts all four-digit years and year ranges from a given response string.

    Parameters:
        response (str): The response string potentially containing years and year ranges.

    Returns:
        list: A sorted list of unique years extracted from the response.
    """
    years = re.findall(r'\b((?:19|20)\d{2})\b', response)
    years = [int(year) for year in years]
    
    # Find year ranges like "2012-2014", "2012 to 2014", or "between 2012 and 2014"
    ranges = re.findall(r'((?:19|20)\d{2})\s*(?:–|-|to|and)\s*((?:19|20)\d{2})', response)
    for start, end in ranges:
        start_year = int(start)
        end_year = int(end)
        # Extend the list with the full range of years
        years.extend(range(start_year, end_year + 1))
    
    # Remove duplicates and sort the list
    years = sorted(set(years))
    return years
    
def is_correct_prediction(ground_truth_year, predicted_years):
    """
    Checks if the ground truth year is present in the list of predicted years.

    Parameters:
        ground_truth_year (int): The actual year of the car.
        predicted_years (list): A list of years predicted by the model.

    Returns:
        bool: True if the ground truth year is in the predicted years; False otherwise.
    """
    return ground_truth_year in predicted_years

def is_prediction_correct(df_test):
    """
    Computes how the model prediction extracted from the GPT responses compares to the ground truth years.

    Parameters:
        df_test (pd.DataFrame): The test DataFrame containing 'GPT_Response' and 'Ground Truth Year'.

    Returns:
        pd.DataFrame: The updated DataFrame with 'Predicted Years' and 'Correct Prediction' columns added.
    """
    df_test['Predicted Years'] = df_test['GPT_Response'].apply(extract_predicted_years)
    
    df_test['Correct Prediction'] = df_test.apply(lambda x: is_correct_prediction(x['Ground Truth Year'], x['Predicted Years']), axis=1)


    return df_test

def plot_per_model_accuracy(df_test):
    """
    Generates a per-model accuracy plot and prints a confusion matrix-like table showing
    the number of correct and incorrect predictions for each car model.

    Parameters:
        df_test (pd.DataFrame): The test DataFrame containing 'Class Name' and 'Correct Prediction'.

    Returns:
        None
        
    """
    # Group by 'Class Name' and 'Correct Prediction' and count the occurrences
    confusion = df_test.groupby(['Class Name', 'Correct Prediction']).size().reset_index(name='Count')
    
    # Pivot the table to have 'Correct Prediction' as columns
    confusion_pivot = confusion.pivot(index='Class Name', columns='Correct Prediction', values='Count').fillna(0)
    
    # Ensure columns are ordered as False, True
    confusion_pivot = confusion_pivot.reindex(columns=[False, True], fill_value=0)
    
    # Print the confusion matrix-like table
    # print(confusion_pivot)

    # Compute total predictions per class
    confusion_pivot['Total'] = confusion_pivot[False] + confusion_pivot[True]
    
    # Compute accuracy per class
    confusion_pivot['Accuracy'] = confusion_pivot[True] / confusion_pivot['Total']
    
    # Display the confusion matrix with accuracy
    print(confusion_pivot[[False, True, 'Total', 'Accuracy']])

    # Reset index to make 'Class Name' a column
    confusion_pivot_reset = confusion_pivot.reset_index()
    
    # Plot per-class accuracy
    plt.figure(figsize=(5, 5))
    plt.bar(confusion_pivot_reset['Class Name'], confusion_pivot_reset['Accuracy'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.title('Per-Model Accuracy')
    plt.tight_layout()
    plt.show()
    

def make_test_data(df):
    """
    Prepares test and prompt datasets for GPT zero-shot and few-shot examples by selecting
    specific car models and splitting their images into prompt and test sets.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the full dataset with at least 'Class Name' and 'image' columns.

    Returns:
        None: The function saves 'test-df.csv' and 'prompt-df.csv' to the specified directories.
    """
    df['Ground Truth Year'] = df['Class Name'].apply(extract_ground_truth_year)

    # Define the years we are interested in
    years = [2009, 2010, 2011, 2012]
    
    # Filter the dataframe for the specified years
    filtered_df = df[df['Ground Truth Year'].isin(years)]

    selected_classes = [
    'Dodge Charger SRT-8 2009',
    'Chevrolet Malibu Hybrid Sedan 2010',
    'Audi S6 Sedan 2011',
    'Dodge Charger Sedan 2012'
    ]

    # Filter the DataFrame to include only the selected classes
    similar_cars_df = df[df['Class Name'].isin(selected_classes)].reset_index(drop=True)

    # Specify the folder path
    data_path = "data"
    results_path = "results"
    
    # Create the folder if it does not exist
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    
    # pick 5 images per class
    # Initialize empty dataframes for prompt and testing
    prompt_df = pd.DataFrame()
    test_df = pd.DataFrame()
    
    # For each class, select 10 examples, then split into 5 for prompt and 5 for test
    for class_label in selected_classes:
        class_subset = similar_cars_df[similar_cars_df['Class Name'] == class_label]
        sampled_images = class_subset.sample(n=10, random_state=42)['image']  # Sample 10 images
        
        prompt_images = sampled_images.iloc[:5]  # First 5 images for prompt
        test_images = sampled_images.iloc[5:]    # Next 5 images for testing
        
        prompt_examples = class_subset[class_subset['image'].isin(prompt_images)]
        test_examples = class_subset[class_subset['image'].isin(test_images)]
        
        prompt_df = pd.concat([prompt_df, prompt_examples], ignore_index=True)
        test_df = pd.concat([test_df, test_examples], ignore_index=True)

    # Shuffle the rows of the DataFrame
    test_df = test_df.sample(frac=1).reset_index(drop=True)

    # Save test df
    test_df.to_csv(f"{data_path}/test-df.csv", index=False)
    # Save prompt images
    prompt_df.to_csv(f"{data_path}/prompt-df.csv", index=False)


def show_prompt_images(prompt_df):
    """
    Displays images from the prompt DataFrame in a grid layout.

    Parameters:
        prompt_df (pd.DataFrame): DataFrame containing image information.
            Expected to have at least an 'image' column with image file names
            and a 'Class Name' column for labeling.

    Returns:
        None

    This function creates a grid of images using matplotlib to visualize the images
    specified in 'prompt_df'. It calculates the number of rows based on the total
    number of images and a fixed number of images per row. It handles missing images
    by printing a warning message and continues displaying the rest. Each image is
    displayed without axis ticks and is titled with its corresponding 'Class Name'.
    """
    # Number of images per row
    images_per_row = 5
    
    # Calculate the number of rows needed
    num_images = len(prompt_df)
    num_rows = num_images // images_per_row + int(num_images % images_per_row > 0)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(15, 3 * num_rows))
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    
    for ax in axes[num_images:]:
        # Hide any unused subplots
        ax.axis('off')
    
    # Iterate over the DataFrame and display images
    for idx, (index, row) in enumerate(prompt_df.iterrows()):
        image_path = os.path.join(image_dir, row['image'])
        if not os.path.isfile(image_path):
            print(f"Image file not found: {image_path}")
            continue
        img = PIL.Image.open(image_path)
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(row['Class Name'], fontsize=10)
    
    plt.tight_layout()
    plt.show()


def check_and_convert_image_modes(image_paths):
    """
    Checks a list of image paths to identify images that are not in RGB or RGBA mode,
    and converts them to RGB format if necessary.

    This function iterates over a list of image file paths, opens each image,
    checks its mode, and if the image mode is not 'RGB' or 'RGBA', it converts
    the image to 'RGB' mode and saves it back to the same path. This is useful
    for datasets where images are expected to be in RGB format, such as when
    fine-tuning GPT-4o.

    Parameters:
    - image_paths (list of str): A list of file paths to the images to be checked and converted.

    Returns:
    - None
    """
    for image_path in image_paths:
        try:
            with Image.open(image_path) as img:
                if img.mode not in ('RGB', 'RGBA'):
                    print(f"Image {image_path} is in mode {img.mode}. Converting to RGB.")
                    rgb_img = img.convert('RGB')
                    rgb_img.save(image_path)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")



import os
import random

def prepare_ft_data(df):
    """
    Prepares data for fine-tuning by splitting the dataset into training, validation, and test sets,
    converting images to RGB format if necessary, and creating JSONL files for training.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing image data and class labels.

    Returns:
    - None
    """
    # Extract 'Ground Truth Year' from 'Class Name'
    df['Ground Truth Year'] = df['Class Name'].apply(extract_ground_truth_year)
    classes = df['Ground Truth Year'].unique()
    
    # Initialize dictionaries for train, val, test data
    train_data = {}
    val_data = {}
    test_data = {}
    
    # Set the number of images per split
    train_count = 15
    val_count = 10
    test_count = 10

    for cls in classes:
        cls_df = df[df['Ground Truth Year'] == cls]
        image_paths = cls_df['image_path'].tolist()
        num_images = len(image_paths)
    
        # Check if we have enough images
        total_required = train_count + val_count + test_count
        if num_images < total_required:
            print(f"Warning: Not enough images for class {cls}. Required: {total_required}, Available: {num_images}")
            # Adjust counts proportionally
            ratio = num_images / total_required
            adjusted_train_count = max(1, int(train_count * ratio))
            adjusted_val_count = max(1, int(val_count * ratio))
            adjusted_test_count = num_images - adjusted_train_count - adjusted_val_count
        else:
            adjusted_train_count = train_count
            adjusted_val_count = val_count
            adjusted_test_count = test_count
    
        # Shuffle the images
        random.shuffle(image_paths)
    
        # Assign images to sets
        train_images = image_paths[:adjusted_train_count]
        val_images = image_paths[adjusted_train_count:adjusted_train_count + adjusted_val_count]
        test_images = image_paths[adjusted_train_count + adjusted_val_count:adjusted_train_count + adjusted_val_count + adjusted_test_count]
    
        # Store in dictionaries
        train_data[cls] = train_images
        val_data[cls] = val_images
        test_data[cls] = test_images

    # Define output directories and file paths
    fine_tune_dir = 'data/fine-tune'
    train_jsonl_path = f'{fine_tune_dir}/train.jsonl'
    val_jsonl_path = f'{fine_tune_dir}/val.jsonl'
    train_image_paths_file = f'{fine_tune_dir}/train_image_paths.txt'
    val_image_paths_file = f'{fine_tune_dir}/val_image_paths.txt'
    test_image_paths_file = f'{fine_tune_dir}/test_image_paths.txt'

    # Create directories if they do not exist
    os.makedirs(fine_tune_dir, exist_ok=True)
    
    # Collect and write train image paths
    train_image_paths = []
    for cls, paths in train_data.items():
        for path in paths:
            if image_dir:
                full_path = os.path.join(image_dir, path)
            else:
                full_path = path  # Assuming 'image_path' already contains full paths
            train_image_paths.append(full_path)
    
    with open(train_image_paths_file, 'w') as f:
        for path in train_image_paths:
            f.write(f"{path}\n")

    # Collect and write validation image paths
    val_image_paths = []
    for cls, paths in val_data.items():
        for path in paths:
            if image_dir:
                full_path = os.path.join(image_dir, path)
            else:
                full_path = path
            val_image_paths.append(full_path)
    
    with open(val_image_paths_file, 'w') as f:
        for path in val_image_paths:
            f.write(f"{path}\n")

    # Collect and write test image paths
    test_image_paths = []
    for cls, paths in test_data.items():
        for path in paths:
            if image_dir:
                full_path = os.path.join(image_dir, path)
            else:
                full_path = path
            test_image_paths.append(full_path)
    
    with open(test_image_paths_file, 'w') as f:
        for path in test_image_paths:
            f.write(f"{path}\n")

    # Check if JSONL files already exist
    if os.path.exists(train_jsonl_path) and os.path.exists(val_jsonl_path):
        print("JSONL files already exist. Skipping creation.")
    else:
        # Process training images
        # Check and convert non-RGB images in training data
        check_and_convert_image_modes(train_image_paths)
        
        # Create JSONL file after conversion
        create_jsonl(train_data, train_jsonl_path, 'data/fine-tune/train_image_paths.txt', image_dir)

        # Process validation images
        # Check and convert non-RGB images in validation data
        check_and_convert_image_modes(val_image_paths)
        
        create_jsonl(val_data, val_jsonl_path, 'data/fine-tune/val_image_paths.txt', image_dir)

    # Process test images
    # Check and convert non-RGB images in test data
    check_and_convert_image_modes(test_image_paths)



def create_jsonl(selected_data, jsonl_path, image_paths_file, image_dir):
    """
    Creates a JSONL file from the selected data for fine-tuning GPT models and saves image paths.

    This function processes images specified in `selected_data`, encodes them in base64 format,
    constructs JSON objects suitable for fine-tuning GPT models, and writes them to a JSONL file.
    It also saves the paths of the images used in a separate file.

    Parameters:
    - selected_data (dict): A dictionary where keys are class labels and values are lists of image paths.
    - jsonl_path (str): The file path where the JSONL output will be saved.
    - image_paths_file (str): The file path where the list of image paths used will be saved.
    - image_dir (str): The directory where the images are located.

    Returns:
    - None
    """
    examples = []
    all_image_paths = []

    for cls, paths in selected_data.items():
        # Shuffle paths to ensure random order
        random.shuffle(paths)
        for path in paths:
            try:
                full_path = os.path.join(image_dir, path)
                if not os.path.exists(full_path):
                    print(f"Image file not found: {full_path}. Skipping.")
                    continue
                encoded_image = encode_image(full_path)
                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                }
                all_image_paths.append(full_path)

                # Create the example with one image per example
                example = {
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
                        },
                        {
                            "role": "assistant",
                            "content": str(cls)
                        }
                    ]
                }
                examples.append(example)

            except Exception as e:
                print(f"Error processing image {path}: {e}. Skipping this image.")
                continue

    # Write to JSONL file
    with open(jsonl_path, 'w') as f:
        for example in tqdm(examples, desc=f"Writing to {jsonl_path}"):
            f.write(json.dumps(example) + '\n')
    print(f"Created {jsonl_path} with {len(examples)} examples.")

    # Save image paths used
    with open(image_paths_file, 'w') as f:
        for path in all_image_paths:
            f.write(f"{path}\n")
    print(f"Saved image paths to {image_paths_file}")

def save_batch_id(batch_id, key):
    """
    Save the batch ID to the JSON file under the specified key.
    
    Args:
        batch_id (str): The batch ID to be saved.
        key (str): The key under which to save the batch ID.
    """
    # Load existing batch data if the file exists
    if os.path.exists(BATCH_FILE):
        with open(BATCH_FILE, 'r') as f:
            batch_data = json.load(f)
    else:
        batch_data = {}

    # Update the batch data with the new batch ID
    batch_data[key] = batch_id

    # Save the updated data back to the file
    with open(BATCH_FILE, 'w') as f:
        json.dump(batch_data, f, indent=4)

    print(f"Saved batch ID '{batch_id}' under key '{key}'.")

def load_batch_id(key):
    """
    Load the batch ID from the JSON file using the specified key.
    
    Args:
        key (str): The key under which the batch ID is saved.
    
    Returns:
        str: The batch ID if found, otherwise None.
    """
    if os.path.exists(BATCH_FILE):
        with open(BATCH_FILE, 'r') as f:
            batch_data = json.load(f)
        return batch_data.get(key, None)
    else:
        print(f"Batch file '{BATCH_FILE}' not found.")
        return None

def post_process(df_responses_base_model):
    """
    Processes a DataFrame containing model responses to extract content and years.

    This function performs post-processing on a DataFrame that contains the raw JSON responses from a GPT model. It extracts the 'content' field from each response, parses it to retrieve the textual content, and then extracts years mentioned in that content using the `extract_years_from_content` function. The extracted content and years are added as new columns to the DataFrame. 

    Parameters:
    - df_responses_base_model (pd.DataFrame): A DataFrame containing model responses with a 'response' column, where each entry is a JSON string representing the model's response.

    Returns:
    - pd.DataFrame: The updated DataFrame with additional columns:
        - 'content': The extracted textual content from the model's response.
        - 'extracted_years': A list of years extracted from the content.
    """
    # Initialize a list to store extracted content
    contents = []
    
    for index, row in df_responses_base_model.iterrows():
        json_str = row['response']
    
        try:
            # Remove the starting and ending double quotes if present
            if json_str.startswith('"') and json_str.endswith('"'):
                json_str = json_str[1:-1]
    
            # Safely evaluate the string as a Python literal
            response_dict = ast.literal_eval(json_str)
            # Navigate to the 'content' field
            content = response_dict['body']['choices'][0]['message']['content']
            # Store the content
            contents.append(content)
        except Exception as e:
            # In case of parsing error, append None
            contents.append(None)
            print(f"Error parsing row {index}: {e}")
    
    # Add the extracted content to the DataFrame
    df_responses_base_model['content'] = contents

    extracted_years_list = []

    for content in df_responses_base_model['content'].to_list():
        if pd.notnull(content):
            years = extract_years_from_content(content)
            extracted_years_list.append(years)
        else:
            extracted_years_list.append(None)
    
    df_responses_base_model['extracted_years'] = extracted_years_list

    df_responses_base_model.reset_index(inplace=True)
    df_responses_base_model['index'] = df_responses_base_model['index'].astype(int)

    return df_responses_base_model

def extract_years_from_content(content):
    """
    Extracts all four-digit years and year ranges from a given text content.

    This function scans the input text for patterns that represent years (e.g., "2012")
    and year ranges (e.g., "2010-2012", "from 2010 to 2012", "between 2010 and 2012").
    It accounts for various ways years and ranges might be mentioned in the text,
    including different delimiters and phrasing.

    Parameters:
    - content (str): The text content from which to extract years and year ranges.

    Returns:
    - years_list (list of str): A sorted list of unique years (as strings) extracted from the content.

    Notes:
    - The function handles years in the range from 1900 to 2099.
    - For year ranges, all years within the range are included in the output list.
    - Duplicate years are removed, and the final list is sorted in ascending order.
    - The function is useful for parsing textual responses where years are mentioned
      in various formats, ensuring comprehensive extraction of relevant years.

    """
    years_list = []

    range_patterns = [
        r'from\s+(\b(?:19|20)\d{2})\s+to\s+(\b(?:19|20)\d{2})',
        r'between\s+(\b(?:19|20)\d{2})\s+and\s+(\b(?:19|20)\d{2})',
        r'produced\s+(?:from|between)\s+(\b(?:19|20)\d{2})\s+(?:to|and)\s+(\b(?:19|20)\d{2})',
        r'\b(?:model\s+years?|years?)\s+(\b(?:19|20)\d{2})\s*(?:-|to|–|—)\s*(\b(?:19|20)\d{2})',
        r'(\b(?:19|20)\d{2})\s*-\s*(\b(?:19|20)\d{2})',
        r'around\s+(\b(?:19|20)\d{2})\s*(?:-|to|–|—)?\s*(\b(?:19|20)\d{2})?',
    ]

    for pattern in range_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            years_in_match = [year for year in match if year]
            if len(years_in_match) == 2:
                start_year = int(years_in_match[0])
                end_year = int(years_in_match[1])
                if start_year > end_year:
                    start_year, end_year = end_year, start_year
                years = [str(year) for year in range(start_year, end_year + 1)]
                years_list.extend(years)
            elif len(years_in_match) == 1:
                years_list.append(years_in_match[0])

    single_year_pattern = r'\b(?:19|20)\d{2}\b'
    single_years = re.findall(single_year_pattern, content)
    years_list.extend(single_years)

    years_list = list(set(years_list))
    years_list.sort()

    return years_list