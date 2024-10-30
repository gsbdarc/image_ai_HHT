"""
Script containing helper functions for GPT-based image analysis notebooks.

This script includes functions to:
- Extract the ground truth year from car class names.
- Extract predicted years from GPT responses.
- Compare ground truth years with predicted years to compute accuracy.
- Prepare test and prompt datasets for zero-shot and few-shot GPT examples.
- Plot per-model accuracy based on the predictions.
- Display images used in few shot prompt.

These functions assist in processing and evaluating GPT model outputs in the context
of image analysis of car models and years. They are designed to facilitate the preparation
of data, computation of accuracy metrics, and visualization of results for analysis.

Author: Natalya Rapstine
Email: nrapstin@stanford.edu
Date: Oct. 30, 2024
"""

import pandas as pd
import os
import re
import random 
import matplotlib.pyplot as plt
import PIL

random.seed(42)

proj_dir = f'/zfs/projects/darc/nrapstin_hht_image_ai'
image_dir = f"{proj_dir}/stanford-cars/data/images"

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

def compute_accuracy(df_test):
    """
    Computes the accuracy of the model predictions by comparing the ground truth years
    with the predicted years extracted from the GPT responses.

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

    # print(f'similar_cars_df.shape: {similar_cars_df.shape}')
    # print(f'similar_cars_df.head: {similar_cars_df.head()}')

    # Specify the folder path
    data_path = f"{proj_dir}/stanford-cars/gpt/data"
    results_path = f"{proj_dir}/stanford-cars/gpt/results"
    
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

    # save test df
    test_df.to_csv(f"{data_path}/test-df.csv", index=False)
    # save prompt images
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