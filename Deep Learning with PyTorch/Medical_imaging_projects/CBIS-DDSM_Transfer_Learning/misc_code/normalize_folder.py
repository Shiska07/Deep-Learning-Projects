import os
import numpy as np
from PIL import Image

def normalize_image(image):
    image_array = np.array(image, dtype=np.float32)
    min_val = np.min(image_array)
    max_val = np.max(image_array)

    if min_val == max_val:
        # Handle the case where the range is zero
        normalized_image = np.full_like(image_array, 0.5)
    else:
        normalized_image = (image_array - min_val) / (max_val - min_val)

    return normalized_image

def calculate_dataset_statistics(folder_path):
    mean_list = []
    std_list = []

    # Iterate through images in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)

            # Normalize the image and calculate mean and std
            normalized_image = normalize_image(image)
            mean_list.append(np.mean(normalized_image))
            std_list.append(np.std(normalized_image))

    # Calculate mean and std of the entire dataset
    dataset_mean = np.mean(np.asarray(mean_list))
    dataset_std = np.mean(np.asarray(std_list))

    # print std and mean
    print(f'Mean: {dataset_mean}')
    print(f'std: {dataset_std}')

    return dataset_mean, dataset_std

def normalize_and_save_images(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Calculate mean and std of the entire dataset
    dataset_mean, dataset_std = calculate_dataset_statistics(input_folder)

    # Normalize and save each image
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open the image
            image = Image.open(image_path)

            # Normalize the image
            normalized_image = normalize_image(image)

            # remove brightness and contrast variations
            final_image = (normalized_image - dataset_mean)/dataset_std


            scaled_image = (normalize_image(final_image)* 255).astype(np.uint8)


            # Save the normalized image
            Image.fromarray(scaled_image).save(output_path)



if __name__ == "__main__":
    # Replace 'input_folder' and 'output_folder' with your actual folder paths
    input_folder = 'data/CBIS-DDSM/patch_11_29_smallcrp_all/images/train/BENIGN_Mass'
    output_folder = 'data/CBIS-DDSM/patch_11_29_smallcrp_all/images/train/BRNIGN_Mass_norm'

    normalize_and_save_images(input_folder, output_folder)