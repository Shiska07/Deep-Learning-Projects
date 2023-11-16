import os
import csv
import shutil

# Function to create the annotations CSV file
def create_annotations_csv(root_folder, label_map, csv_path):

    with open(csv_path, 'w', newline='') as csv_file:
        fieldnames = ['ImageName', 'Label']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for class_folder in os.listdir(root_folder):
            class_path = os.path.join(root_folder, class_folder)

            # Check if the item in the directory is a folder
            if os.path.isdir(class_path):
                label = label_map.get(class_folder, None)

                # Process images in the class folder
                for image_name in os.listdir(class_path):
                    # Assuming images have common formats like JPG or PNG
                    if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        writer.writerow({'ImageName': image_name, 'Label': label})

        print(f'Successfully created annotations.csv file at {csv_path}.\n')


def move_images_to_single_directory(source_path, destination_path):

    dest_folder = os.path.join(destination_path, 'allimages')

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Iterate over each class folder in the source path
    for class_folder in os.listdir(source_path):
        class_path = os.path.join(source_path, class_folder)

        # Check if the item in the directory is a folder
        if os.path.isdir(class_path):
            # Process images in the class folder
            for image_name in os.listdir(class_path):
                # Assuming images have common formats like JPG or PNG
                if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    source_image_path = os.path.join(class_path, image_name)
                    destination_image_path = os.path.join(dest_folder, image_name)

                    # Move the image to the destination directory
                    shutil.move(source_image_path, destination_image_path)

    print(f'Successfully moved all images from {source_path} to {dest_folder}.\n')



# data folder containing images in separate class folders
data_root = 'data/CBIS-DDSM/processed/aug_cropped/images/test'

# destination directory
save_dir = 'data/CBIS-DDSM/processed/aug_cropped/images/test_attr'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Dictionary containing class names as keys and label maps as values
class_label_map = {'BACKGROUND': 0, 'BENIGN_Mass': 1, 'MALIGNANT_Mass': 2}

# Output CSV file path
csv_file_path = os.path.join(save_dir, 'annotations.csv')

# Call the function to create the annotations CSV
create_annotations_csv(data_root, class_label_map, csv_file_path)
move_images_to_single_directory(data_root, save_dir)
