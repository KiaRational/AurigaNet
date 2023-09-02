from tqdm import tqdm
import os
import sys 
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from PreProcess.Dataloader import LabelGenerator , DataLoaderX

def main():
    intro  = '''
    _____        _                 _     _           _          _    _____                           _             
    |  __ \      | |               | |   | |         | |        | |  / ____|                         | |            
    | |  | | __ _| |_ __ _ ___  ___| |_  | |     __ _| |__   ___| | | |  __  ___ _ __   ___ _ __ __ _| |_ ___  _ __ 
    | |  | |/ _` | __/ _` / __|/ _ \ __| | |    / _` | '_ \ / _ \ | | | |_ |/ _ \ '_ \ / _ \ '__/ _` | __/ _ \| '__|
    | |__| | (_| | || (_| \__ \  __/ |_  | |___| (_| | |_) |  __/ | | |__| |  __/ | | |  __/ | | (_| | || (_) | |   
    |_____/ \__,_|\__\__,_|___/\___|\__| |______\__,_|_.__/ \___|_|  \_____|\___|_| |_|\___|_|  \__,_|\__\___/|_|   
                                                                                                                    
                                                                                
        '''

    print(intro)

    # Usage
    data_path = "/home/kia/BDD100K/bdd100k_images_100k_5/bdd100k/images/100k/train/"
    label_path = '/home/kia/BDD100K/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json'
    save_path = '/home/kia/BDD100K/Generated/'

    class_mapping = {
        'traffic_light': 0,
        'traffic_sign': 1,
        'car': 2
    }

    dataset = LabelGenerator(data_path, label_path, save_path ,image_size=(640, 640), normalize=True, class_mapping=class_mapping)
    batch_size = 1
    train_data_loader = DataLoaderX(dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=1)
    # Create subfolders if they don't exist
    save_dir = save_path
    image_save_dir = os.path.join(save_dir, "images")
    mask_save_dir = os.path.join(save_dir, "masks")

    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    for i, (image, cluster_mask, instance_mask , objects_annotations , image_name) in enumerate(tqdm(train_data_loader)):
        # image_save_name = str(image_name[0][:-4]) + ".npy"
        # cluster_mask_save_name = str(image_name[0][:-4]) + "_cluster_mask.npy"
        # instance_mask_save_name = str(image_name[0][:-4]) + "_instance_mask.npy"

        # image_np = image.numpy()
        # cluster_mask_np = cluster_mask.numpy()
        # instance_mask_np = instance_mask.numpy()

        # image_path = os.path.join(image_save_dir, image_save_name)
        # cluster_mask_path = os.path.join(mask_save_dir, cluster_mask_save_name)
        # instance_mask_path = os.path.join(mask_save_dir, instance_mask_save_name)

        # # Create memory-mapped arrays
        # image_mm = np.memmap(image_path, dtype=image_np.dtype, mode='w+', shape=image_np.shape)
        # cluster_mask_mm = np.memmap(cluster_mask_path, dtype=cluster_mask_np.dtype, mode='w+', shape=cluster_mask_np.shape)
        # instance_mask_mm = np.memmap(instance_mask_path, dtype=instance_mask_np.dtype, mode='w+', shape=instance_mask_np.shape)

        # # Copy the data into memory-mapped arrays
        # np.copyto(image_mm, image_np)
        # np.copyto(cluster_mask_mm, cluster_mask_np)
        # np.copyto(instance_mask_mm, instance_mask_np)

        # # Delete the memory-mapped arrays to save the changes
        # del image_mm
        # del cluster_mask_mm
        # del instance_mask_mm
        continue

if __name__ == "__main__":

    main()
