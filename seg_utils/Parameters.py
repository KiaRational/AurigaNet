
#   _____                                     _                   
#  |  __ \                                   | |                  
#  | |__) |__ _  _ __  __ _  _ __ ___    ___ | |_  ___  _ __  ___ 
#  |  ___// _` || '__|/ _` || '_ ` _ \  / _ \| __|/ _ \| '__|/ __|
#  | |   | (_| || |  | (_| || | | | | ||  __/| |_|  __/| |   \__ \
#  |_|    \__,_||_|   \__,_||_| |_| |_| \___| \__|\___||_|   |___/
                                                        
class Parameters():

    ##############################################################################
    # Dataset Parameters
    ##############################################################################

    train_data_path = "/home/kia/BDD100K/bdd100k_images_100k_5/bdd100k/images/100k/train/"
    train_label_path = '/home/kia/BDD100K/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json'
    val_data_path = "/home/kia/BDD100K/bdd100k_images_100k_5/bdd100k/images/100k/val/"
    val_label_path = '/home/kia/BDD100K/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json'
    save_path = '/home/kia/BDD100K/Generated/'
    train_Lane_Mask_path = "/home/kia/BDD100K/Generated/train/Lane_Masks"
    train_Area_Mask_path = "/home/kia/BDD100K/Generated/train/Area_Mask"
    val_Lane_Mask_path = "/home/kia/BDD100K/Generated/val/Lane_Masks"
    val_Area_Mask_path = "/home/kia/BDD100K/Generated/val/Area_Mask"
    # train_data_path = "/home/ubuntu/BDD100K/bdd100k/images/100k/train/"
    # train_label_path = '/home/ubuntu/BDD100K/bdd100k/labels/bdd100k_labels_images_train.json'
    # val_data_path = "/home/ubuntu/BDD100K/bdd100k/images/100k/val/"
    # val_label_path = '/home/ubuntu/BDD100K/bdd100k/labels/bdd100k_labels_images_val.json'
    # save_path = '/home/ubuntu/BDD100K/Generated'
    # train_Lane_Mask_path = "/home/ubuntu/BDD100K/Generated/train/Lane_Masks"
    # train_Area_Mask_path = "/home/ubuntu/BDD100K/Generated/train/Area_Mask"
    # val_Lane_Mask_path = "/home/ubuntu/BDD100K/Generated/val/Lane_Masks"
    # val_Area_Mask_path = "/home/ubuntu/BDD100K/Generated/val/Area_Mask"
    class_mapping = {
        'traffic_light': 0,
        'traffic_sign': 1,
        'car': 2
    }


    ##############################################################################
    # Model Parameters
    ##############################################################################

    feature_size = 4
    grid_x = 40
    grid_y = 40


    ##############################################################################
    # Loss Functions Parameters
    ##############################################################################

    K1 = 1.00
    Alpha1 = 1
    Alpha2 = 1
    Alpha3 = 0.25
    

    ##############################################################################
    # Train Parameters
    ##############################################################################
    
    epoch_number = 30