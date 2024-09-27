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
    train_data_path = "./bdd100k_images_100k_5/bdd100k_images_100k_5/bdd100k/images/100k/train"
    train_label_path = './bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json'
    val_data_path = "./bdd100k_images_100k_5/bdd100k_images_100k_5/bdd100k/images/100k/val"
    val_label_path = './bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json'
    save_path = './artifacts'
    train_Lane_Mask_path = "./Generated/Generated/train/Lane_Masks"
    train_Area_Mask_path = "./Generated/Generated/train/Area_Mask"
    val_Lane_Mask_path = "./Generated/Generated/val/Lane_Masks"
    val_Area_Mask_path = "./Generated/Generated/val/Area_Mask"


    # class_mapping = {
    #     'car': 0,
    #     'rider': 1,
    #     'pedestrian': 2,
    #     'truck': 3,
    #     'bus': 4,
    #     'train': 5,
    #     'motor': 6,
    #     'bicycle': 7,
    #     'traffic light': 8,
    #     'traffic sign': 9
    # }
    # bdd = [
    #     'car' ,
    #     'rider' ,
    #     'pedestrian' ,
    #     'truck' ,
    #     'bus' ,
    #     'train' ,
    #     'motor' ,
    #     'bicycle' ,
    #     'traffic light' ,
    #     'traffic sign' 
    #     ]
    class_mapping = {
        'car': 0

    }
    bdd = [
        'car' 
        ]
    ##############################################################################
    # Model Parameters
    ##############################################################################

    feature_size = 8
    grid_x = 80
    grid_y = 80
    embedding_grid_size = 80

    anchors = [
        [[10, 13 ], [16, 30  ], [33, 23  ]],  # P3/8
        [[30, 61 ], [62, 45  ], [59, 119 ]],  # P4/16
        [[116, 90], [156, 198], [373, 326]]  # P5/32
    ]

    strides = [8, 16, 32]
    # anchors = [[10,13, 16,30, 33,23],  # P3/8
    #           [30,61, 62,45, 59,119],  # P4/16
    #           [116,90, 156,198, 373,326]] # P5/32

    ##############################################################################
    # Loss Functions Parameters
    ##############################################################################
    image_size = 640

    K1 = 1.00
    Alpha1 = 1.0
    Alpha2 = 1.0
    Alpha3 = 1.0
    Alpha4 = 1.0
    embedding_alpha = 0.02
    embedding_delta = 0.5
    embedding_lambda_intra = 1.0
    embedding_lambda_inter = 1.0
    box = 0.05  # box loss gain
    clss = 0.5  # cls loss gain
    cls_pw = 1.0  # cls BCELoss positive_weight
    obj = 1.0  # obj loss gain (scale with pixels)
    obj_pw = 1.0  # obj BCELoss positive_weight
    iou_t = 0.20  # IoU training threshold
    anchor_t = 4.0  # anchor-multiple threshold
    fl_gamma = 0.0  # focal loss gamma (efficientDet default gamma=1.5)

    ##############################################################################
    # Train Parameters
    ##############################################################################
    
    epoch_number = 250
    accumulate_batch_step = 64
    lrf = 0.2
    lr0_adam = 1e-4 #0.001
    lr0_sgd = 0.01
    warmup_epochs = 0
    warmup_bias_lr = 1e-2 #0.1
    warmup_momentum = 0.8
    momentum = 0.937
    
