import torch
from torch.utils.data import Dataset, DataLoader
from dataset import CustomDataset

from utils.loss import ComputeLoss
from utils.torch_utils import smart_optimizer, de_parallel
from utils.callbacks import Callbacks

from models.yolo import Model




# Usage
data_path = "/content/dataset/bdd100k/bdd100k/images/100k/train/"
label_path = '/content/dataset/bdd100k_labels_release/bdd100k/labels/'



dataset = CustomDataset(data_path, label_path, image_size=(720, 1280), normalize=True, class_mapping=class_mapping)
batch_size = 8  # Choose your desired batch size
train_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)


# Model attributes
nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
hyp['box'] *= 3 / nl  # scale to layers
hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
hyp['label_smoothing'] = opt.label_smoothing
model.nc = nc  # attach number of classes to model
model.hyp = hyp  # attach hyperparameters to model
model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
model.names = names

# Start training
t0 = time.time()
nb = len(train_loader)  # number of batches
nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
# nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
last_opt_step = -1
maps = np.zeros(nc)  # mAP per class
results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
scheduler.last_epoch = start_epoch - 1  # do not move
scaler = torch.cuda.amp.GradScaler(enabled=amp)
stopper, stop = EarlyStopping(patience=opt.patience), False
compute_loss = ComputeLoss(model)  # init loss class
callbacks.run('on_train_start')
LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
            f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
            f"Logging results to {colorstr('bold', save_dir)}\n"
            f'Starting training for {epochs} epochs...')

start_epoch = 1
end_epoch = 10 + 1

for i in range(start_epoch, end_epoch):

    model.train()

    mloss = torch.zeros(3, device=device)  # mean losses

    optimizer.zero_grad()


    # Example of how to iterate through the data loader during training
    for i, (batch_images, drivable_masks, lane_masks, batch_obj_annots) in enumerate(train_data_loader):
        # Here you can use the batch_images, drivable_masks, and lane_masks for training
        # Remember that the batch size is determined by the 'batch_size' parameter you set above
        # Perform your training process here

        ni = i + nb * epoch  # number integrated batches (since train start)
        imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0 # NOTE: if implemented seperately, remove this line

        try:
            print(batch_obj_annots)
            print(i * batch_size)  # Print the starting index of each batch
        except:
            continue
