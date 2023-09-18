'''
Usage:
python my_train.py --data sample_data/images/ --labels sample_data/labels/ --save_dir sample_data/generated --weights example.pt --device cuda
'''
import os
import sys
# Add the project root to the sys path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import torch
from torch.utils.data import Dataset, DataLoader
from Dataloader.Dataset import LabelGenerator, DataLoaderX
from Dataloader.Preprocess import CustomDataLoader

from loss import ComputeLoss
from hyperparameters import Hyperparameters # hyperparamters: learning rate, iou threshold ....
# from utils.torch_utils import smart_optimizer, de_parallel
# from utils.callbacks import Callbacks


from tqdm import tqdm
import time

from Models.MultiNet import MultiNet


import argparse

# instantiating argparse
parser = argparse.ArgumentParser(
                    prog='validation',
                    description='it validates multiNet model',
                    epilog='Text at the bottom of help')

parser.add_argument('-data', '--data', help='path to the data directory')
parser.add_argument('-labels', '--labels', help='path to the labels directory')
parser.add_argument('-save_dir', '--save_dir', help='save directory', default='./')
parser.add_argument('-weight', '--weight', help='path to weight file', required=False)
parser.add_argument('-device', '--device', help='cpu or cuda')


args = parser.parse_args()

# args.data = args.data
# args.labels = args.labels
# args.save_dir = args.save_dir


class_mapping = {
    'pedestrian': 0,
    'rider': 1,
    'car': 2,
    'truck': 3,
    'bus': 4,
    'train': 5,
    'motorcycle': 6,
    'bicycle': 7,
    'traffic light': 8,
    'traffic sign': 9
}

# device 
DEVICE = torch.device(args.device)


# Usage
# args.data = "/content/dataset/bdd100k/bdd100k/images/100k/train/"
# args.labels = '/content/dataset/bdd100k_labels_release/bdd100k/labels/'

hyp = Hyperparameters().hyp # hyperparameters

model = MultiNet(hyp) # instantiating model

# load model to continue tarining
if args.weight is not None:
    model.load_state_dict(torch.load(args.weight, map_location=DEVICE))


nl = 3 # numnber of detection layers
nc = len(class_mapping.keys()) # numnber of classes
imgsz = 640 # input image size

# Model attributes
# nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
hyp['box'] *= 3 / nl  # scale to layers
hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
# hyp['label_smoothing'] = opt.label_smoothing
model.nc = nc  # attach number of classes to model
model.hyp = hyp  # attach hyperparameters to model
# model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
# model.names = names

# Start training
t0 = time.time()
# nb = len(train_loader)  # number of batches
# nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
# # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
# last_opt_step = -1
# maps = np.zeros(nc)  # mAP per class
# results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
# scheduler.last_epoch = start_epoch - 1  # do not move
scaler = torch.cuda.amp.GradScaler()
# stopper, stop = EarlyStopping(patience=opt.patience), False
compute_loss = ComputeLoss(model)  # instantiate loss class
# callbacks.run('on_train_start')
# LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
#             f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
#             f"Logging results to {colorstr('bold', save_dir)}\n"
#             f'Starting training for {epochs} epochs...')


start_epoch = 1
end_epoch = 1 + 1
epochs  = end_epoch - start_epoch # total number of epochs

optimizer = torch.optim.Adam(model.parameters(), 0.001) # instantiate optimizer

batch_size = 2
dataset = LabelGenerator(args.data, args.labels, args.save_dir ,image_size=(640, 640), normalize=True, class_mapping = class_mapping, num_batches=batch_size)

train_data_loader = DataLoaderX(dataset, batch_size = batch_size, shuffle = False, pin_memory = False, num_workers = 0, collate_fn=LabelGenerator.collate_fn)


TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format


for epoch in range(start_epoch, end_epoch):

    model.train()

    mloss = torch.zeros(3, device = DEVICE)  # mean losses

    optimizer.zero_grad()

    tloader = enumerate(train_data_loader)

    tloader = tqdm(tloader, total=batch_size, bar_format=TQDM_BAR_FORMAT)

    t0 = time.time() # TODO : delete later
    # Example of how to iterate through the data loader during training
    # for i, (batch_images, cluster_mask , instance_mask , image_name, batch_obj_annots) in tloader:
    for i, (batch_images, batch_obj_annots) in tloader:
        # Here you can use the batch_images, drivable_masks, and lane_masks for training
        # Remember that the batch size is determined by the 'batch_size' parameter you set above
        # Perform your training process here 

        print(f"\n\n shape of batch image 0: {batch_images[0].shape} \n\n")
        # print(batch_images.shape)
        # print(batch_obj_annots.shape)

        ni = i + batch_size * epoch  # number integrated batches (since train start)


        # forward
        # batch_images = batch_images.to(DEVICE, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0 # NOTE: if implemented seperately, remove this line

        # batch_images = torch.permute(batch_images, (0, 3, 2, 1))

        preds = model(batch_images)

        # print(f"'''''''''''''{preds[0].shape}'''''''''''''")

        loss, loss_items = compute_loss(preds, batch_obj_annots.to(DEVICE))

        # mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses

        # mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
        # tloader.set_description(('%11s' * 2 + '%11.4g' * 5) %
        #                              (f'epoch: {epoch}/{epochs}  ', f'memory: {mem}  ', *mloss, batch_obj_annots.shape[0], batch_images.shape[-1]))

        # # Backward
        # scaler.scale(loss).backward()

        # # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html TODO: understand this
        # scaler.unscale_(optimizer)  # unscale gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
        # scaler.step(optimizer)  # optimizer.step
        # scaler.update()
        print()
        # optimizer.zero_grad()
    t1 = time.time() # TODO : delete later
    print(f"time taken to iterate through data: {t1 - t0}")


t1 = time.time()


print(f"\n----------------------------- total training time was {t1 - t0} seconds -------------------------\n")

# saving model state dict containing all the learned parameters
try:
    torch.save(model.state_dict(), args.save_dir + '/model.pt')
    print(f"model has been saved successfully to {args.save_dir + '/model.pt'}")
except Exception as e:
    print(e)