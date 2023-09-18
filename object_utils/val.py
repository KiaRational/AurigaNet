'''
usage python my_val.py --data [path to image data], --labels [path to labels file] --save_dir [path to save directory] --weight [path to weight] --device [cpu or gpu(cuda)]
'''

import os
import sys
# Add the project root to the sys path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import torch
from torch.utils.data import Dataset, DataLoader
from Dataloader.Dataset import LabelGenerator, DataLoaderX
from Dataloader.PreProcess import CustomDataLoader

from utils import intersection_over_union, non_max_suppression

from loss import ComputeLoss

from network import MultiNet

from tqdm import tqdm
import time
import argparse

import numpy as np

from hyperparameters import Hyperparameters

# instantiating argparse
parser = argparse.ArgumentParser(
                    prog='validation',
                    description='it validates multiNet model',
                    epilog='Text at the bottom of help')

parser.add_argument('-data', '--data', help='path to the data directory')
parser.add_argument('-labels', '--labels', help='path to the labels directory')
parser.add_argument('-save_dir', '--save_dir', help='save directory')
parser.add_argument('-weight', '--weight', help='path to weight file', required=True)
parser.add_argument('-device', '--device', help='cpu or cuda')


args = parser.parse_args()

# Usage
# args.data = "/content/dataset/bdd100k/bdd100k/images/100k/train/"
# args.labels = '/content/dataset/bdd100k_labels_release/bdd100k/labels/'



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


hyp = Hyperparameters().hyp # hyperparameters

model = MultiNet(hyp)

# TODO : add the following to hyperparamters after organizing the code
DEVICE = args.device
IMGSZ = (640, 640)
OBJECTNESS_THESH = 0.5

dataset = LabelGenerator(args.data, args.labels, args.save_dir ,image_size=IMGSZ, normalize=True, class_mapping = class_mapping)
batch_size = 2
train_data_loader = DataLoaderX(dataset, batch_size = batch_size, shuffle = False, pin_memory = False, num_workers = 0)

# load model parameters for validation
if args.weight is not None:
    model.load_state_dict(torch.load(args.weight, map_location=DEVICE))


nl = 3 # numnber of detection layers
nc = len(class_mapping.keys()) # numnber of classes
imgsz = 640 # input image size

hyp['box'] *= 3 / nl  # scale to layers
hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
# hyp['label_smoothing'] = opt.label_smoothing
model.nc = nc  # attach number of classes to model
model.hyp = hyp  # attach hyperparameters to model


compute_loss = ComputeLoss(model)  # instantiate loss class

def process_accuracy(preds, targets, input_image):
    # TODO: calculate accuracy
    
    bs = preds[0].shape[0] # batch size
    for b in range(bs):
        input_w, input_h = input_image[b].shape[1], input_image[b].shape[0]


        p = [preds[0][b, ...], preds[1][b, ...], preds[2][b, ...]] # list of predictions
        p0 = np.reshape(p[0], (p[0].shape[0]*p[0].shape[1]*p[0].shape[2], p[0].shape[-1]))
        p1 = np.reshape(p[1], (p[1].shape[0]*p[1].shape[1]*p[1].shape[2], p[1].shape[-1]))
        p2 = np.reshape(p[2], (p[2].shape[0]*p[2].shape[1]*p[2].shape[2], p[2].shape[-1]))
        p = np.vstack((p0, p1, p2)) # predictions reshaped

        # p = p[p[:, 4] > OBJECTNESS_THESH] # filter predictions based on objectness conf threshold

        classes = p[:, 5:-1] # array containing classes per row
        boxes   = p[:, :4]*IMGSZ[0] # NOTE: bad approach => only works with square shape just to test
        confs   = p[:, 4] # filtered objecteness scores

        class_ids = np.argmax(classes, axis=1).reshape((classes.shape[0], 1)) # containing classid per row
        bboxes = np.hstack((class_ids, np.expand_dims(confs, axis=1), boxes))

        nms_boxes = non_max_suppression(bboxes, 0.01, 0.082, 'midpoint')

        print(f" shape of bounding boxes with class ids: {len(nms_boxes)}")
        
        # TODO to implement nms and calculate tp and fp and mAP with IoU


# # Start training
t0 = time.time()

with torch.no_grad():
    # Set model to eval mode
    model.eval()

    mloss = torch.zeros(3, device=DEVICE)  # mean losses

    tloader = enumerate(train_data_loader)

    # Example of how to iterate through the data loader during training
    for i, (batch_images, cluster_mask , instance_mask , image_name, batch_obj_annots) in tloader:
        # Here you can use the batch_images, drivable_masks, and lane_masks for training
        # Remember that the batch size is determined by the 'batch_size' parameter you set above
        # Perform your training process here 

        ni = i + batch_size  # number integrated batches (since train start)

        # forward
        batch_images = batch_images.to(DEVICE, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0 # NOTE: if implemented seperately, remove this line

        batch_images = torch.permute(batch_images, (0, 3, 2, 1))

        preds = model(batch_images)

        process_accuracy(preds, batch_obj_annots, batch_images) # TODO : to complete accuracy

        loss, loss_items = compute_loss(preds, batch_obj_annots.to(DEVICE))

        mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses



        # TODO => Calculate accuracy
        print(len(preds))

        


t1 = time.time()


print(f"\n----------------------------- total validation time was {t1 - t0} seconds -------------------------\n")




# @torch.no_grad()
# def run(
#         data,
#         weights=None,  # model.pt path(s)
#         batch_size=32,  # batch size
#         imgsz=640,  # inference size (pixels)
#         conf_thres=0.001,  # confidence threshold
#         iou_thres=0.6,  # NMS IoU threshold
#         task='val',  # train, val, test, speed or study
#         device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
#         workers=8,  # max dataloader workers (per RANK in DDP mode)
#         single_cls=False,  # treat as single-class dataset
#         augment=False,  # augmented inference
#         verbose=False,  # verbose output
#         save_txt=False,  # save results to *.txt
#         save_hybrid=False,  # save label+prediction hybrid results to *.txt
#         save_conf=False,  # save confidences in --save-txt labels
#         save_json=False,  # save a COCO-JSON results file
#         project=ROOT / 'runs/val',  # save to project/name
#         name='exp',  # save to project/name
#         exist_ok=False,  # existing project/name ok, do not increment
#         half=True,  # use FP16 half-precision inference
#         dnn=False,  # use OpenCV DNN for ONNX inference
#         model=None,
#         dataloader=None,
#         save_dir=Path(''),
#         plots=True,
#         callbacks=Callbacks(),
#         compute_loss=None,
# ):
#     # Initialize/load model and set device
#     training = model is not None
#     if training:  # called by train.py
#         device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
#         half &= device.type != 'cpu'  # half precision only supported on CUDA
#         model.half() if half else model.float()
#     else:  # called directly
#         device = torch.device('cuda:0' if device == 'cuda' else 'cpu')

#         # Directories
#         save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
#         (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

#         # Load model
#         model = MultiNet(hyp) 
#         # Data
#         data = check_dataset(data)  # check

#     # Configure
#     model.eval()
#     cuda = device.type != 'cpu'
#     nc = 1 if single_cls else int(data['nc'])  # number of classes
#     iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
#     niou = iouv.numel()

#     # Dataloader
#     if not training:
#         if pt and not single_cls:  # check --weights are trained on --data
#             ncm = model.model.yaml['nc']
#             assert ncm == nc, f'{weights[0]} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
#                               f'classes). Pass correct combination of --weights and --data that are trained together.'
#         pad = 0.0 if task in ('speed', 'benchmark') else 0.5
#         rect = False if task == 'benchmark' else pt  # square inference for benchmarks
#         task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
#         dataloader = tloader() # TODO

#     seen = 0
#     confusion_matrix = ConfusionMatrix(nc=nc)
#     names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
#     class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
#     s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
#     dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
#     loss = torch.zeros(3, device=device)
#     jdict, stats, ap, ap_class = [], [], [], []
#     callbacks.run('on_val_start')
#     pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
#     for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
#         callbacks.run('on_val_batch_start')
#         t1 = time_sync()
#         if cuda:
#             im = im.to(device, non_blocking=True)
#             targets = targets.to(device)
#         im = im.half() if half else im.float()  # uint8 to fp16/32
#         im /= 255  # 0 - 255 to 0.0 - 1.0
#         nb, _, height, width = im.shape  # batch size, channels, height, width
#         t2 = time_sync()
#         dt[0] += t2 - t1

#         # Inference
#         out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs
#         dt[1] += time_sync() - t2

#         # Loss
#         if compute_loss:
#             loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

#         # NMS
#         targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
#         lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
#         t3 = time_sync()
#         out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
#         dt[2] += time_sync() - t3

#         # Metrics
#         for si, pred in enumerate(out):
#             labels = targets[targets[:, 0] == si, 1:]
#             nl = len(labels)
#             tcls = labels[:, 0].tolist() if nl else []  # target class
#             path, shape = Path(paths[si]), shapes[si][0]
#             seen += 1

#             if len(pred) == 0:
#                 if nl:
#                     stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
#                 continue

#             # Predictions
#             if single_cls:
#                 pred[:, 5] = 0
#             predn = pred.clone()
#             scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

#             # Evaluate
#             if nl:
#                 tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
#                 scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
#                 labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
#                 correct = process_batch(predn, labelsn, iouv)
#                 if plots:
#                     confusion_matrix.process_batch(predn, labelsn)
#             else:
#                 correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
#             stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

#             # Save/log
#             if save_txt:
#                 save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
#             if save_json:
#                 save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
#             callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

#         # Plot images
#         if plots and batch_i < 3:
#             f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
#             Thread(target=plot_images, args=(im, targets, paths, f, names), daemon=True).start()
#             f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
#             Thread(target=plot_images, args=(im, output_to_target(out), paths, f, names), daemon=True).start()

#         callbacks.run('on_val_batch_end')

#     # Compute metrics
#     stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
#     if len(stats) and stats[0].any():
#         tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
#         ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
#         mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
#         nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
#     else:
#         nt = torch.zeros(1)

#     # Print results
#     pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
#     LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

#     # Print results per class
#     if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
#         for i, c in enumerate(ap_class):
#             LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

#     # Print speeds
#     t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
#     if not training:
#         shape = (batch_size, 3, imgsz, imgsz)
#         LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

#     # Plots
#     if plots:
#         confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
#         callbacks.run('on_val_end')

#     # Save JSON
#     if save_json and len(jdict):
#         w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
#         anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
#         pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
#         LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
#         with open(pred_json, 'w') as f:
#             json.dump(jdict, f)

#         try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
#             check_requirements(['pycocotools'])
#             from pycocotools.coco import COCO
#             from pycocotools.cocoeval import COCOeval

#             anno = COCO(anno_json)  # init annotations api
#             pred = anno.loadRes(pred_json)  # init predictions api
#             eval = COCOeval(anno, pred, 'bbox')
#             if is_coco:
#                 eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
#             eval.evaluate()
#             eval.accumulate()
#             eval.summarize()
#             map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
#         except Exception as e:
#             LOGGER.info(f'pycocotools unable to run: {e}')

#     # Return results
#     model.float()  # for training
#     if not training:
#         s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
#         LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
#     maps = np.zeros(nc) + map
#     for i, c in enumerate(ap_class):
#         maps[c] = ap[i]

#     return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t