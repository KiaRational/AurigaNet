'''
python dataloader_test.py --data sample_data/images/ --labels sample_data/labels/ --save_dir sample_data/generated --weights example.pt --device cuda
'''


from Dataloader.Dataset import LabelGenerator, DataLoaderX
from Dataloader.Preprocess import CustomDataLoader


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

if __name__ == "__main__":
    batch_size = 1
    dataset = LabelGenerator(args.data, args.labels, args.save_dir, image_size=(640, 640), normalize=True, class_mapping = class_mapping, num_batches=batch_size)

    train_data_loader = DataLoaderX(dataset, batch_size = batch_size, shuffle = False, pin_memory = False, num_workers = 0, collate_fn=LabelGenerator.collate_fn)


for i, data in enumerate(train_data_loader):
    print(type(data))