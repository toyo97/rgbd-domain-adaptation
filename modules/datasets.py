import tqdm
from torchvision.datasets import VisionDataset
from PIL import Image
import os
import os.path
from .transforms import CoupledRotation


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class SynROD_ROD(VisionDataset):

    def __init__(self, root, RAM=False, category=None, split=None):
        super(SynROD_ROD, self).__init__(root, transforms=None)

        self.images = []
        self.RAM = RAM
       
        if category not in ["ROD", "synROD"]:
            raise ValueError("category not acceptable!")
            
        if category=="synROD": 
            if split not in ["train", "test"]:
                raise ValueError("split not acceptable!")
        
            basename = os.path.join(root, category, "synARID_50k-split_")
            path1 = basename + "depth_" + split + "1.txt"
            path2 = basename + "rgb_" + split + "1.txt"

            with open(path1, "r") as f1, open(path2, "r") as f2:
                num_lines = sum(1 for line in open(path1, "r"))
                pbar = tqdm.tqdm(total=num_lines, position=0, leave=True)
                for line1, line2 in zip(f1, f2):
                    pbar.update(1)
                    fields1 = line1.split(" ")
                    fields2 = line2.split(" ")
                    imagedepth = os.path.join(root, "synROD", fields1[0])
                    imageRGB = os.path.join(root, "synROD", fields2[0])

                    if RAM:
                        self.images.append(((pil_loader(imageRGB), pil_loader(imagedepth)), int(fields1[1])))
                    else:
                        self.images.append(((imageRGB, imagedepth), int(fields1[1])))
                        
        else:
            
            filename = os.path.join(root, category, "rod-split_sync.txt")
            with open(filename, "r") as f:
                num_lines = sum(1 for _ in open(filename))
                pbar = tqdm.tqdm(total=num_lines, position=0, leave=True)
                for line in f:
                    pbar.update(1)
                    fields = line.split(" ")
                    rgb_path = fields[0].replace("???", "rgb")
                    rgb_path = rgb_path.replace("***", "crop")
                    depth_path = fields[0].replace("???", "surfnorm")
                    depth_path = depth_path.replace("***", "depthcrop")
                    rgb_img = os.path.join(root, "ROD", rgb_path)
                    depth_img = os.path.join(root, "ROD", depth_path)
                    label = int(fields[1])

                    if self.RAM:
                        self.images.append(((pil_loader(rgb_img), pil_loader(depth_img)), label))
                    else:
                        self.images.append(((rgb_img, depth_img), label)) 

    def __getitem__(self, index):

        (rgb_img, depth_img), label = self.images[index]


        if not self.RAM:
            rgb_img = pil_loader(rgb_img)
            depth_img = pil_loader(depth_img)

        return rgb_img, depth_img, label

    def __len__(self):

        length = len(self.images)

        return length

class TransformedDataset(SynROD_ROD):
    r"""
    Variation of a dataset (for relative rotation).

    Arguments:
        dataset (SynROD_ROD): The whole Dataset
    """
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, idx):
        old_label = self.dataset[idx][2]
        new_rgb, new_depth, new_label = self.transforms(self.dataset[idx][0], self.dataset[idx][1])
        if new_label==None:
            return new_rgb, new_depth, old_label
        else:
            return new_rgb, new_depth, new_label
        

    def __len__(self):
        return len(self.dataset)
