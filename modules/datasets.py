import tqdm
from torchvision.datasets import VisionDataset
import glob

from PIL import Image

import os
import os.path
import sys

def pil_loader(path):

    with open(path, 'rb') as f:

        img = Image.open(f)
        return img.convert('RGB')


class synROD(VisionDataset):
  
    def __init__(self, root, RAM, split=None, transform=None, target_transform=None):
        super(synROD, self).__init__(root, transform=transform, target_transform=target_transform)

        self.images=[]
        self.RAM=RAM
        if split not in ["train", "test"]:
          raise ValueError("Split not acceptable!")

        basename = os.path.join(root,"synROD","synARID_50k-split_")
        path1 = basename + "depth_" + split + "1.txt"
        path2 = basename + "rgb_" + split + "1.txt"

        with open(path1, "r") as f1, open(path2, "r") as f2:
          num_lines = sum(1 for line in open(path1,"r"))
          pbar = tqdm.tqdm(total=num_lines,position=0, leave=True)
          for line1, line2 in zip(f1,f2):
            pbar.update(1)
            fields1 = line1.split(" ")
            fields2 = line2.split(" ")
            imagedepth = os.path.join(root,"synROD",fields1[0])
            imageRGB = os.path.join(root,"synROD",fields2[0])
            
            if RAM:
              self.images.append(((pil_loader(imageRGB), pil_loader(imagedepth)), int(fields1[1])))
            else:
              self.images.append(((imageRGB,imagedepth),int(fields1[1])))


    def __getitem__(self, index):

        images, label = self.images[index] 

        if self.transform is not None:

            if self.RAM:
              image1 = self.transform(images[0])
              image2 = self.transform(images[1])
            else:
              image1 = self.transform(pil_loader(images[0]))
              image2 = self.transform(pil_loader(images[1]))

        return image1,image2, label

    def __len__(self):

        length = len(self.images) 

        return length
        
class ROD(VisionDataset):
  
    def __init__(self, root, RAM, transform=None, target_transform=None):
        super(ROD, self).__init__(root, transform=transform, target_transform=target_transform)

        self.images=[]
        self.RAM=RAM
        filename = os.path.join(root,"ROD","rod-split_sync.txt")
        with open(filename,"r") as f:
          num_lines = sum(1 for line in open(filename))
          pbar = tqdm.tqdm(total=num_lines,position=0, leave=True)
          for line in f:
            pbar.update(1)
            fields = line.split(" ")
            rgb_path = fields[0].replace("???","rgb")
            rgb_path = rgb_path.replace("***","crop")
            depth_path = fields[0].replace("???","surfnorm")
            depth_path = depth_path.replace("***","depthcrop")
            imageRGB = os.path.join(root,"ROD",rgb_path)
            imagedepth = os.path.join(root,"ROD",depth_path)
            label = int(fields[1])

            if RAM:
              self.images.append(((pil_loader(imageRGB), pil_loader(imagedepth)), label))
            else:
              self.images.append(((imageRGB,imagedepth),label))



    def __getitem__(self, index):

        images, label = self.images[index] 

        if self.transform is not None:

            if self.RAM:
              image1 = self.transform(images[0])
              image2 = self.transform(images[1])
            else:
              image1 = self.transform(pil_loader(images[0]))
              image2 = self.transform(pil_loader(images[1]))

        return image1,image2, label

    def __len__(self):

        length = len(self.images) 

        return length
