### CELLA 1
from torchvision.datasets import VisionDataset
import glob

from PIL import Image

import os
import os.path
import sys


def create_txt_dataset(root, ROD=True):

  numCat = -1
  if ROD:
    filename = os.path.join(root,"ROD","all_dataset.txt")
    if os.path.isfile(filename):
      raise FileExistsError
    with open(filename, 'w') as outfile:

      RGB_folder = os.path.join(root,"ROD","ROD_rgb")
      depth_folder = os.path.join(root,"ROD","ROD_surfnorm")
      for class_object in sorted(os.listdir(RGB_folder)):
        path = os.path.join(RGB_folder, class_object)
        if os.path.isdir(path):
          numCat+=1
          for subfolder in os.listdir(path):
            subpath = os.path.join(path, subfolder)
            if os.path.isdir(subpath):
              rgb_path = os.path.join(subpath, "*.png")
              root_depth_path = os.path.join(depth_folder, class_object, subfolder)
              for image_rgb in glob.glob(rgb_path):
                image_name = "_".join(image_rgb.split("/")[-1].split("_")[:-1]) + "_depthcrop.png"
                image_depth = os.path.join(root_depth_path, image_name)
                if os.path.isfile(image_depth):
                  outfile.write(image_rgb+","+image_depth+","+str(numCat)+"\n")

  else:
      filename = os.path.join(root,"synROD","all_dataset.txt")
      if os.path.isfile(filename):
        raise FileExistsError
      with open(filename, 'w') as outfile:
        for class_object in sorted(os.listdir(os.path.join(root,"synROD"))):
            path = os.path.join(root,"synROD", class_object)
            if os.path.isdir(path):
              numCat+=1
              rgb_path = os.path.join(path, "rgb", "*.png")
              root_depth_path = os.path.join(path, "depth")
              for image_rgb in glob.glob(rgb_path):
                image_depth = os.path.join(root_depth_path, image_rgb.split("/")[-1])
                if os.path.isfile(image_depth):
                  outfile.write(image_rgb+","+image_depth+","+str(numCat)+"\n")
  outfile.close()

### CELLA 2

import tqdm
def pil_loader(path):

    with open(path, 'rb') as f:

        img = Image.open(f)
        return img.convert('RGB')


class syn_ROD(VisionDataset):
  
    def __init__(self, root, dataset, RAM, TrainTestFile=False, split=None, transform=None, target_transform=None):
        super(syn_ROD, self).__init__(root, transform=transform, target_transform=target_transform)

        self.images=[]
        self.categories={}
        self.RAM=RAM
        
        if TrainTestFile and dataset!="synROD":
          raise ValueError("Only synROD has a Train/Test split on file")

        if TrainTestFile:
          if split not in ["train", "test"]:
            raise ValueError("Split not acceptable!")

          basename = os.path.join(root,dataset,"synARID_50k-split_")
          path1 = basename + "depth_" + split + "1.txt"
          path2 = basename + "rgb" + split + "1.txt"
          with open(path1) as f1, open(path2) as f2:
            for line1, line2 in zip(f1,f2):
              fields1 = line1.split(" ")
              fields2 = line2.split(" ")
              imagedepth = os.path.join(root,dataset,fields1[0])
              imageRGB = os.path.join(root,dataset,fields2[0])
              self.categories[fields1[0].split("/")[0]]=int(fields1[1])
              # in categories memorizzo nuova conversione label-numero ma come la uso su ROD? 
              # Scarto tutte le categorie che non sono al suo interno?
              if RAM:
                self.images.append(((pil_loader(imageRGB), pil_loader(imagedepth)), int(fields1[1])))
              else:
                self.images.append(((imageRGB,imagedepth),int(fields1[1])))
          with open()


        else:
          if dataset not in ["ROD", "synROD"]:
            raise ValueError("Dataset not acceptable")

          filename = os.path.join(root,dataset,"reading.txt")

          num_lines = sum(1 for line in open(filename))
          pbar = tqdm.tqdm(total=num_lines,position=0, leave=True)
          for line in open(filename,"r").read().splitlines():
            pbar.update(1)
            field = line.split(",")
            if RAM:
              self.images.append(((pil_loader(field[0]), pil_loader(field[1])), int(field[2])))
            else:
              self.images.append(((field[0],field[1]),int(field[2])))

          

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
