from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import glob
from PIL import Image



class BaseDataset(Dataset):
    def __init__(self, path):
        self.imgs_path = path
        file_list = glob.glob(self.imgs_path + "*")
        print(file_list)
        self.data = []
        self.class_map = {}
        i = 0

        # for each folder
        for class_path in file_list:
            class_name = class_path.split("\\")[-1]
            self.class_map[class_name] = i
            i += 1

            # add image path and the corr class to the data
            for img_path in glob.glob(class_path + "/*.png"):
                self.data.append([img_path, class_name])
        #print(self.data)
        #print(self.class_map)
        self.img_dim = (64, 64)

    def __len__(self):
        return len(self.data)

# We only augment training data (we want consistent tests)
class ASLTrainDataset(BaseDataset):
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        # load img using pillow for data augmantation 
        img = Image.open(img_path).convert('1') # open image in blavk and white
        #img = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)

        # resize it to make sure it is 64*64
        img = img.resize(self.img_dim)
        class_id = self.class_map[class_name]


        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # randomly flip the image horizontally with a 50% chance
            transforms.RandomVerticalFlip(p=0.5),    # randomly flip the image vertically with a 50% chance
            transforms.RandomRotation(degrees=(0, 50)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
            transforms.ToTensor(),  # Converts image to a Tensor in the range [0, 1]
            transforms.Normalize((0.5,), (0.5,))   
        ])

        # Apply the transformation to the image
        img_tensor = transform(img)

        return img_tensor.float(), torch.tensor(class_id, dtype=torch.long)
    
class ASLTestDataset(BaseDataset):
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        # load img using pillow for data augmantation 
        img = Image.open(img_path).convert('1') # open image in blavk and white
        #img = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)

        # resize it to make sure it is 64*64
        img = img.resize(self.img_dim)
        class_id = self.class_map[class_name]


        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts image to a Tensor in the range [0, 1]
            transforms.Normalize((0.5,), (0.5,))   
        ])

        # Apply the transformation to the image
        img_tensor = transform(img)

        return img_tensor.float(), torch.tensor(class_id, dtype=torch.long)
    

