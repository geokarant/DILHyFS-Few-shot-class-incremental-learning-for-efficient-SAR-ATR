import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels,select_samples_per_class
from PIL import Image

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

def build_transform(is_train, args):
    input_size = args["input_dim"]
    crop_size= args["crop_size"]
    if is_train:
        scale = (0.8, 1.0)
        transform = [
            transforms.CenterCrop(crop_size),
            transforms.RandomResizedCrop(input_size, scale=scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    else:
        transform = [
        transforms.CenterCrop(input_size),
        transforms.ToTensor()]
    print(transform)
    return transform

class mstar(iData):
    
    def __init__(self, backbone_type,setup,portion,args):
        super().__init__()
        self.use_path = True
        self.backbone_type = backbone_type
        self.common_trsf = [
            # transforms.ToTensor(),
        ]
        self.portion = portion
        self.seed= args["seed"]
        self.args=args
        self.few_shot =args["few_shot"]
        
        self.train_trsf = build_transform(True, self.args)
        self.test_trsf = build_transform(False, self.args)
        if setup == 2:
            self.class_order = [9, 2, 6, 8, 4, 3, 7, 0, 1, 5]  
        elif setup == 1 :
            self.class_order = [2, 8, 6, 9, 0,7,3,1,4,5]            
    def download_data(self):
        print(self.backbone_type,":downloading MSTAR")
        train_dir = "./datasets/MSTAR/train"
        test_dir = "./datasets/MSTAR/test"
        
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        if self.few_shot == True:
            target_classes = self.class_order[self.args["init_cls"]:]
            print("few shot classes:",target_classes)
            if self.args['rotated_angles'] == False:
                rotated_angles= None
            else:
                rotated_angles = self.args["rotated_angles"]
            train_samples = select_samples_per_class(train_dset, 5, target_classes,self.seed,rotated_angles )
        else:
            train_samples= train_dset.imgs
        self.train_data, self.train_targets = split_images_labels(train_samples)                
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        if self.portion != 1:
            from sklearn.model_selection import train_test_split
            train_images, train_labels = split_images_labels(train_dset.imgs)
            self.train_data, val_images, self.train_targets, val_labels = train_test_split(
                train_images, train_labels, test_size= (1 -self.portion), random_state=42 , stratify=train_labels
                )        
            print("We use the ", self.portion*100, " of data ")

class mstar_cross(iData):
    
    def __init__(self, backbone_type,setup,portion,args):
        super().__init__()
        self.use_path = True
        self.backbone_type = backbone_type
        self.common_trsf = [
            # transforms.ToTensor(),
        ]
        self.portion = portion
        self.seed= args["seed"]
        self.args=args
        self.few_shot =args["few_shot"]
        
        self.train_trsf = build_transform(True, self.args)
        self.test_trsf = build_transform(False, self.args)
        self.class_order = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]            
    def download_data(self):
        print(self.backbone_type,":downloading MSTAR_CROSS")
        train_dir = "./datasets/MSTAR_CROSS/train"
        test_dir = "./datasets/MSTAR_CROSS/test"
        
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        if self.few_shot == True:
            target_classes = self.class_order[self.args["init_cls"]:]
            print("few shot classes:",target_classes)
            if self.args['rotated_angles'] == False:
                rotated_angles= None
            else:
                rotated_angles = self.args["rotated_angles"]
            train_samples = select_samples_per_class(train_dset, 5, target_classes,self.seed,rotated_angles )
        else:
            train_samples= train_dset.imgs
        self.train_data, self.train_targets = split_images_labels(train_samples)                
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        if self.portion != 1:
            from sklearn.model_selection import train_test_split
            train_images, train_labels = split_images_labels(train_dset.imgs)
            self.train_data, val_images, self.train_targets, val_labels = train_test_split(
                train_images, train_labels, test_size= (1 -self.portion), random_state=42 , stratify=train_labels
                )        
            print("We use the ", self.portion*100, " of data ")