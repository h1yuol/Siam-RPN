from flag.utils import bb_intersection_over_union

from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import datetime

from torchvision import transforms


class RandomFlagGenerator:
    def __init__(self, colorset=None):
        """
        colorset should be a (n, 3) array
        """
        import matplotlib
        self.name2hex = {name:hexStr for name, hexStr in matplotlib.colors.cnames.items()}
        if colorset is None:
            self.colorset = list(map(self.hex2RGB, self.name2hex.values()))
        else:
            for idx,color in enumerate(colorset):
                if isinstance(color, str):
                    if color.startswith('#'):
                        colorset[idx] = self.hex2RGB(color)
                    elif color in self.name2hex:
                        colorset[idx] = self.hex2RGB(self.name2hex[color])
                    else:
                        raise ValueError("'{}'' is a invalid color".format(color))
            self.colorset = colorset
            print(self.colorset)

    @classmethod
    def hex2RGB(self, hexStr):
        assert hexStr[0] == '#' and len(hexStr) == 7
        return int(hexStr[1:3], base=16), int(hexStr[3:5], base=16), int(hexStr[5:7], base=16)

    def getRandomFlag(self, size):
        """
        size should be a tuple of (Height, width)
        """
        img = np.zeros((*size, 3), dtype=np.uint8)
        delta_height = size[0]//3
        delta_width = size[1]//3
        grids_color_indices = np.random.choice(len(self.colorset), 9, replace=True)
        for idx in range(9):
            grid_i = idx//3
            grid_j = idx%3
            color = np.array(self.colorset[grids_color_indices[idx]], dtype=np.uint8)
            assert color.shape == (3, )
            img[grid_i*delta_height:(grid_i+1)*delta_height, grid_j*delta_width:(grid_j+1)*delta_width] = color

        # self.displaySingleImg(img)
        img[img==0] = 1
        return img

class FlagBuilder:
    def __init__(self, root_dir='./'):
        self.root_dir = Path(root_dir)
        (self.root_dir/'data').mkdir(exist_ok=True)

        self.transforms = transforms.Compose([
                    transforms.ToPILImage(),
                    # transforms.RandomRotation(15,expand=np.random.uniform()>0.5),
                    # transforms.ColorJitter(0.5,0.5,0.5,0.1),
                    transforms.RandomAffine(15,shear=20),
                ])

    def load_image(self, img_path):
        img = cv2.imread(img_path)
        # img = np.array(Image.open(img_path))
        return img

    def save_image(self, img, img_path):
        cv2.imwrite(img_path, img)

    def build_randomGallery(self, num_train_classes=1, num_test_classes=1, size=(120,120)):
        flagGen = RandomFlagGenerator()
        for phase in ['train','test']:
            num_classes = eval('num_{}_classes'.format(phase))
            gallery_dir = self.dataset_dir / phase / 'gallery'
            gallery_dir.mkdir(exist_ok=True, parents=True)
            for idx in range(num_classes):
                img = flagGen.getRandomFlag(size)
                self.save_image(img, str(gallery_dir/'{}.png'.format(idx)))

    def load_gallery(self, phase):
        gallery_dir = self.dataset_dir / phase / 'gallery'
        gallery = {}
        for path in gallery_dir.glob('*'):
            ind = int(path.name.split('.')[0])
            img = self.load_image(str(path))
            gallery[ind] = img
        return gallery

    def random_insert_flag(self, img, flag, scale=0.25, prevbboxes=None):
        size = min(img.shape[:2])
        flag_height = size * scale
        flag_scale = np.random.uniform(0.7, 1.3)
        flag_width = int(flag_height * flag_scale)
        flag_height = int(flag_height)

        while True:
            pos = (np.random.choice(size-flag_height), np.random.choice(size-flag_width))
            bbox = [pos[1],pos[0],pos[1]+flag_width,pos[0]+flag_height]  # xmin, ymin, xmax, ymax
            overlap = False
            if prevbboxes is not None:
                for prevbbox in prevbboxes:
                    iou = bb_intersection_over_union(prevbbox, bbox)
                    if iou > 0.1:
                        overlap = True
                        break
            if not overlap:
                break
        flag = np.array(self.transforms(flag))
        flag = cv2.resize(flag, dsize=(flag_width, flag_height))

        mask = (flag==0)
        img[bbox[1]:bbox[3], bbox[0]:bbox[2]] *= mask
        img[bbox[1]:bbox[3], bbox[0]:bbox[2]] += flag
        return img[:size, :size], bbox
    
    def random_insert_multiflags(self, img, flagList, scaleRange):
        bboxList = []
        for flag in flagList:
            scale = np.random.uniform(scaleRange[0], scaleRange[1])
            img, bbox = self.random_insert_flag(img, flag.copy(), scale=scale, prevbboxes=bboxList)
            bboxList.append(bbox)
        return img, bboxList

    def center_insert_flag(self, img, flag, scale=0.25, random=False):
        size = min(img.shape[:2])
        flag_height = size * scale
        flag_scale = np.random.uniform(0.7, 1.3)
        flag_width = int(flag_height * flag_scale)
        flag_height = int(flag_height)

        if random:
            pos = (np.random.choice(size-flag_height), np.random.choice(size-flag_width))
        else:
            pos = ((size - flag_height)//2 , (size - flag_width)//2)
        bbox = [pos[1],pos[0],pos[1]+flag_width,pos[0]+flag_height]  # xmin, ymin, xmax, ymax

        flag = np.array(self.transforms(flag))
        flag = cv2.resize(flag, dsize=(flag_width, flag_height))

        mask = (flag==0)
        img[bbox[1]:bbox[3], bbox[0]:bbox[2]] *= mask
        img[bbox[1]:bbox[3], bbox[0]:bbox[2]] += flag
        return img[:size, :size], bbox

    def build(self, name, iter_img_paths, num_flags=1, exist_ok=False, num_train_classes=100, num_test_classes=100, 
                scaleRange=None,iter_loop=1):
        self.dataset_dir = self.root_dir / 'data' / name
        try:
            self.dataset_dir.mkdir(exist_ok=exist_ok)
        except:
            raise Exception("{} exists!".format(name))

        self.build_randomGallery(num_train_classes, num_test_classes)
        print("Built train and test galleries!")

        train_gallery = self.load_gallery('train')
        test_gallery = self.load_gallery('test')
        flagIndicesDict = {
            'train': list(train_gallery.keys()),
            'test': list(test_gallery.keys()),
        }
        for phase in ['train','validation','test']:
            (self.dataset_dir / phase / 'imgs').mkdir(parents=True)
        infoListDict = {
            'train': [],
            'validation': [],
            'test': [],
        }

        for idx, path in tqdm(enumerate(iter_img_paths)):
            image = self.load_image(str(path))
            for phase in ['train','validation','test']:
                if phase=='train':
                    loop = iter_loop
                else:
                    loop = 1
                for i in range(loop):
                    if phase in ['train','validation']:
                        gallery = train_gallery
                        flagIndices = flagIndicesDict['train']
                        img = image.copy()
                    else:
                        gallery = test_gallery
                        flagIndices = flagIndicesDict['test']
                        img = image.copy()
                    imgs_dir = self.dataset_dir / phase / 'imgs'
                    flagIndices = np.random.choice(flagIndices,size=num_flags)
                    flagList = list(map(lambda idx: gallery[idx], flagIndices))
                    if scaleRange is None:
                        scale = 0.25
                        img, bboxList = self.random_insert_multiflags(img,flagList,[scale-0.05,scale+0.05])
                    else:
                        img, bboxList = self.random_insert_multiflags(img,flagList,scaleRange)
                    save_path = imgs_dir / '{}-{}.png'.format(idx,i)
                    self.save_image(img, str(save_path))
                    info = {
                        # 'index': idx,
                        'labelList': flagIndices.tolist(),
                        'path': str(save_path),
                        'source': str(path),
                        'bboxList': bboxList,
                        'phase': phase,
                    }
                    infoListDict[phase].append(info)



                    # flagIdx = np.random.choice(flagIndices,size=num_flags)
                    # flag = gallery[flagIdx].copy()
                    # if scaleRange is None:
                    #     scale = 0.25
                    # else:
                    #     scale = np.random.uniform(scaleRange[0], scaleRange[1])
                    # img, bbox = self.center_insert_flag(img, flag, scale, random=True)
                    # save_path = imgs_dir / '{}-{}.png'.format(idx,i)
                    # self.save_image(img, str(save_path))
                    # info = {
                    #     # 'index': idx,
                    #     'label': int(flagIdx),
                    #     'path': str(save_path),
                    #     'source': str(path),
                    #     'bbox': bbox,
                    #     'phase': phase,
                    # }
                    # infoListDict[phase].append(info)
        import json
        for phase in ['train','validation','test']:
            jsonStr = json.dumps(infoListDict[phase])
            with open(str(self.dataset_dir / phase / 'infoList.json'), 'w') as hd:
                hd.write(jsonStr)

    def build_train_dataset(self, name, iter_img_paths, exist_ok=False, num_train_classes=100, num_test_classes=100, 
                scaleRange=None):
        self.dataset_dir = self.root_dir / 'data' / name
        try:
            self.dataset_dir.mkdir(exist_ok=exist_ok)
        except:
            raise Exception("{} exists!".format(name))

        self.build_randomGallery(num_train_classes, num_test_classes)
        print("Built train and test galleries!")

        phase = 'train'
        imgs_dir = self.dataset_dir / phase / 'imgs'
        imgs_dir.mkdir()

        gallery = self.load_gallery(phase)
        flagIndices = list(gallery.keys())
        infoList = []
        for idx, path in tqdm(enumerate(iter_img_paths)):
            img = self.load_image(str(path))
            flagIdx = np.random.choice(flagIndices)
            flag = gallery[flagIdx].copy()
            if scaleRange is None:
                scale = 0.25
            img, bbox = self.center_insert_flag(img, flag, scale)
            save_path = imgs_dir / '{}.png'.format(idx)
            self.save_image(img, str(save_path))
            info = {
                'index': idx,
                'label': int(flagIdx),
                'path': str(save_path),
                'source': str(path),
                'bbox': bbox,
            }
            infoList.append(info)
        import json
        jsonStr = json.dumps(infoList)
        with open(str(self.dataset_dir / phase / 'infoList.json'), 'w') as hd:
            hd.write(jsonStr)

    def build_val_dataset(self, name, iter_img_paths, scaleRange=None):
        self.dataset_dir = self.root_dir / 'data' / name
        assert self.dataset_dir.exists()
        
        phase = 'validation'
        imgs_dir = self.dataset_dir / phase / 'imgs'
        imgs_dir.mkdir(parents=True)

        gallery = self.load_gallery('train')
        flagIndices = list(gallery.keys())
        infoList = []
        for idx, path in tqdm(enumerate(iter_img_paths)):
            img = self.load_image(str(path))
            flagIdx = np.random.choice(flagIndices)
            flag = gallery[flagIdx].copy()
            if scaleRange is None:
                scale = 0.25
            img, bbox = self.center_insert_flag(img, flag, scale, random=True)
            save_path = imgs_dir / '{}.png'.format(idx)
            self.save_image(img, str(save_path))
            info = {
                'index': idx,
                'label': int(flagIdx),
                'path': str(save_path),
                'source': str(path),
                'bbox': bbox,
            }
            infoList.append(info)
        import json
        jsonStr = json.dumps(infoList)
        with open(str(self.dataset_dir / phase / 'infoList.json'), 'w') as hd:
            hd.write(jsonStr)

    def build_test_dataset(self, name, iter_img_paths, scaleRange=None):
        self.dataset_dir = self.root_dir / 'data' / name
        assert self.dataset_dir.exists()
        
        phase = 'test'
        imgs_dir = self.dataset_dir / phase / 'imgs'
        imgs_dir.mkdir()

        gallery = self.load_gallery(phase)
        flagIndices = list(gallery.keys())
        infoList = []
        for idx, path in tqdm(enumerate(iter_img_paths)):
            img = self.load_image(str(path))
            flagIdx = np.random.choice(flagIndices)
            flag = gallery[flagIdx].copy()
            if scaleRange is None:
                scale = 0.25
            img, bbox = self.center_insert_flag(img, flag, scale, random=True)
            save_path = imgs_dir / '{}.png'.format(idx)
            self.save_image(img, str(save_path))
            info = {
                'index': idx,
                'label': int(flagIdx),
                'path': str(save_path),
                'source': str(path),
                'bbox': bbox,
            }
            infoList.append(info)
        import json
        jsonStr = json.dumps(infoList)
        with open(str(self.dataset_dir / phase / 'infoList.json'), 'w') as hd:
            hd.write(jsonStr)



