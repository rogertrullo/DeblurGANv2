import os
from copy import deepcopy
from functools import partial
from glob import glob
from hashlib import sha1
from typing import Callable, Iterable, Optional, Tuple
from torchvision import transforms
import torch
import cv2
import numpy as np
from glog import logger
from joblib import Parallel, cpu_count, delayed
from skimage.io import imread
from torch.utils.data import Dataset
from tqdm import tqdm

import aug


def subsample(data: Iterable, bounds: Tuple[float, float], hash_fn: Callable, n_buckets=100, salt='', verbose=True):
    data = list(data)
    buckets = split_into_buckets(data, n_buckets=n_buckets, salt=salt, hash_fn=hash_fn)

    lower_bound, upper_bound = [x * n_buckets for x in bounds]
    msg = f'Subsampling buckets from {lower_bound} to {upper_bound}, total buckets number is {n_buckets}'
    if salt:
        msg += f'; salt is {salt}'
    if verbose:
        logger.info(msg)
    return np.array([sample for bucket, sample in zip(buckets, data) if lower_bound <= bucket < upper_bound])


def hash_from_paths(x: Tuple[str, str], salt: str = '') -> str:
    path_a, path_b = x
    names = ''.join(map(os.path.basename, (path_a, path_b)))
    return sha1(f'{names}_{salt}'.encode()).hexdigest()


def split_into_buckets(data: Iterable, n_buckets: int, hash_fn: Callable, salt=''):
    hashes = map(partial(hash_fn, salt=salt), data)
    return np.array([int(x, 16) % n_buckets for x in hashes])


def _read_img(x: str):
    img = cv2.imread(x)
    if img is None:
        logger.warning(f'Can not read image {x} with OpenCV, switching to scikit-image')
        img = imread(x)
    return img


class PairedDataset(Dataset):
    def __init__(self,
                 files_a: Tuple[str],
                 files_b: Tuple[str],
                 transform_fn: Callable,
                 normalize_fn: Callable,
                 corrupt_fn: Optional[Callable] = None,
                 preload: bool = True,
                 preload_size: Optional[int] = 0,
                 verbose=True):

        assert len(files_a) == len(files_b)

        self.preload = preload
        self.data_a = files_a
        self.data_b = files_b
        self.verbose = verbose
        self.corrupt_fn = corrupt_fn
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        logger.info(f'Dataset has been created with {len(self.data_a)} samples')

        if preload:
            preload_fn = partial(self._bulk_preload, preload_size=preload_size)
            if files_a == files_b:
                self.data_a = self.data_b = preload_fn(self.data_a)
            else:
                self.data_a, self.data_b = map(preload_fn, (self.data_a, self.data_b))
            self.preload = True

    def _bulk_preload(self, data: Iterable[str], preload_size: int):
        jobs = [delayed(self._preload)(x, preload_size=preload_size) for x in data]
        jobs = tqdm(jobs, desc='preloading images', disable=not self.verbose)
        return Parallel(n_jobs=cpu_count(), backend='threading')(jobs)

    @staticmethod
    def _preload(x: str, preload_size: int):
        img = _read_img(x)
        if preload_size:
            h, w, *_ = img.shape
            h_scale = preload_size / h
            w_scale = preload_size / w
            scale = max(h_scale, w_scale)
            img = cv2.resize(img, fx=scale, fy=scale, dsize=None)
            assert min(img.shape[:2]) >= preload_size, f'weird img shape: {img.shape}'
        return img

    def _preprocess(self, img, res):
        def transpose(x):
            return np.transpose(x, (2, 0, 1))

        return map(transpose, self.normalize_fn(img, res))

    def __len__(self):
        return len(self.data_a)

    def __getitem__(self, idx):
        a, b = self.data_a[idx], self.data_b[idx]
        if not self.preload:
            a, b = map(_read_img, (a, b))
        a, b = self.transform_fn(a, b)
        if self.corrupt_fn is not None:
            a = self.corrupt_fn(a)
        a, b = self._preprocess(a, b)
        return {'a': a, 'b': b}

    @staticmethod
    def from_config(config):
        config = deepcopy(config)
        files_a, files_b = map(lambda x: sorted(glob(config[x], recursive=True)), ('files_a', 'files_b'))
        transform_fn = aug.get_transforms(size=config['size'], scope=config['scope'], crop=config['crop'])
        normalize_fn = aug.get_normalize()
        corrupt_fn = aug.get_corrupt_function(config['corrupt'])

        hash_fn = hash_from_paths
        # ToDo: add more hash functions
        verbose = config.get('verbose', True)
        data = subsample(data=zip(files_a, files_b),
                         bounds=config.get('bounds', (0, 1)),
                         hash_fn=hash_fn,
                         verbose=verbose)

        files_a, files_b = map(list, zip(*data))

        return PairedDataset(files_a=files_a,
                             files_b=files_b,
                             preload=config['preload'],
                             preload_size=config['preload_size'],
                             corrupt_fn=corrupt_fn,
                             normalize_fn=normalize_fn,
                             transform_fn=transform_fn,
                             verbose=verbose)
    
    
class CarpetTest(Dataset):
    TYPE=['color',  'cut',  'good',  'hole',	'metal_contamination',  'thread']
    def __init__(self, root, resize=None, transformation=None, normalize=True, crop=128,patch_based=True):
        gt_root_path = os.path.join(root,'ground_truth')
        imgs_rooth_path = os.path.join(root,'test')
        self.normalize= normalize
        if resize is not None: 
            self.resize=[resize,resize]
            open_transform = transforms.Compose([transforms.Lambda(lambda img:imread(img)), transforms.ToTensor(),
                                                  transforms.ToPILImage(), transforms.Resize(self.resize)])
            gt_transform = transforms.Compose([transforms.Lambda(lambda img:imread(img)),
                                                  transforms.ToPILImage(), transforms.Resize(self.resize)])
        else:  

            open_transform = transforms.Compose([transforms.Lambda(lambda img:imread(img)), transforms.ToTensor(),
                                                  transforms.ToPILImage()])
            gt_transform = transforms.Compose([transforms.Lambda(lambda img:imread(img)),
                                                    transforms.ToPILImage()])
            self.resize= None
        self.transformation = transformation if transformation is not None else transforms.ToTensor()
        self.data = []
        for ttype in self.TYPE:
            gt_path_type = os.path.join(gt_root_path, ttype)
            imgs_path_type = os.path.join(imgs_rooth_path, ttype)
            for tmpi, (gtp, imgp) in enumerate(zip(sorted(glob(gt_path_type+'/*.png')) , sorted(glob(imgs_path_type+'/*.png')))):
                if tmpi>8:
                    break
                gt, img = gt_transform(gtp), open_transform(imgp)
                if self.resize is None:
                    self.resize = [gt.shape[1],gt.shape[2]]
                self.data.append((img, gt, ttype))
        self.crop = crop
        self.patch_based  = patch_based
                
    def __len__(self):
        return len(self.data)
    def __getitem__(self,i):
        img, gt, label = self.data[i]
        img = self.transformation(img)
        img = NORMALIZE(img) if self.normalize else img
        if self.patch_based:
            img =F.unfold(img.unsqueeze(0), kernel_size=(self.crop, self.crop), stride=(self.crop, self.crop)).reshape(img.shape[0], self.crop, self.crop, -1).permute(3,0,1,2)
        return img, self.transformation(gt), label

class CarpetInMemory(Dataset):    
    def __init__(self, root='data/carpet',  resize=512,  region_selection_transformation=None, sample=None,train=True,
                 input_transformation=None, on_output_trans=False, normalize=False):
        trainpath=os.path.join(root, 'train')
        self.training_paths=sorted(glob(trainpath+'/*/*.png'))
        self.resize=[resize,resize]
        self.train, self.normalize= train, normalize
        init_trf= transforms.Compose([transforms.ToPILImage(), transforms.Resize(self.resize)])
        self.region_selection_transformation = region_selection_transformation if region_selection_transformation is not None else transforms.ToTensor()

        self.input_transformation = input_transformation if input_transformation is not None else transforms.Lambda(lambda x: x)
        self.on_output_trans= on_output_trans
        self.data = []

        npatches_per_image=np.ceil(sample/len(self.training_paths)).astype(np.int)

        for idimage, p in enumerate(self.training_paths):
            if not train:                
                if (idimage%10)==0: print('reading img',os.path.basename(p))
                rgb_img=imread(p)    
                img_ini=init_trf(rgb_img)
                self.data.append(img_ini)
                if idimage>=20:
                    break
            else:
                
                if (idimage%50)==0: 
                    print('reading img',os.path.basename(p))
                rgb_img=imread(p)    
                img_ini=init_trf(rgb_img)
                i=0

                while i<npatches_per_image:
                    x=self.region_selection_transformation(img_ini)
                    if torch.min(x)==0:#avoid air...
                        continue

                    i+=1
                    self.data.append(x)

    def __getitem__(self, i):
        return self.data[i], self.data[i]

    def __len__(self):
        return len(self.data)
