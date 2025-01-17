from __future__ import print_function

import numpy as np
import os
import os.path
import sys
import shutil
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.autograd import Variable
from torchvision.datasets import VisionDataset
from torchvision.datasets import utils

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
    
from collections import defaultdict
from os import path as osp

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length, prob=1.0):
        self.length = length
        self.prob = prob

    def __call__(self, img):
        if np.random.binomial(1, self.prob):
            h, w = img.size(1), img.size(2)
            mask = np.ones((h, w), np.float32)
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img *= mask
        return img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length,
                                                 args.cutout_prob))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        #['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not utils.check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not utils.check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        utils.download_and_extract_archive(self.url, self.root,
                                           filename=self.filename,
                                           md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def adj_matrix_to_adj_list(adj_matrix):
  G = defaultdict(list)
  n, m = len(adj_matrix), len(adj_matrix[0])
  for i in range(n):
      for j in range(m):
          if adj_matrix[i][j] == 1:
              G[i].append(j)
  return G

def genotype_to_adjacency_list(genotype, steps=4):
  # Should pass in genotype.normal or genotype.reduce
  G = defaultdict(list)
  for nth_node, connections in enumerate(chunks(genotype, 2), start=2): # Darts always keeps two connections per node and first two nodes are fixed input
    for connection in connections:
      G[connection[1]].append(nth_node)
  # Add connections from all intermediate nodes to Output node
  for intermediate_node in [2,3,4,5]:
    G[intermediate_node].append(6)
  return G
    
def DFS(G,v,seen=None,path=None):
    if seen is None: seen = []
    if path is None: path = [v]

    seen.append(v)

    paths = []
    for t in G[v]:
        if t not in seen:
            t_path = path + [t]
            paths.append(tuple(t_path))
            paths.extend(DFS(G, t, seen[:], t_path))
    return paths
  
def count_edges_along_path(genotype, path):
  count = 0
  for i in range(1, len(path)-1): #Leave out the first and last nodes
    idx_in_genotype = path[i]-2
    relevant_edges = genotype[idx_in_genotype*2:idx_in_genotype*2+2]
    for edge in relevant_edges:
      if edge[1] == path[i-1]:
        count += 1
  return count

def genotype_depth(adj_matrix):
  # The shortest path can start in either of the two input nodes
  all_paths0 = DFS(adj_matrix_to_adj_list(adj_matrix), 0)
  all_paths1 = DFS(adj_matrix_to_adj_list(adj_matrix), 1)

  cand0 = max(len(p)-1 for p in all_paths0) if len(all_paths0) > 0 else 1
  
  cand1 = max(len(p)-1 for p in all_paths1) if len(all_paths1) > 0 else 1
    
  
  # max_paths0 = [p for p in all_paths0 if len(p) == cand0]
  # max_paths1 = [p for p in all_paths1 if len(p) == cand1]

  # path_depth0 = max([count_edges_along_path(genotype, p) for p in max_paths0])
  # path_depth1 = max([count_edges_along_path(genotype, p) for p in max_paths1])
  
  # return max(path_depth0, path_depth1)

  return max(cand0, cand1)

def genotype_width(adj_matrix):
  import networkx
  G = networkx.convert_matrix.from_numpy_matrix(adj_matrix)
  width = networkx.algorithms.approximation.maxcut.one_exchange(G)[0]
  return width
      
def save_checkpoint2(state, filename, logger=None, quiet=False, backup=True):
  try:
    if osp.isfile(filename):
      print('Find {:} exist, delete is at first before saving'.format(filename))
      if backup:
        shutil.copy(filename, os.fspath(filename)+"_backup")
        print(f"Made backup of checkpoint to {os.fspath(filename)+'_backup'}")
      os.remove(filename)
    try:
      torch.save(state, filename.parent / (filename.name + 'tmp'))
      print(f"Saved checkpoint to tmp, now replacing the original")
      os.replace(filename.parent / (filename.name + 'tmp'), filename)
    except Exception as e:
      print(f"Failed to save new checkpoint into {filename} due to {e}")
    assert osp.isfile(filename), 'save filename : {:} failed, which is not found.'.format(filename)
    if hasattr(logger, 'log') and not quiet: print('save checkpoint into {:}'.format(filename))
  except Exception as e:
    print(f"Failed to save_checkpoint to {filename} due to {e}")
  return filename