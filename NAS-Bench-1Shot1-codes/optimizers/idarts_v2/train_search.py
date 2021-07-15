# python NAS-Bench-1Shot1-codes/optimizers/idarts_v2/train_search.py --T=5 --K=2 --merge_train_val=True

import argparse
import glob
import json
import logging
import os
import pickle
import sys
import time
import matplotlib.pyplot as plt

from torch.autograd import Variable

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset

from genotypes import count_ops
from utils import genotype_width, genotype_depth
import sys

from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))

from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2
from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3
from optimizers.idarts_v2 import utils
from optimizers.idarts_v2.architect_approx import Architect
#from optimizers.idarts_v2.architect_hess import Architect
from optimizers.idarts_v2.model_search import Network



from nasbench_analysis import eval_darts_one_shot_model_in_nasbench as naseval
from nasbench_analysis.utils import NasbenchWrapper


import wandb
from tqdm import tqdm
from sotl_utils import format_input_data, fo_grad_if_possible, hyper_meta_step, hypergrad_outer, approx_hessian, exact_hessian

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the darts corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=9, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=1, help='random_ws seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training darts')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--output_weights', type=bool, default=True, help='Whether to use weights on the output nodes')
parser.add_argument('--search_space', choices=['1', '2', '3'], default='3')
parser.add_argument('--debug', action='store_true', default=False, help='run only for some batches')
parser.add_argument('--warm_start_epochs', type=int, default=0,
                    help='Warm start one-shot model before starting architecture updates.')

parser.add_argument('--merge_train_val', type=lambda x: False if x in ["False", "false", "", "None", False, None] else True, default=False,
                    help='Warm start one-shot model before starting architecture updates.')
parser.add_argument('--T', type=int, default=1, help='random_ws seed')
parser.add_argument('--K', type=int, default=1, help='random_ws seed')
parser.add_argument('--steps_per_epoch', type=int, default=None,
                    help='Warm start one-shot model before starting architecture updates.')
parser.add_argument('--hessian', type=lambda x: False if x in ["False", "false", "", "None", False, None] else True, default=True,
                    help='Warm start one-shot model before starting architecture updates.')

args = parser.parse_args()

args.save = 'experiments/idarts_v2/search_space_{}/search_step5_k2_new-{}-{}-{}-{}'.format(args.search_space, args.save,
                                                                          time.strftime("%Y%m%d-%H%M%S"), args.seed,
                                                                          args.search_space)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

# Dump the config of the run
with open(os.path.join(args.save, 'config.json'), 'w') as fp:
    json.dump(args.__dict__, fp)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10

def wandb_auth(fname: str = "nas_key.txt"):
  gdrive_path = "/content/drive/MyDrive/colab/wandb/nas_key.txt"
  if "WANDB_API_KEY" in os.environ:
      wandb_key = os.environ["WANDB_API_KEY"]
  elif os.path.exists(os.path.abspath("~" + os.sep + ".wandb" + os.sep + fname)):
      # This branch does not seem to work as expected on Paperspace - it gives '/storage/~/.wandb/nas_key.txt'
      print("Retrieving WANDB key from file")
      f = open("~" + os.sep + ".wandb" + os.sep + fname, "r")
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key
  elif os.path.exists("/root/.wandb/"+fname):
      print("Retrieving WANDB key from file")
      f = open("/root/.wandb/"+fname, "r")
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key

  elif os.path.exists(
      os.path.expandvars("%userprofile%") + os.sep + ".wandb" + os.sep + fname
  ):
      print("Retrieving WANDB key from file")
      f = open(
          os.path.expandvars("%userprofile%") + os.sep + ".wandb" + os.sep + fname,
          "r",
      )
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key
  elif os.path.exists(gdrive_path):
      print("Retrieving WANDB key from file")
      f = open(gdrive_path, "r")
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key
  wandb.login()

def get_torch_home():
    if "TORCH_HOME" in os.environ:
        return os.environ["TORCH_HOME"]
    elif "HOME" in os.environ:
        return os.path.join(os.environ["HOME"], ".torch")
    else:
        raise ValueError(
            "Did not find HOME in os.environ. "
            "Please at least setup the path of HOME or TORCH_HOME "
            "in the environment."
        )
        
        
try:
    nasbench = NasbenchWrapper(
            dataset_file=os.path.join(get_torch_home(), 'nasbench_only108.tfrecord'))
except:
    nasbench = NasbenchWrapper(
        dataset_file=os.path.join(get_torch_home(), 'nasbench_full.tfrecord'))

def main():
    # Select the search space to search in
    if args.search_space == '1':
        search_space = SearchSpace1()
    elif args.search_space == '2':
        search_space = SearchSpace2()
    elif args.search_space == '3':
        search_space = SearchSpace3()
    else:
        raise ValueError('Unknown search space')

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    logger = logging.getLogger()

    wandb_auth()
    run = wandb.init(project="NAS", group=f"Search_Cell_idarts", reinit=True)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, output_weights=args.output_weights,
                    steps=search_space.num_intermediate_nodes, search_space=search_space)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)
    if args.merge_train_val:
        valid_queue = train_queue
        
    if os.path.exists(Path(args.save) / "checkpoint.pt"):
        checkpoint = torch.load(Path(args.save) / "checkpoint.pt")
        optimizer.load_state_dict(checkpoint["w_optimizer"])
        architect.optimizer.load_state_dict(checkpoint["a_optimizer"])
        model.load_state_dict(checkpoint["model"])
        scheduler.load_state_dict(checkpoint["w_scheduler"])
        start_epoch = checkpoint["epoch"]
        all_logs = checkpoint["all_logs"]

    else:
        print(f"Path at {Path(args.save) / 'checkpoint.pt'} does not exist")
        start_epoch=0
        all_logs=[]   
         
    for epoch in tqdm(range(args.epochs), total =args.epochs, desc="Iterating over epochs"):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        # increase the cutout probability linearly throughout search
        train_transform.transforms[-1].cutout_prob = args.cutout_prob * epoch / (args.epochs - 1)
        logging.info('epoch %d lr %e cutout_prob %e', epoch, lr,
                     train_transform.transforms[-1].cutout_prob)

        # Save the one shot model architecture weights for later analysis
        arch_filename = os.path.join(args.save, 'one_shot_architecture_{}.obj'.format(epoch))
        with open(arch_filename, 'wb') as filehandler:
            numpy_tensor_list = []
            for tensor in model.arch_parameters():
                numpy_tensor_list.append(tensor.detach().cpu().numpy())
            pickle.dump(numpy_tensor_list, filehandler)

        # Save the entire one-shot-model
        filepath = os.path.join(args.save, 'one_shot_model_{}.obj'.format(epoch))
        torch.save(model.state_dict(), filepath)

        #logging.info('architecture', numpy_tensor_list)
        print(numpy_tensor_list)

        # training
        train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, steps_per_epoch=args.steps_per_epoch)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)


        logging.info('STARTING EVALUATION_epoch %f', epoch)
        test, valid, runtime, params = naseval.eval_one_shot_model(config=args.__dict__, model=arch_filename, nasbench=nasbench)
        logging.info('TEST ERROR: %.3f | VALID ERROR: %.3f | RUNTIME: %f | PARAMS: %d'
                     % (test,
                        valid,
                        runtime,
                        params)
                     )
        genotype_perf = test
        

        if args.hessian and torch.cuda.get_device_properties(0).total_memory > 15147483648:
            eigenvalues = exact_hessian(network=model, val_loader=valid_queue, criterion=criterion, xloader=valid_queue, epoch=epoch, logger=logger, args=args)

        else:
            eigenvalues = None
        
        adj_matrix, ops_list = naseval.extract_arch(config=args.__dict__,
                                                               model=arch_filename, nasbench=nasbench)
        
        width = genotype_width(adj_matrix)
        depth = genotype_depth(adj_matrix)
        ops_count = count_ops(ops_list)
        print(f"Adj matrix: {adj_matrix}, ops_list: {ops_list}, width: {width}, depth: {depth}, ops_count: {ops_count}")

        
        wandb_log = {"val_acc": valid_acc, "valid_loss":valid_obj,
                     "search.final.cifar10": genotype_perf, "epoch":epoch, "eigval":eigenvalues, 
                     "ops": ops_count, "width":width, "depth":depth}

        all_logs.append(wandb_log)
        wandb.log(wandb_log)
        
        
        utils.save_checkpoint2({"model":model.state_dict(), "w_optimizer":optimizer.state_dict(), 
                    "a_optimizer":architect.optimizer.state_dict(), "w_scheduler":scheduler.state_dict(), "epoch": epoch, 
                    "all_logs":all_logs}, 
                    Path(args.save) / "checkpoint.pt", logger=None)
        print(f"Saved checkpoint to {Path(args.save) / 'checkpoint.pt'}")
    for log in tqdm(all_logs, desc = "Logging search logs"):
        wandb.log(log)

def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, steps_per_epoch=None):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input_search, target_search) in tqdm(enumerate(valid_queue), desc = "Iterating over batches", total = len(valid_queue)):
        if steps_per_epoch is not None and step >= steps_per_epoch:
            break
        model.train()
       # n = input.size(0)
        
        
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda()

        architect.step(train_queue, input_search, target_search, lr, optimizer, args.unrolled)


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            if step > 101:
                break
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
                if args.debug:
                    break

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
