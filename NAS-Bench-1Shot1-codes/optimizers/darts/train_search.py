import argparse
import glob
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset

lib_dir = (Path(__file__).parent / '..' / '..').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))

lib_dir = (Path(__file__).parent / '..' / '..' / '..' / '..'/ '..' / '..' /'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(1, str(lib_dir))
from datasets import get_datasets, get_nas_search_loaders

from pathlib import Path

import wandb
from nasbench import api
from nasbench_analysis import eval_darts_one_shot_model_in_nasbench as naseval
from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2
from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3
from nasbench_analysis.utils import NasbenchWrapper
from optimizers.darts import utils
from optimizers.darts.architect import Architect
from optimizers.darts.model_search import Network
from optimizers.sotl_utils import wandb_auth
from tqdm import tqdm
from numpy import linalg as LA

from utils.train_loop import approx_hessian, exact_hessian

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
parser.add_argument('--seed', type=int, default=2, help='random_ws seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training darts')
parser.add_argument('--unrolled', type=lambda x: False if x in ["False", "false", "", "None", False, None] else True, default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--output_weights', type=bool, default=True, help='Whether to use weights on the output nodes')
parser.add_argument('--search_space', choices=['1', '2', '3'], default='1')
parser.add_argument('--debug', action='store_true', default=False, help='run only for some batches')
parser.add_argument('--warm_start_epochs', type=int, default=0,
                    help='Warm start one-shot model before starting architecture updates.')
parser.add_argument('--steps_per_epoch', type=int, default=None,
                    help='Warm start one-shot model before starting architecture updates.')
parser.add_argument('--hessian', type=lambda x: False if x in ["False", "false", "", "None", False, None] else True, default=True,
                    help='Warm start one-shot model before starting architecture updates.')
parser.add_argument('--dataset', type=str, default="cifar10",
                    help='Warm start one-shot model before starting architecture updates.')
parser.add_argument('--perturb_alpha', type=str, default=None, help='portion of training data')
parser.add_argument('--epsilon_alpha', type=float, default=0.3, help='max epsilon for alpha')


parser.add_argument('--total_samples',          type=int, default=None, help='Number of total samples in dataset. Useful for limiting Cifar5m')
parser.add_argument('--data_path'   ,       type=str, default="$TORCH_HOME/cifar.python",   help='Path to dataset')
parser.add_argument('--mmap',          type=str, default="r", help='Whether to mmap cifar5m')


args = parser.parse_args()

args.save = 'experiments/darts/search_space_{}/search-{}-{}-{}-{}'.format(args.search_space, args.save,
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

logger = logging.getLogger()

CIFAR_CLASSES = 10

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
    
    wandb_auth()
    run = wandb.init(project="NAS", group=f"Search_Cell_nb101", reinit=True)
    wandb.config.update(args)


    try:
        nasbench = NasbenchWrapper(os.path.join(get_torch_home() ,'nasbench_only108.tfrecord'))
    except:
        nasbench = NasbenchWrapper(os.path.join(get_torch_home() ,'nasbench_full.tfrecord'))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, output_weights=args.output_weights,
                    steps=search_space.num_intermediate_nodes, search_space=search_space)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.weights_parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    if args.dataset == "cifar10" or args.dataset == "cifar100":
        if args.dataset == "cifar10":
          train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        elif args.dataset == "cifar100":
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
    elif args.dataset == "cifar5m":
        train_data, valid_data, xshape, class_num = get_datasets(args.dataset, args.data_path, -1, mmap=args.mmap, total_samples=args.total_samples)
        _, train_queue, valid_queue = get_nas_search_loaders(train_data, valid_data, args.dataset, 'configs/nas-benchmark/', 
            (args.batch_size, args.batch_size), workers=0, 
            epochs=args.epochs, determinism="all", 
            merge_train_val = False, merge_train_val_and_use_test = False, 
            extra_split = True, valid_ratio=1, use_only_train=True, xargs=args)
        train_queue.sampler.auto_counter = True
        valid_queue.sampler.auto_counter = True

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)
    
    # if os.path.exists(Path(args.save) / "checkpoint.pt"):
    #     checkpoint = torch.load(Path(args.save) / "checkpoint.pt")
    #     optimizer.load_state_dict(checkpoint["w_optimizer"])
    #     architect.optimizer.load_state_dict(checkpoint["a_optimizer"])
    #     model.load_state_dict(checkpoint["model"])
    #     scheduler.load_state_dict(checkpoint["w_scheduler"])
    #     start_epoch = checkpoint["epoch"]
    #     all_logs = checkpoint["all_logs"]

    # else:
    #     print(f"Path at {Path(args.save) / 'checkpoint.pt'} does not exist")
    #     start_epoch=0
    #     all_logs=[]
    start_epoch = 0
    all_logs = []
        
    for epoch in tqdm(range(start_epoch, args.epochs), desc = "Iterating over epochs", total = args.epochs - start_epoch):
        
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

        logging.info(f'architecture : {numpy_tensor_list}')
        
        if args.perturb_alpha:
          epsilon_alpha = 0.03 + (args.epsilon_alpha - 0.03) * epoch / args.epochs
          logging.info('epoch %d epsilon_alpha %e', epoch, epsilon_alpha)
        else:
          epsilon_alpha = None
          
        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, perturb_alpha=utils.Random_alpha, epsilon_alpha=epsilon_alpha)
        logging.info('train_acc %f', train_acc)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)
        
        genotype_perf, _, _, _ = naseval.eval_one_shot_model(config=args.__dict__,
                                                               model=arch_filename, nasbench=nasbench)
        print(f"Genotype performance: {genotype_perf}" )
        
        if args.hessian == "analyzer":
          analyzer = Analyzer(model, args)
          
          _data_loader = deepcopy(train_queue)
          input, target = next(iter(_data_loader))

          input = input.cuda()
          target = target.cuda(non_blocking=True)
          
          H = analyzer.compute_Hw(input, target, input_search, target_search,
                            lr, optimizer, False)
          # g = analyzer.compute_dw(input, target, input_search, target_search,
          #                         lr, optimizer, False)
          # g = torch.cat([x.view(-1) for x in g])

          del _data_loader
          
          ev = max(LA.eigvals(H.cpu().data.numpy()))
          ev = np.linalg.norm(ev)
          eigenvalues = {"eigval": ev}

        elif args.hessian and torch.cuda.get_device_properties(0).total_memory < 15147483648:
            eigenvalues = approx_hessian(network=model, val_loader=valid_queue, criterion=criterion, xloader=valid_queue, args=args)
            # eigenvalues = exact_hessian(network=model, val_loader=valid_queue, criterion=criterion, xloader=valid_queue, epoch=epoch, logger=logger, args=args)
        elif args.hessian and torch.cuda.get_device_properties(0).total_memory > 15147483648:
            eigenvalues = exact_hessian(network=model, val_loader=valid_queue, criterion=criterion, xloader=valid_queue, epoch=epoch, logger=logger, args=args)

        else:
            eigenvalues = None
        
        
        wandb_log = {"train_acc":train_acc, "train_loss":train_obj, "val_acc": valid_acc, "valid_loss":valid_obj,
                     "search.final.cifar10": genotype_perf, "epoch":epoch, "eigval":eigenvalues}
        all_logs.append(wandb_log)
        wandb.log(wandb_log)
        
        utils.save_checkpoint2({"model":model.state_dict(), "w_optimizer":optimizer.state_dict(), 
                           "a_optimizer":architect.optimizer.state_dict(), "w_scheduler":scheduler.state_dict(), "epoch": epoch, 
                           "all_logs":all_logs}, 
                          Path(args.save) / "checkpoint.pt", logger=None)
        print(f"Saved checkpoint to {Path(args.save) / 'checkpoint.pt'}")
        # utils.save(model, os.path.join(args.save, 'weights.pt'))

    logging.info('STARTING EVALUATION')
    test, valid, runtime, params = naseval.eval_one_shot_model(config=args.__dict__,
                                                               model=arch_filename, nasbench=nasbench)
    index = 0
    logging.info('TEST ERROR: %.3f | VALID ERROR: %.3f | RUNTIME: %f | PARAMS: %d'
                 % (test,
                    valid,
                    runtime,
                    params)
                 )
    wandb.log({"test_error":test, "valid_error": valid, "runtime":runtime, "params":params})
    for log in tqdm(all_logs, desc = "Logging search logs"):
        wandb.log(log)

def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, perturb_alpha=None, epsilon_alpha=None):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = input.cuda()
        target = target.cuda(non_blocking=True)

        # get a minibatch from the search queue with replacement
        try:
            input_search, target_search = next(valid_queue_iter)
        except:
            valid_queue_iter = iter(valid_queue)
            input_search, target_search = next(valid_queue_iter)

        input_search = input_search.cuda()
        target_search = target_search.cuda(non_blocking=True)

        # Allow for warm starting of the one-shot model for more reliable architecture updates.
        if epoch >= args.warm_start_epochs:
            architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        if args.perturb_alpha:
            # print('before softmax', model.arch_parameters())
            model.softmax_arch_parameters()
                
            #perturb on alpha
            # print('after softmax', model.arch_parameters())
            perturb_alpha(model, input, target, epsilon_alpha)
            optimizer.zero_grad()
            architect.optimizer.zero_grad()
            # print('afetr perturb', model.arch_parameters())

        optimizer.zero_grad()
        logits = model(input, updateType = 'weight' if args.perturb_alpha else 'alpha')
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if step == 0:
            print(f"Arch params before step for debugging if they change: {model.alphas_mixed_op[0]}")
        optimizer.step()
        if step == 0:
            print(f"Arch params after step for debugging if they change: {model.alphas_mixed_op[0]}")
            
        if args.perturb_alpha:
          model.restore_arch_parameters()
        # print('after restore', model.arch_parameters())
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            if args.debug:
                break

    return top1.avg, objs.avg


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
