import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.utils
from copy import deepcopy
import utils
import logging
criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)
        self.args = args


    def _compute_unrolled_model_sotl(self, train_queue, eta, network_optimizer):

        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()    

        train_queue_iter = iter(train_queue)
        so_grad = [torch.zeros_like(p) for p in self.model.parameters()]
        fo_grad = [torch.zeros_like(p) for p in self.model.arch_parameters()]

        all_inputs, all_targets = [], []
        
        
        for step in range(self.args.T):
            input, target = next(train_queue_iter)		
            all_inputs.append(input)
            all_targets.append(target)

            self.model.train()
            n = input.size(0)

            input = Variable(input, requires_grad=False).cuda()
            target = Variable(target, requires_grad=False).cuda()

            network_optimizer.zero_grad()
            logits = self.model(input)
            loss = criterion(logits, target)

            loss.backward(create_graph=True)
            nn.utils.clip_grad_norm(self.model.parameters(), 5)
            
            if self.args.sotl_order == "second":
                k = max(0, step-1) # t=2 corresponds to second-order DARTS, which still has no I-Hessian term. So for T=4 unrolling steps we only have the complicated gradient for the last two losses 
                if k == 0:
                    grads_2 = [torch.zeros_like(p) for p in self.model.parameters()]
                else:
                    grad_norm=0
                    for p in self.model.parameters(): # Empirical Fisher to approximate Hessian in formula
                        grad = p.grad
                        grad_norm +=grad.pow(2).sum()
                    loss_last=grad_norm.sqrt()
                    loss_last.backward()
                    
                    if k == 1:
                        grads_2 = [(1-eta*v.grad.data).pow(k) for v in self.model.parameters()]        #######consider the approximation with only the diagonal elements
                    else:
                        grads_2 = [(1-eta*v.grad.data)*g for g,v in zip(so_grad, self.model.parameters())]
                with torch.no_grad():
                    for g1, g2 in zip(so_grad, grads_2):
                        g1.data += g2.data
            # if not (self.args.sotl_order == "second" and step == self.args.T - 1):
            with torch.no_grad():
                for g1, p in zip(fo_grad, self.model.arch_parameters()):
                    g1.data += p.grad.data
            
            network_optimizer.step()
            
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)
        
        self.model.zero_grad()
        with torch.no_grad():
            for p in self.model.parameters():
                p.grad = None
        # unrolled_model = deepcopy(self.model)
        
        unrolled_model = self.model.new()
        unrolled_model.load_state_dict(self.model.state_dict())
        
        unrolled_network_optimizer = deepcopy(network_optimizer)

        return unrolled_model.cuda(), so_grad, fo_grad

    def _compute_unrolled_model(self, train_queue, eta, network_optimizer):

        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()    

        train_queue_iter = iter(train_queue)
        #for step, (input, target) in enumerate(train_queue):
        
        fo_grad = [torch.zeros_like(p) for p in self.model.arch_parameters()]
        
        for step in range(self.args.T):
            input, target = next(train_queue_iter)		


            self.model.train()
            n = input.size(0)

            input = Variable(input, requires_grad=False).cuda()
            target = Variable(target, requires_grad=False).cuda()

            network_optimizer.zero_grad()
            logits = self.model(input)
            loss = criterion(logits, target)

            loss.backward()
            nn.utils.clip_grad_norm(self.model.parameters(), 5)
            network_optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)
            
            # if not (step == self.args.T - 1):
            with torch.no_grad():
                for g1, g2 in zip(fo_grad, self.model.arch_parameters()):
                    g1.add_(g2)
                

        try:
            model_last=deepcopy(self.model)
        except:
            print("Failed once")
            try:
                model_last = deepcopy(self.model)
            except:
                print("Failed twice")
                model_last = deepcopy(self.model)

        logits = model_last(input)
        loss_l1 = criterion(logits, target)
        grads_1 = torch.autograd.grad(loss_l1, model_last.parameters(), create_graph=True)#[0]

        grad_norm=0
        for grad in grads_1: # Empirical Fisher to approximate Hessian in formula
            grad_norm +=grad.pow(2).sum()
        loss_last=grad_norm.sqrt()
        loss_last.backward()

        # grads_2 = [(1+(1-eta*v.grad.data)+(1-eta*v.grad.data).pow(2)) for v in model_last.parameters()]        #######consider the approximation with only the diagonal elements
        num_K = self.args.K+1 # The +1 is so that the range() takes values in [0, .., K]
        
        grads_2 = [sum([(1-eta*v.grad.data).pow(k) for k in range(num_K)]) for v in model_last.parameters()]        #######consider the approximation with only the diagonal elements

        
        del model_last
        unrolled_model = deepcopy(self.model)
        unrolled_network_optimizer = deepcopy(network_optimizer)

        return unrolled_model.cuda(), grads_2, fo_grad

    def step(self, train_queue, input_valid, target_valid, eta, network_optimizer, unrolled):
        self.optimizer.zero_grad()

        if unrolled:
            self._backward_step_unrolled(train_queue, input_valid, target_valid, eta, network_optimizer)
        else:
            self._backward_step(input_valid, target_valid)

        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        #print(loss)
        loss.backward()

    def _backward_step_unrolled(self, train_queue, input_valid, target_valid, eta, network_optimizer):
        if not self.args.sotl_order or self.args.sotl_order == "basic":
            unrolled_model, grads_2, arch_fo_grad = self._compute_unrolled_model(train_queue, eta, network_optimizer)######copy a model for the L_val, since the model should nog trained by validation data
        elif self.args.sotl_order in ["second", "first"]:
            unrolled_model, grads_2, arch_fo_grad = self._compute_unrolled_model_sotl(train_queue, eta, network_optimizer)######copy a model for the L_val, since the model should nog trained by validation data

        
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)
        #print(unrolled_loss)
        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]

        del unrolled_model
        implicit_grads = self._hessian_vector_product(vector, train_queue, grads_2)######this should be L_train(w*,a), so the data should be train
        
        
        # print(arch_fo_grad)
        if self.args.sotl_order in ["first", "second", "basic"]:
            with torch.no_grad():
                for g1, g2 in zip(implicit_grads, arch_fo_grad):
                    g1.add_(g2)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _hessian_vector_product(self, vector, train_queue, grads_2, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v, g in zip(self.model.parameters(), vector, grads_2):
            p.data.add_(R, v*g)

        input, target = next(iter(train_queue))     
        n = input.size(0)
        input = Variable(input).cuda()
        target = Variable(target).cuda()
        loss = self.model._loss(input, target)                


        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())
        #print(grads_p)

        for p, v, g in zip(self.model.parameters(), vector, grads_2):
            p.data.sub_(2*R, v*g)

        loss = self.model._loss(input, target)                 
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        # NOTE doesnt seem to work?
        # for p, v in zip(self.model.parameters(), vector):###We empirically found adding noise for supernet training could improve performance.
        #     p.data.add_(R, v*g)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

    
