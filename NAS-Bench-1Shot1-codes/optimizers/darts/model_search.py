import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torch import nn
from pathlib import Path
import sys
lib_dir = (Path(__file__).parent / '..' / '..').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
from optimizers.darts.genotypes import PRIMITIVES
from optimizers.darts.operations import *
import numpy as np
import random
import copy
from copy import deepcopy
from typing import *

class NB101Arch(tuple):
  def __new__(self, args):
    instance = super().__new__(NB101Arch, args)
    return instance
  def tostr(self):
    return str(self)

class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            '''
            Not used in NASBench
            if 'pool' in primitive:
              op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            '''
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) if w > 0 else 0 for w, op in zip(weights, self._ops))


class ChoiceBlock(nn.Module):
    """
    Adapted to match Figure 3 in:
    Bender, Gabriel, et al. "Understanding and simplifying one-shot architecture search."
    International Conference on Machine Learning. 2018.
    """

    def __init__(self, C_in):
        super(ChoiceBlock, self).__init__()
        # Pre-processing 1x1 convolution at the beginning of each choice block.
        self.mixed_op = MixedOp(C_in, stride=1)

    def forward(self, inputs, input_weights, weights):
        if input_weights is not None:
            # Weigh the input to the choice block
            inputs = [w * t if w > 0 else 0 for w, t in zip(input_weights.squeeze(0), inputs)]

        # Sum input to choice block
        # https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_builder.py#L298
        input_to_mixed_op = sum(inputs)

        # Apply Mixed Op
        output = self.mixed_op(input_to_mixed_op, weights=weights)
        return output


class Cell(nn.Module):

    def __init__(self, steps, C_prev, C, layer, search_space):
        super(Cell, self).__init__()
        # All cells are normal cells in NASBench case.
        self._steps = steps

        self._choice_blocks = nn.ModuleList()
        self._bns = nn.ModuleList()
        self.search_space = search_space

        self._input_projections = nn.ModuleList()
        # Number of input channels is dependent on whether it is the first layer or not. Any subsequent layer has
        # C_in * (steps + 1) input channels because the output is a concatenation of the input tensor and all
        # choice block outputs
        C_in = C_prev if layer == 0 else C_prev * steps

        # Create the choice block and the input
        for i in range(self._steps):
            choice_block = ChoiceBlock(C_in=C)
            self._choice_blocks.append(choice_block)
            self._input_projections.append(ConvBnRelu(C_in=C_in, C_out=C, kernel_size=1, stride=1, padding=0))

        # Add one more input preprocessing for edge from input to output of the cell
        self._input_projections.append(ConvBnRelu(C_in=C_in, C_out=C * self._steps, kernel_size=1, stride=1, padding=0))

    def forward(self, s0, weights, output_weights, input_weights):
        # Adaption to NASBench
        # Only use a single input, from the previous cell
        states = []

        # Loop through the choice blocks of each cell
        for choice_block_idx in range(self._steps):
            # Select the current weighting for input edges to each choice block
            if input_weights is not None:
                # Node 1 has no choice with respect to its input
                if (choice_block_idx == 0) or (choice_block_idx == 1 and type(self.search_space) == SearchSpace1):
                    input_weight = None
                else:
                    input_weight = input_weights.pop(0)

            # Iterate over the choice blocks
            # Apply 1x1 projection only to edges from input of the cell
            # https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_builder.py#L289
            s = self._choice_blocks[choice_block_idx](inputs=[self._input_projections[choice_block_idx](s0), *states],
                                                      input_weights=input_weight, weights=weights[choice_block_idx])
            states.append(s)

        # Add projected input to the state
        # https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_builder.py#L328
        input_to_output_edge = self._input_projections[-1](s0)
        assert (len(input_weights) == 0, 'Something went wrong here.')

        if output_weights is None:
            tensor_list = states
        else:
            # Create weighted concatenation at the output of the cell
            tensor_list = [w * t for w, t in zip(output_weights[0][1:], states)]

        # Concatenate to form output tensor
        # https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_builder.py#L325
        return output_weights[0][0] * input_to_output_edge + torch.cat(tensor_list, dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, output_weights, search_space, steps=4):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._output_weights = output_weights
        self._op_names = PRIMITIVES # Some compatibility with NB201 code
        self._max_nodes = steps
        self.search_space = search_space

        # In NASBench the stem has 128 output channels
        C_curr = C
        self.stem = ConvBnRelu(C_in=3, C_out=C_curr, kernel_size=3, stride=1)

        self.cells = nn.ModuleList()
        C_prev = C_curr
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                # Double the number of channels after each down-sampling step
                # Down-sample in forward method
                C_curr *= 2
            cell = Cell(steps=self._steps, C_prev=C_prev, C=C_curr, layer=i, search_space=search_space)
            self.cells += [cell]
            C_prev = C_curr
        self.postprocess = ReLUConvBN(C_in=C_prev * self._steps, C_out=C_curr, kernel_size=1, stride=1, padding=0,
                                      affine=False)

        self.classifier = nn.Linear(C_prev, num_classes)
        self._initialize_alphas()

    def weights_parameters(self):
        xlist = list( self.stem.parameters() ) + list( self.cells.parameters() )
        xlist+= list( self.postprocess.parameters() )
        xlist+= list( self.classifier.parameters() )
        return xlist

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion,
                            steps=self.search_space.num_intermediate_nodes, output_weights=self._output_weights,
                            search_space=self.search_space).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _preprocess_op(self, x, discrete, normalize):
        if discrete and normalize:
            raise ValueError("architecture can't be discrete and normalized")
        # If using discrete architecture from random_ws search with weight sharing then pass through architecture
        # weights directly.
        if discrete:
            return x
        elif normalize:
            arch_sum = torch.sum(x, dim=-1)
            if arch_sum > 0:
                return x / arch_sum
            else:
                return x
        else:
            # Normal search softmax over the inputs and mixed ops.
            return F.softmax(x, dim=-1)

    def forward(self, input, discrete=False, normalize=False, updateType='alpha'):
        # NASBench only has one input to each cell
        s0 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if i in [self._layers // 3, 2 * self._layers // 3]:
                # Perform down-sampling by factor 1/2
                # Equivalent to https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_builder.py#L68
                s0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)(s0)

            if updateType == 'alpha':
                # Normalize mixed_op weights for the choice blocks in the graph
                mixed_op_weights = self._preprocess_op(self._arch_parameters[0], discrete=discrete, normalize=False)

                # Normalize the output weights
                output_weights = self._preprocess_op(self._arch_parameters[1], discrete,
                                                    normalize) if self._output_weights else None
                # Normalize the input weights for the nodes in the cell
                input_weights = [self._preprocess_op(alpha, discrete, normalize) for alpha in self._arch_parameters[2:]]
            elif updateType == 'weight':
                # keep weights
                mixed_op_weights = self._arch_parameters[0]
                output_weights = self._arch_parameters[1]
                input_weights = self._arch_parameters[2:]
            s0 = cell(s0, mixed_op_weights, output_weights, input_weights)

        # Include one more preprocessing step here
        s0 = self.postprocess(s0)  # [N, C_max * (steps + 1), w, h] -> [N, C_max, w, h]

        # Global Average Pooling by averaging over last two remaining spatial dimensions
        # https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_builder.py#L92
        out = s0.view(*s0.shape[:2], -1).mean(-1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    # def _initialize_alphas(self):
    #     # Initializes the weights for the mixed ops.
    #     num_ops = len(PRIMITIVES)
    #     self.alphas_mixed_op = torch.nn.parameter.Parameter(1e-3 * torch.randn(self._steps, num_ops).cuda(), requires_grad=True)

    #     # For the alphas on the output node initialize a weighting vector for all choice blocks and the input edge.
    #     self.alphas_output = torch.nn.parameter.Parameter(1e-3 * torch.randn(1, self._steps + 1).cuda(), requires_grad=True)

    #     if type(self.search_space) == SearchSpace1:
    #         begin = 3
    #     else:
    #         begin = 2
    #     # Initialize the weights for the inputs to each choice block.
    #     self.alphas_inputs = [torch.nn.parameter.Parameter(1e-3 * torch.randn(1, n_inputs).cuda(), requires_grad=True) for n_inputs in
    #                           range(begin, self._steps + 1)]

    #     # Total architecture parameters
    #     self._arch_parameters = torch.nn.ParameterList([
    #         self.alphas_mixed_op,
    #         self.alphas_output,
    #         *self.alphas_inputs
    #     ])
    
    def _initialize_alphas(self):
        # Initializes the weights for the mixed ops.
        num_ops = len(PRIMITIVES)
        self.alphas_mixed_op = Variable(1e-3 * torch.randn(self._steps, num_ops).cuda(), requires_grad=True)

        # For the alphas on the output node initialize a weighting vector for all choice blocks and the input edge.
        self.alphas_output = Variable(1e-3 * torch.randn(1, self._steps + 1).cuda(), requires_grad=True)

        if type(self.search_space) == SearchSpace1:
            begin = 3
        else:
            begin = 2
        # Initialize the weights for the inputs to each choice block.
        self.alphas_inputs = [Variable(1e-3 * torch.randn(1, n_inputs).cuda(), requires_grad=True) for n_inputs in
                              range(begin, self._steps + 1)]

        # Total architecture parameters
        self._arch_parameters = [
            self.alphas_mixed_op,
            self.alphas_output,
            *self.alphas_inputs
        ]

    def arch_parameters(self):
        return self._arch_parameters
    
    def arch_params(self):
        return self.arch_parameters()


    ###### SDARTS
    def _save_arch_parameters(self):
        self._saved_arch_parameters = [p.clone() for p in self._arch_parameters]
    
    def _save_parameters(self):
        self._saved_parameters = [p.clone() for p in self.parameters()]
      
    
    def softmax_arch_parameters(self, save=True):
        if save:
            self._save_arch_parameters()
        for p in self._arch_parameters:
            p.data.copy_(F.softmax(p, dim=-1))
            
    def restore_arch_parameters(self):
        for i, p in enumerate(self._arch_parameters):
            p.data.copy_(self._saved_arch_parameters[i])
        del self._saved_arch_parameters
    
    def restore_parameters(self):
        for i, p in enumerate(self.parameters()):
            p.data.copy_(self._saved_parameters[i])
        del self._saved_parameters
    
    def clip(self):
        for p in self.arch_parameters():
            for line in p:
                max_index = line.argmax()
                line.data.clamp_(0, 1)
                if line.sum() == 0.0:
                    line.data[max_index] = 1.0
                line.data.div_(line.sum())

class NetworkNB101(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, output_weights, search_space, steps=4, discrete=True):
      super(NetworkNB101, self).__init__()
      self._C = C
      self._num_classes = num_classes
      self._layers = layers
      self._criterion = criterion
      self._steps = steps
      self._output_weights = output_weights
      self.search_space = search_space
      self._op_names = PRIMITIVES

      # In NASBench the stem has 128 output channels
      C_curr = C
      self.stem = ConvBnRelu(C_in=3, C_out=C_curr, kernel_size=3, stride=1)

      self.cells = nn.ModuleList()
      C_prev = C_curr
      for i in range(layers):
          if i in [layers // 3, 2 * layers // 3]:
              # Double the number of channels after each down-sampling step
              # Down-sample in forward method
              C_curr *= 2
          cell = Cell(steps=self._steps, C_prev=C_prev, C=C_curr, layer=i, search_space=search_space)
          self.cells += [cell]
          C_prev = C_curr
      self.postprocess = ReLUConvBN(C_in=C_prev * self._steps, C_out=C_curr, kernel_size=1, stride=1, padding=0,
                                    affine=False)

      self.classifier = nn.Linear(C_prev, num_classes)
      self._initialize_alphas()
      
      # NEWLY ADDED STUFF to match up to other NB201 code
      self._mode        = None
      self.dynamic_cell = None
      self._tau         = None
      self._algo        = None
      self._drop_path   = None
      self.verbose      = False
      self.logits_only = False
      self.arch_sampler = None
      self.discrete = discrete
      self._max_nodes = 4 # TODO should be the same I think?
      print(f"Instantiated DARTS model from NB101 with discrete={discrete}")
      
  # def new(self):
  #   model_new = NetworkNB(self._C, self._num_classes, self._layers, self._criterion).cuda()
  #   for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
  #       x.data.copy_(y.data)
  #   return model_new

  def forward(self, input, discrete=False, normalize=False):
      # NASBench only has one input to each cell
      s0 = self.stem(input)
      for i, cell in enumerate(self.cells):
          if i in [self._layers // 3, 2 * self._layers // 3]:
              # Perform down-sampling by factor 1/2
              # Equivalent to https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_builder.py#L68
              s0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)(s0)

          # Normalize mixed_op weights for the choice blocks in the graph
          mixed_op_weights = self._preprocess_op(self._arch_parameters[0], discrete=discrete, normalize=False)

          # Normalize the output weights
          output_weights = self._preprocess_op(self._arch_parameters[1], discrete,
                                              normalize) if self._output_weights else None
          # Normalize the input weights for the nodes in the cell
          input_weights = [self._preprocess_op(alpha, discrete, normalize) for alpha in self._arch_parameters[2:]]

              
          s0 = cell(s0, mixed_op_weights, output_weights, input_weights)

      # Include one more preprocessing step here
      s0 = self.postprocess(s0)  # [N, C_max * (steps + 1), w, h] -> [N, C_max, w, h]

      # Global Average Pooling by averaging over last two remaining spatial dimensions
      # https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_builder.py#L92
      out = s0.view(*s0.shape[:2], -1).mean(-1)
      logits = self.classifier(out.view(out.size(0), -1))
      
      if self.logits_only:
        return logits
      else:
        return "placeholder", logits
  def _preprocess_op(self, x, discrete, normalize):
      if discrete and normalize:
          raise ValueError("architecture can't be discrete and normalized")
      # If using discrete architecture from random_ws search with weight sharing then pass through architecture
      # weights directly.
      if discrete or self.discrete:
          return x
      elif normalize:
          arch_sum = torch.sum(x, dim=-1)
          if arch_sum > 0:
              return x / arch_sum
          else:
              return x
      else:
          # Normal search softmax over the inputs and mixed ops.
          return F.softmax(x, dim=-1)
  def _initialize_alphas(self):
      # Initializes the weights for the mixed ops.
      num_ops = len(PRIMITIVES)
      self.alphas_mixed_op = torch.nn.parameter.Parameter(1e-3 * torch.randn(self._steps, num_ops).cuda(), requires_grad=True)

      # For the alphas on the output node initialize a weighting vector for all choice blocks and the input edge.
      self.alphas_output = torch.nn.parameter.Parameter(1e-3 * torch.randn(1, self._steps + 1).cuda(), requires_grad=True)

      if type(self.search_space) == SearchSpace1:
          begin = 3
      else:
          begin = 2
      # Initialize the weights for the inputs to each choice block.
      self.alphas_inputs = [torch.nn.parameter.Parameter(1e-3 * torch.randn(1, n_inputs).cuda(), requires_grad=True) for n_inputs in
                            range(begin, self._steps + 1)]

      # Total architecture parameters
      self._arch_parameters = torch.nn.ParameterList([
          self.alphas_mixed_op,
          self.alphas_output,
          *self.alphas_inputs
      ])

  def arch_parameters(self):
    return self._arch_parameters

  def set_algo(self, algo):
    # used for searching
    assert self._algo is None, 'This functioin can only be called once.'
    self._algo = algo
    # if algo == 'enas':
    #   self.controller = Controller(self.edge2index, self._op_names, self._max_nodes)
    # else:
    #   self.arch_parameters = nn.Parameter( 1e-3*torch.randn(self._num_edge, len(self._op_names)) )
    if algo == 'gdas':
        self._tau         = 10

  def set_drop_path(self, progress, drop_path_rate):
    pass
  def arch_params(self):
    return self.alphas
  @property
  def mode(self):
    return self._mode
  def set_tau(self, tau):
    self._tau = tau

  @property
  def tau(self):
    return self._tau
  @property
  def drop_path(self):
    return self._drop_path

  @property
  def weights(self):
    return self.get_weights()
  
  @property
  def alphas(self):
    return self.get_alphas()

  @property
  def message(self):
    string = self.extra_repr()
    for i, cell in enumerate(self._cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self._cells), cell.extra_repr())
    return string

      
  def return_topK(self, K, use_random=False, size_percentile=None, perf_percentile=None, api=None, dataset=None):
    """NOTE additionaly outputs perf/size_all_dict.pkl mainly with shape {arch_str: perf_metric} """

    sampled = [self.random_topology_func() for _ in range(K)]
    return sampled

  def set_cal_mode(self, mode, dynamic_cell=None, sandwich_cells=None):
    assert mode in ['gdas', 'enas', 'urs', 'joint', 'select', 'dynamic', 'sandwich']
    self._mode = mode
    if mode == 'dynamic':
      # self.dynamic_cell = deepcopy(dynamic_cell)
      weights = self.get_weights_from_arch(dynamic_cell)
      self.set_model_weights(weights)
      self.discrete = True
      
    else                : self.dynamic_cell = None
    if mode == "sandwich":
      assert sandwich_cells is not None
      self.sandwich_cells = sandwich_cells
    
    if mode == "joint":
        self.discrete = False

  @property
  def arch_parameters(self):
    return self.get_alphas()

  @property
  def genotype(self) -> Dict[Text, List]:
    return "TODO"
  
  def get_genotype(self):
    return "TODO"
    
  def random_topology_func(self, k=1, **kwargs):
    archs = [self.sample_arch() for _ in range(k)]
    if k == 1:
      return archs[0]
    else:
      return archs

  def get_weights(self) -> List[torch.nn.Parameter]:
    xlist = list( self.stem.parameters() ) + list( self.cells.parameters() )
    xlist+= list( self.postprocess.parameters() )
    xlist+= list( self.classifier.parameters() )
    return xlist

  def get_alphas(self) -> List[torch.nn.Parameter]:
    return self._arch_parameters

  def show_alphas(self):
    return "TODO"

  def get_message(self):
    string = self.extra_repr()
    for i, cell in enumerate(self._cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self._cells), cell.extra_repr())
    return string

  def extra_repr(self):
    return ("NetworkNB101 model (TODO get better repr)")

  ### THIS PART IS FROM RANDOM_NAS
  def get_weights_from_arch(self, arch):
      adjacency_matrix, node_list = arch
      num_ops = len(PRIMITIVES)

      # Assign the sampled ops to the mixed op weights.
      # These are not optimized
      alphas_mixed_op = torch.nn.parameter.Parameter(torch.zeros(self._steps, num_ops).cuda(), requires_grad=False)
      for idx, op in enumerate(node_list):
          alphas_mixed_op[idx][PRIMITIVES.index(op)] = 1

      # Set the output weights
      alphas_output = torch.nn.parameter.Parameter(torch.zeros(1, self._steps + 1).cuda(), requires_grad=False)
      for idx, label in enumerate(list(adjacency_matrix[:, -1][:-1])):
          alphas_output[0][idx] = label

      # Initialize the weights for the inputs to each choice block.
      if type(self.search_space) == SearchSpace1:
          begin = 3
      else:
          begin = 2
      alphas_inputs = [torch.nn.parameter.Parameter(torch.zeros(1, n_inputs).cuda(), requires_grad=False) for n_inputs in
                        range(begin, self._steps + 1)]
      for alpha_input in alphas_inputs:
          connectivity_pattern = list(adjacency_matrix[:alpha_input.shape[1], alpha_input.shape[1]])
          for idx, label in enumerate(connectivity_pattern):
              alpha_input[0][idx] = label

      # Total architecture parameters
      arch_parameters = torch.nn.ParameterList([
          alphas_mixed_op,
          alphas_output,
          *alphas_inputs
      ])
      return arch_parameters

  def set_model_weights(self, weights):
    # weights should be a tuple of (normal_weights, reduce_weights)
    self.set_arch_model_weights(weights)
    
  def set_arch_model_weights(self, weights):
      self._arch_parameters = weights
      
  def sample_arch(self):
      # The arch is meant to be a tuple of those two things already, so just do arch = .sample_arch()
      adjacency_matrix, op_list = self.search_space.sample(with_loose_ends=True, upscale=False)
      return NB101Arch((adjacency_matrix, op_list))
