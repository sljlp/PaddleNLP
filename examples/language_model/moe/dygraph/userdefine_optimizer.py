# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import sys
import paddle
from paddle.optimizer import Optimizer
from paddle.nn import ClipGradByGlobalNorm
from paddle import framework
from paddle.static import Variable
from paddle.framework import core
from paddle.fluid import layers
from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients

__all__ = []


def _obtain_optimizer_parameters_list(optimizer):
    if getattr(optimizer, '_param_groups', None) and isinstance(
            optimizer._param_groups[0], dict):
        parameters_list = []
        for group in optimizer._param_groups:
            for param in group['params']:
                parameters_list.append(param)
    else:
        parameters_list = [param for param in optimizer._parameter_list]

    return parameters_list


class HybridParallelClipGrad:

    def __init__(self, clip, hcg):
        self._clip = clip
        self._hcg = hcg

    @paddle.no_grad
    def _dygraph_clip(self, params_grads):
        params_and_grads = []
        sum_square_list_dist = []
        sum_square_list_not_dist = []

        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                continue
            merge_grad = g
            if g.type == core.VarDesc.VarType.SELECTED_ROWS:
                merge_grad = layers.merge_selected_rows(g)
                merge_grad = layers.get_tensor_from_selected_rows(merge_grad)
            square = paddle.tensor.math.square(merge_grad)
            sum_square = paddle.sum(square)

            not_shared_enable = (not hasattr(p, 'is_firstly_shared')) or (
                hasattr(p, 'is_firstly_shared')
                and getattr(p, 'is_firstly_shared', True))

            if not_shared_enable:
                if p.is_distributed:
                    sum_square_list_dist.append(sum_square)
                else:
                    sum_square_list_not_dist.append(sum_square)

        global_norm_var_dist = paddle.concat(sum_square_list_dist) if len(
            sum_square_list_dist) != 0 else paddle.concat(
                [paddle.to_tensor([0.])])
        global_norm_var_dist = paddle.sum(global_norm_var_dist)

        global_norm_var_not_dist = paddle.concat(
            sum_square_list_not_dist
        ) if len(sum_square_list_not_dist) != 0 else paddle.concat(
            [paddle.to_tensor([0.])])
        global_norm_var_not_dist = paddle.sum(global_norm_var_not_dist)

        # add all reduce to get global norm of distributed params_and_grads
        if self._hcg.get_model_parallel_world_size() > 1:
            paddle.distributed.all_reduce(
                global_norm_var_dist,
                group=self._hcg.get_check_parallel_group())

        # add all reduce to get global norm of non-distributed params_and_grads in groups of pp
        if self._hcg.get_pipe_parallel_world_size() > 1:
            paddle.distributed.all_reduce(
                global_norm_var_not_dist,
                group=self._hcg.get_pipe_parallel_group())

        # In Sharding mode, param and grad is mapping different rank in optimizer.
        # ClipGradByGlobalNorm need allreduce to get globol norm
        if self._hcg.get_sharding_parallel_world_size() > 1:
            paddle.distributed.all_reduce(
                global_norm_var_not_dist,
                group=self._hcg.get_sharding_parallel_group())

        global_norm_var = paddle.sqrt(global_norm_var_dist +
                                      global_norm_var_not_dist)

        max_global_norm = paddle.full(shape=[1],
                                      dtype=global_norm_var.dtype,
                                      value=self.clip_norm)
        clip_var = paddle.divide(x=max_global_norm,
                                 y=paddle.maximum(x=global_norm_var,
                                                  y=max_global_norm))
        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                params_and_grads.append((p, g))
                continue
            new_grad = paddle.multiply(x=g, y=clip_var)
            params_and_grads.append((p, new_grad))

        return params_and_grads

    def __getattr__(self, item):
        return getattr(self._clip, item)

    def __call__(self, params_grads):
        return self._dygraph_clip(params_grads)


class MOEOptimizer:
    # adapter wrapper for optimizer
    def __init__(self, optimizer, hcg):
        self._inner_opt = optimizer
        self._hcg = hcg
        if isinstance(self._inner_opt._grad_clip, ClipGradByGlobalNorm):
            self._inner_opt._grad_clip = HybridParallelClipGrad(
                self._inner_opt._grad_clip, hcg)

    @paddle.no_grad
    @framework.dygraph_only
    def step(self):
        parameters_list = _obtain_optimizer_parameters_list(self._inner_opt)
        parameters_sync = []
        for param in parameters_list:
            # need_broadcast means not moe parameters
            if getattr(param, "no_sync", False):
                continue
            parameters_sync.append(param)

        fused_allreduce_gradients(list(parameters_sync), self._hcg)
        self._inner_opt.step()

    def __getattr__(self, item):
        return getattr(self._inner_opt, item)
