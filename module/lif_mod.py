

'''This file contains an implementation of the modified LIFneuron suggested in'Unsupervised Learning of a Hierarchical Spiking Neural Network for Optical Flow Estimation: From Events to Global Motion Perception'''

from typing import Tuple
import numpy as np
import torch

from norse.torch.functional.lif import LIFState, LIFFeedForwardState
from norse.torch.functional.lif_refrac import LIFRefracState, LIFRefracFeedForwardState

from functional.lif_mod2 import (
    LIFModParameters,
    LIFModParametersNeuronTrace,
    LIFModState,
    LIFModStateNeuronTrace,
    LIFModFeedForwardStateNT,
    LIFModFeedForwardStateWTA,
    lif_mod_feed_forward_step_WTA,
    lif_mod_feed_forward_step_NT
)

from collections import deque

class LIFModFeedForwardCell(torch.nn.Module):
    """Module that computes a single euler-integration step of a modified
    LIF neuron-model with absolute refractory period. More specifically
    it implements one integration step of the following ODE.

    .. math::
        \\begin{align*}
            \dot{v} &= 1/\\tau_{\\text{mem}} (1-\Theta(\\rho)) \
            (v_{\\text{leak}} - v + i) \\\\
            \dot{i} &= -1/\\tau_{\\text{syn}} i \\\\
            \dot{\\rho} &= -1/\\tau_{\\text{refrac}} \Theta(\\rho)
        \end{align*}

    together with the jump condition

    .. math::
        \\begin{align*}
            z &= \Theta(v - v_{\\text{th}}) \\\\
            z_r &= \Theta(-\\rho)
        \end{align*}

    and transition equations

    .. math::
        \\begin{align*}
            v &= (1-z) v + z v_{\\text{reset}} \\\\
            \\rho &= \\rho + z_r \\rho_{\\text{reset}}
        \end{align*}

    Parameters:
        shape: Shape of the processed spike input
        p (LIFRefracParameters): parameters of the lif neuron
        dt (float): Integration timestep to use

    Examples:
        >>> batch_size = 16
        >>> lif = LIFRefracFeedForwardCell((20, 30))
        >>> input = torch.randn(batch_size, 20, 30)
        >>> s0 = lif.initial_state(batch_size)
        >>> output, s0 = lif(input, s0)
    """

    def __init__(
        self, shape, p: LIFModParameters = LIFModParameters(), dt: float = 10**(-3),
    ):
        super(LIFModFeedForwardCell, self).__init__()
        self.shape = shape
        self.p = p
        self.dt = dt

    def initial_state_NT(self, batch_size, device, dtype) -> LIFModFeedForwardStateNT:
        return LIFModFeedForwardStateNT(
            LIFRefracFeedForwardState(
            LIFFeedForwardState(
                v=torch.zeros(batch_size, *self.shape, device=device, dtype=dtype),
                i=torch.zeros(batch_size, *self.shape, device=device, dtype=dtype),
            ),
            rho=torch.zeros(batch_size, *self.shape, device=device, dtype=dtype)),
            X=torch.zeros(batch_size, self.shape[0], self.p.lifMod.m, self.shape[1], self.shape[2],  device=device, dtype=dtype),
            buffer = deque(torch.zeros(int(max(self.p.lifMod.delays)*10**(-3)/self.dt) + 2, batch_size,  *self.shape, device=device, dtype=dtype))
        )

    def initial_state_WTA(self, batch_size, len_stiffness_buffer, device, dtype) -> LIFModFeedForwardStateWTA:
        return LIFModFeedForwardStateWTA(
            LIFRefracFeedForwardState(
            LIFFeedForwardState(
                v=torch.zeros(batch_size, *self.shape, device=device, dtype=dtype),
                i=torch.zeros(batch_size, *self.shape, device=device, dtype=dtype),
            ),
            rho=torch.zeros(batch_size, *self.shape, device=device, dtype=dtype)),
            buffer = deque(torch.zeros(int(max(self.p.delays)*10**(-3)/self.dt) + 2, batch_size,  *self.shape, device=device, dtype=dtype)),
            stiffness_buffer = torch.zeros(len_stiffness_buffer, batch_size, *self.shape, device=device, dtype=dtype)
            #buffer = deque(torch.zeros(2000, batch_size,  *self.shape, device=device, dtype=dtype))
        )

    def forward_NT(
        self, batch_size: torch.Tensor, i_new: torch.Tensor, state: LIFModFeedForwardStateNT, k: torch.Tensor, s:torch.Tensor, device:torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, LIFModFeedForwardStateNT]:
        return lif_mod_feed_forward_step_NT(batch_size, i_new, state, k,s,device, shape = self.shape, p=self.p, dt=self.dt)

    def forward_WTA(
        self, X: torch.Tensor, batch_size: torch.Tensor, i_new: torch.Tensor, state: LIFModFeedForwardStateWTA, k: torch.Tensor, s:torch.Tensor, device:torch.device, t, training = False
    ) -> Tuple[torch.Tensor, torch.Tensor, LIFModFeedForwardStateWTA]:
        return lif_mod_feed_forward_step_WTA(X, batch_size, i_new, state, k,s,device, training, t,  shape = self.shape, p=self.p, dt=self.dt)
