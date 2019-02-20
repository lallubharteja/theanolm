#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the freeze layer.
"""

import logging

import numpy
import theano
import theano.tensor as tensor

from theanolm.network.basiclayer import BasicLayer

class FreezeLayer(BasicLayer):
    """Freeze Layer

    A freeze layer is not a regular layer in the sense that it doesn't contain
    any neurons. It simply freezes the weights matrices occuring before it. It
    provides two parameters:
     - switch: If set as True, all the weights before this layer are assumed to 
       be frozen (are not retrained) otherwise these weights are trained 
       normally
     - init: Initialization for all the layers after the freeze layer can be
       controlled using this parameter. If set True and weights file is give,
       the weights for the non-frozen layers are initialized from the given
       weights file otherwise these weights are randomly initialized
    """

    def __init__(self, layer_options, *args, **kwargs):
        """Validates the parameters of this layer.
        """

        super().__init__(layer_options, *args, **kwargs)

        if 'switch' in layer_options:
            self._switch = True if layer_options['switch'] == "True" else False
        else:
            self._switch = False

        if self._switch:
            logging.debug("  Freeze layer is switched on.")
        else:
            logging.debug("  Freeze layer is switched off")

        if 'init' in layer_options:
            self._init = True if layer_options['init'] == "True" else False
        else:
            self._init = False

        # Make sure the user hasn't tried to change the number of connections.
        input_size = sum(x.output_size for x in self._input_layers)
        output_size = self.output_size
        if input_size != output_size:
            raise ValueError("Freeze layer size has to match the previous "
                             "layer.")

        self.output = None

    def create_structure(self):
        """Creates the symbolic graph of this layer.

        Sets ``self.output`` to a symbolic matrix that describes the output of
        this layer. During training sets randomly some of the outputs to zero.
        """

        self.output = tensor.concatenate([x.output for x in self._input_layers],
                                         axis=2)
