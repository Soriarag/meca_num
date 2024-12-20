#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# This file is part of SimulationTeachingElan, a python code used for teaching at Elan Inria.
#
# Copyright 2020 Mickael Ly <mickael.ly@inria.fr> (Elan / Inria - Université Grenoble Alpes)
# SimulationTeachingElan is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SimulationTeachingElan is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with SimulationTeachingElan.  If not, see <http://www.gnu.org/licenses/>.
#

import numpy as np
from graphics import *
from geom import *

from .abstract_dynamic_system import AbstractDynamicSystem

def f(X, g, l):
    return np.array(X[1], -g/l * np.sin(X[0]))

## Dummy dynamic system just to test
class PendulumDynamicSystem(AbstractDynamicSystem):

    def __init__(self, l, w, theta_0):
        positions = np.array([0., 0.,
                          l*np.cos(theta_0), -l*np.sin(theta_0)],
                         np.float64)
        colours = np.array([1., 0., 0.,
                            0., 1., 0.])

        rod = Rod2D(positions, colours)


        self.rodRenderable = Rod2DRenderable(rod, thickness = w)
        #viewer.addRenderable(rodRenderable)
        return