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
# import matplotlib.pyplot as plt
from geom import *
from .abstract_dynamic_system import AbstractDynamicSystem

def r_cardioid(phi:int)->int:
    return 1 - np.cos(phi)

# ## Dummy dynamic system just to test
class CardioideDynamicSystem(AbstractDynamicSystem):

    def __init__(self, mesh):
        ## Constructor
        # @param self
        # @param mesh  
        super().__init__()
        self.mesh = mesh

        # Animations parameters
        self.it = 60.
        self.delta = 1.
        self.period = 120.
        self.colours = np.copy(self.mesh.constColours)
        self.translationVector = np.tile([0.01, 0], self.mesh.nbVertices)

    def step(self):

        self.mesh.colours = (self.it / self.period) * self.colours
        self.mesh.positions += self.delta * self.translationVector

        self.it += self.delta
        if (self.it <= 0) or (self.it >= self.period):
            self.delta *= -1.

G = 9.81



def f(X, l):
    return np.array([X[1], -G/l * np.sin(X[0])])

## Dummy dynamic system just to test
class PendulumDynamicSystem(AbstractDynamicSystem):

    def __init__(self, l, w, theta_0, M):
        self.energies = []

        self.M = M

        positions = np.array([0., 0.,
                          l*np.cos(theta_0), -l*np.sin(theta_0)],
                         np.float64)
        colours = np.array([1., 0., 0.,
                            0., 1., 0.])

        self.rod = Rod2D(positions, colours)
        self.l = l


        self.rodRenderable = Rod2DRenderable(self.rod, thickness = w)
        #viewer.addRenderable(rodRenderable)

        self.X = np.array([theta_0, 0])
        self.Xdot = np.array([0., 0.])
        self.h = 1/60

        self.Mmoins1 = np.matrix([[1, self.h], [-self.h*G/l, 1]]) / (1 + (G*self.h**2/l))

        return
    
    def getSpeed(self):
        return self.l * self.X[1]
    
    def getHeight(self):
        return self.rod.positions[3] + self.l
    
    def getPotentialEnergy(self):
        return self.getHeight() * self.M * G
    
    def getKineticEnergy(self):
        return 0.5 * self.M * self.getSpeed()**2
    
    def getMecanicalEnergy(self):
        return self.getKineticEnergy() + self.getPotentialEnergy()
    
    def eulerExplicit(self):
        self.X += self.h * f(self.X, self.l)

    def eulerSemiImplicit(self):
        thetaDot_n = self.X[1]
        self.X += self.h * f(self.X, self.l)
        self.X[0] -= self.h * thetaDot_n
        self.X[0] += self.h * self.X[1]

    def eulerImplicit(self):
        self.X = np.array(self.Mmoins1.dot(self.X))[0]
    
    def step(self):
        self.eulerImplicit()

        # self.X[0] += 0.1
        self.rod.positions[3] = -self.l * np.cos(self.X[0])
        self.rod.positions[2] = -self.l * np.sin(self.X[0])
        #[self.l * np.cos(self.X[0]), -self.l * np.sin(self.X[0])]
        # self.rod.positions[1] += 0.1
        self.energies.append(self.getMecanicalEnergy())
        print(self.getMecanicalEnergy())
        #plt.plot(self.energies)
        
