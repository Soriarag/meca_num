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

H = -2

def r_cardioid(phi):
    #return 1
    return 1 - np.cos(phi)

G = 9.81
g1 = 0
g2 = -G
g = np.array([0, -G])
def wedge(a:np.array, b:np.array):
    return a[0]*b[1] - b[0]*a[1]
# ## Dummy dynamic system just to test
class DummyDynamicSystem(AbstractDynamicSystem):

    def __init__(self, mesh, num_vertices):
        ## Constructor
        # @param self
        # @param mesh  
        super().__init__()
        self.mesh = mesh
        self.num_vertices = num_vertices
        # Animations parameters
        self.it = 60.
        self.delta = 1.
        self.period = 0.00016
        self.colours = np.copy(self.mesh.constColours)
        self.translationVector = np.tile([0.01, 0], self.mesh.nbVertices)
        x_0 = 0.
        y_0 = 0.
        theta_0 = 0.
        self.q = np.array([[x_0, y_0, theta_0]])
        v_x = 0.
        v_y = 0.
        theta_dot = 0.
        self.q_dot = np.array([[v_x, v_y, theta_dot]])
        self.x_F = np.array([-1.0, 0.])
        self.X_f = np.array([1., 0])
        self.F = np.array([0., 0.])
        self.rho = 1
        self.indices_activated = []
        self.J = (15 * np.pi) / 8 * self.rho
        self.M = np.pi/2 * self.rho
        self.Mmat = np.array([[self.M, 0, 0], [0, self.M, 0], [0, 0, self.J]])
        invm = 1 / self.M
        self.invM = np.array([[invm, 0, 0], [0, invm, 0], [0, 0, 1 / self.J]])
        self.original_positions = self.mesh.positions.copy()

        self.n = 100
        id2 = np.identity(2)
        self.Jacobienne_first_half = np.concatenate([id2 for _ in range(self.n)])
        self.Xs = []
        delta_phi = 2*np.pi / self.n
        phi = 0
        for k in range(self.n):
            r = r_cardioid(phi)
            X = np.array([[r*np.cos(phi), r*np.sin(phi)]])
            self.Xs.append(X)
            phi += delta_phi

        #Draw line
        # positions = np.array([0., 0.,
        #                   l*np.cos(theta_0), -l*np.sin(theta_0)],
        #                  np.float64)
        positions = np.array([0, H-0.1,
                          10, H-0.1],
                         np.float64)
        colours = np.array([1., 0., 0.,
                            1., 0., 0.])
        self.rod = Rod2D(positions, colours)


        self.line = Rod2DRenderable(self.rod, thickness = 0.005)


    def get_jacobienne(self):
        c = np.cos(self.q[0, 2])
        s = np.sin(self.q[0, 2])
        R_prime = self.q_dot[0, 2] * np.array([[-s, -c], [c, -s]])
        jacobienne_second_half = np.concatenate([np.dot(R_prime, X.T) for X in self.Xs])
        jacobienne = np.concatenate((self.Jacobienne_first_half, jacobienne_second_half), axis=1)
        print("Jaco : ", jacobienne)
        return jacobienne
    
    def get_grad_g(self):
        grad = None
        self.indices_activated = []
        for i in range(self.n):
            if (self.need_lcp_i(i)):
                self.indices_activated.append(i)
                newline = np.array([[0 for _ in range(2*self.n)]])
                newline[0, 2*i+1] = 1
                if grad is None:
                    grad = newline
                else:
                    grad = np.concatenate((grad, newline))
            #grad[i, 2*i+1] = -1 #TODO pas sûr de ça mais comme l'axe y va vers la bas
        #grad = np.concatenate([np.array([[0, 1]]) for _ in range(self.n)], axis=1)
        #print("grad : ", grad)
        print("indices activated : ", self.indices_activated)
        return grad
    
    def descente_gradient_projete(self, A, b: np.array, alpha: float, x: np.array, tol: float):
        print("enter descente")
        if not (alpha > 0) or not (tol > 0):
            return None
        ctr = 0
        err = float('inf')
        while err > tol:
            ctr+=1
            y = (np.dot(A, x.T) + b).T

            x_P = np.maximum(0, x-alpha*y)
            err = np.linalg.norm(x_P -x)
            x= x_P
            #print("ERR : ",err)

        #print("NOMBNRE BOUCLE : ", ctr)
        print("exit descente")
        return x
    
    def get_x_g(self):
        return self.q[0, :2]
    
    def get_speed(self):
        return self.q_dot[0, :2]
    
    def get_R(self):
        c = np.cos(self.q[0, 2])
        s = np.sin(self.q[0, 2])
        R = np.matrix([[c, -s], [s, c]])
        return R
    
    def need_lcp_i(self, i : int):
        X_i = self.Xs[i]
        x_i = self.get_x_g() + np.dot(self.get_R(), X_i.T).T
        #print("x_i : ", x_i)
        active = x_i[0,1] <= H
        if (active):
            #print("ACTIVE")
            pass
        return active
    
    def get_F(self):
        return np.array([[self.M * g1, self.M * g2, wedge(self.x_F - self.get_x_g(), self.F)]])

    
    def get_f(self):
        return np.dot(self.Mmat, self.q_dot.T).T + self.get_F()
    
    def get_H(self):
        grad_g = self.get_grad_g()
        if grad_g is None:
            return None
        jaco = self.get_jacobienne()
        return np.dot(grad_g, jaco)

    def update(self):
        #print("enter update")
        H = self.get_H()
        constraint_part = np.zeros((1, 3))
        f = self.get_f()
        if H is not None:
            num_active = np.shape(H)[0]
            A = self.period * np.dot(H, np.dot(self.invM, H.T))
            b = np.dot(H, np.dot(self.invM, f.T))
            alpha = 1 / np.linalg.norm(A)
            # print("alpha : ", alpha)
            # print("B : ", b)
            # print("A : ", A)
            x_init = np.array([[0 for _ in range(num_active)]])
            mus = self.descente_gradient_projete(A, b, alpha, x_init, 0.1)
            constraint_part = self.period * np.dot(H.T, mus.T).T
            print("mus : ", self.period * mus)



        right_member = f + constraint_part

        self.q_dot = np.dot(self.invM, right_member.T).T

        print("q_dot : ", self.q_dot)
        self.q += self.period * self.q_dot

        


    
    

    # def stepef(self):
    #     c = np.cos(self.theta)
    #     s = np.sin(self.theta)
    #     R = np.matrix([[c, -s], [s, c]])
    #     I = np.identity(2)
    #     # F = np.array(np.dot(rot, self.F))[0]
    #     F = self.F
        

    #     self.speed += (g + F/self.M)*self.period
    #     self.x_g = np.array(np.dot(I-R, self.X_f))[0] + self.x_g_0
    #     self.theta_dot += self.period/self.J * wedge(self.x_F - self.x_g, F)
    #     self.theta += self.theta_dot * self.period

    #     for i in range(self.num_vertices):
            
    #         x = self.original_positions[2*i]
    #         y = self.original_positions[2*i+1]
            
    #         self.mesh.positions[2*i] = c*x + s*y + self.x_g[0]
    #         self.mesh.positions[2*i+1] = -s*x + c*y + self.x_g[1]


    def step(self):
        return
        #contact
        #print(" y : ", self.x_g[1])

        self.update()
        
        c = np.cos(self.q[0, 2])
        s = np.sin(self.q[0, 2])

        for i in range(self.num_vertices):
            
            x = self.original_positions[2*i]
            y = self.original_positions[2*i+1]

            
            self.mesh.positions[2*i] = c*x + s*y + self.get_x_g()[0]
            self.mesh.positions[2*i+1] = -s*x + c*y + self.get_x_g()[1]










    def stepg(self):
        #chute libre
        print(" y : ", self.x_g[1])
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        #rot = np.matrix([[c, -s], [s, c]])
        # F = np.array(np.dot(rot, self.F))[0]
        F = self.F
        self.update()
        #print("H : ", self.get_H())
        self.speed += (g + F/self.M)*self.period
        self.x_g += self.speed*self.period
        self.theta_dot += self.period/self.J * wedge(self.x_F - self.x_g, F)
        self.theta += self.theta_dot * self.period

        for i in range(self.num_vertices):
            
            x = self.original_positions[2*i]
            y = self.original_positions[2*i+1]
            
            self.mesh.positions[2*i] = c*x + s*y + self.x_g[0]
            self.mesh.positions[2*i+1] = -s*x + c*y + self.x_g[1]

    def stepg(self):
        #avec contraintes unilatérales
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        rot = np.matrix([[c, -s], [s, c]])
        # F = np.array(np.dot(rot, self.F))[0]
        F = self.F

        self.speed += (g + F/self.M)*self.period
        self.x_g += self.speed*self.period
        self.theta_dot += self.period/self.J * wedge(self.x_F - self.x_g, F)
        self.theta += self.theta_dot * self.period

        for i in range(self.num_vertices):
            
            x = self.original_positions[2*i]
            y = self.original_positions[2*i+1]
            
            self.mesh.positions[2*i] = c*x + s*y + self.x_g[0]
            self.mesh.positions[2*i+1] = -s*x + c*y + self.x_g[1]
        




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
        
