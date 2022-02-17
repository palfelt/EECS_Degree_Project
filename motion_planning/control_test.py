#!/usr/bin/env python

from math import sin, cos
from functools import partial
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import cv2

import sys
sys.path.append('/home/oscar_palfelt/MSc_thesis/ompl/py-bindings')
from ompl import base as ob
from ompl import control as oc

mapArr = cv2.imread('EECS_Degree_Project/motion_planning/maps/map0.png', 0)

## @cond IGNORE
# a decomposition is only needed for SyclopRRT and SyclopEST
class MyDecomposition(oc.GridDecomposition):
    def __init__(self, length, bounds):
        super(MyDecomposition, self).__init__(length, 2, bounds)
    def project(self, s, coord):
        coord[0] = s.getX()
        coord[1] = s.getY()
    def sampleFullState(self, sampler, coord, s):
        sampler.sampleUniform(s)
        s.setXY(coord[0], coord[1])
## @endcond

def isStateValid(spaceInformation, state):
    # perform collision checking or check if other constraints are
    # satisfied

    u = round(state.getX() * (mapArr.shape[1] - 1)) # right pointing image axis
    v = round((mapArr.shape[0] - 1) * (1 - state.getY())) # down pointing image axis

    if spaceInformation.satisfiesBounds(state):
        return mapArr[v,u] > 0
    else:
        return False
    # return spaceInformation.satisfiesBounds(state)

def propagate(start, control, duration, state):
    v = 0.1
    state.setX(start.getX() + v * duration * cos(start.getYaw()))
    state.setY(start.getY() + v * duration * sin(start.getYaw()))
    state.setYaw(start.getYaw() + control[0] * duration)

def plan():
    # construct the state space we are planning in
    space = ob.SE2StateSpace()

    # set the bounds for the R^2 part of SE(2)
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0.0)
    bounds.setHigh(1.0)
    space.setBounds(bounds)

    # create a control space
    cspace = oc.RealVectorControlSpace(space, 1)

    # set the bounds for the control space
    cbounds = ob.RealVectorBounds(1)
    cbounds.setLow(-1.3)
    cbounds.setHigh(1.3)
    cspace.setBounds(cbounds)

    # define a simple setup class
    ss = oc.SimpleSetup(cspace)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn( \
        partial(isStateValid, ss.getSpaceInformation())))
    ss.setStatePropagator(oc.StatePropagatorFn(propagate))

    # create a start state
    start = ob.State(space)
    start().setX(0.05)
    start().setY(0.05)
    start().setYaw(0)
 
    # create a goal state
    goal = ob.State(space)
    goal().setX(0.5)
    goal().setY(0.5)
    goal().setYaw(np.pi/2)

    # set the start and goal states
    ss.setStartAndGoalStates(start, goal, 0.05)

    # (optionally) set planner
    si = ss.getSpaceInformation()
    #planner = oc.RRT(si)
    #planner = oc.EST(si)
    #planner = oc.KPIECE1(si) # this is the default
    # SyclopEST and SyclopRRT require a decomposition to guide the search
    decomp = MyDecomposition(32, bounds)
    planner = oc.SyclopEST(si, decomp)
    #planner = oc.SyclopRRT(si, decomp)
    ss.setPlanner(planner)
    # (optionally) set propagation step size
    si.setPropagationStepSize(.1)

    # attempt to solve the problem
    solved = ss.solve(20.0)

    if solved:
        # print the path to screen
        pathArr = np.loadtxt(StringIO(ss.getSolutionPath().printAsMatrix()))

        fig, ax = plt.subplots()
        ax.imshow(mapArr, extent=[0, 255, 0, 255], cmap='gray')
        ax.plot(pathArr[:,0] * (mapArr.shape[1] - 1), pathArr[:,1] * (mapArr.shape[0] - 1), '.-')
        plt.show()

if __name__ == "__main__":
    plan()
