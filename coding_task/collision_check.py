import numpy as np
import matplotlib.pyplot as plt

collisionCheckingRadius = 1.78
vehicleHalfWidth = 1.38
rearAxleFrontBumper = 8.765
rearBumperRearAxle = 3.405
distance_map = np.genfromtxt('EECS_Degree_Project\coding_task\maps\distance_map_1.txt')

plt.imshow(distance_map, cmap='cividis')
plt.show()

class VehicleState:
    def __init__(self):
        self.x = -60
        self.y = -85
        self.yaw = 0
    
    def set_state(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw


def valueFromSatDistMapStatic(x, y):
    x_lower, y_lower, step_xy = -360., -185., 0.1
    ix_coor = (x - x_lower) / step_xy
    iy_coor = (y - y_lower) / step_xy
    return distance_map[ix_coor.astype(int)][iy_coor.astype(int)]


'''------ 4 --------'''
def checkForCollision(vehicleState, collisionCheckingRadius, vehicleHalfWidth, rearAxleFrontBumper, rearBumperRearAxle):
    EPSILON = 0.1

    # Initialize some parameters
    upperLimit = rearAxleFrontBumper
    r = - rearBumperRearAxle
    upperLimitReached = False
    vxnorm = np.cos(vehicleState.yaw)
    vynorm = np.sin(vehicleState.yaw)	
    distanceToNearestObject = np.inf

    while not upperLimitReached:
        if r >= upperLimit - EPSILON:
            r = upperLimit
            upperLimitReached = True

        x = vehicleState.x + vxnorm * r
        y = vehicleState.y + vynorm * r

        d = valueFromSatDistMapStatic(x,y) - collisionCheckingRadius

        #if d > dist:
        if d < distanceToNearestObject:
            if d < EPSILON:
                return d

            distanceToNearestObject = d

        r += max(vehicleHalfWidth, d)

    return distanceToNearestObject


def main():
    vehicleState = VehicleState()

    d = checkForCollision(vehicleState, collisionCheckingRadius, vehicleHalfWidth, rearAxleFrontBumper, rearBumperRearAxle)
    print(d)


if __name__ == '__main__':
        main()