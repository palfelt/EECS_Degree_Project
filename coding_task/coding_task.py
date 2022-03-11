import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree as KDTree
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import sys
sys.path.append('/home/oscar_palfelt/MSc_thesis/ompl/py-bindings')

envmap = np.genfromtxt('EECS_Degree_Project/coding_task/maps/scene_map.txt')

ix = np.arange(envmap.shape[0])
iy = np.arange(envmap.shape[1])
grid = np.meshgrid(ix, iy, indexing='ij')
envmap_coor = np.vstack([*map(np.ravel, grid)]).T

'''------ 1 --------'''
def valueFromSceneMap(x, y):
    x_lower, y_lower, step_xy = -360., -185., 0.1
    ix_coor = (x - x_lower) / step_xy
    iy_coor = (y - y_lower) / step_xy
    return envmap[ix_coor.astype(int)][iy_coor.astype(int)]

'''------ 2 --------'''
# verify valueFromSceneMap() function
x_range = np.arange(-360, 165, 0.1)
y_range = np.arange(-185, 40, 0.1)
test_envmap = np.zeros(shape=(x_range.size, y_range.size)) # shape mismatch negligible
for i, x in enumerate(x_range):
    for j, y in enumerate(y_range):
        try:
            test_envmap[i][j] = valueFromSceneMap(x,y)
        except:
            # index outside range, skip
            pass

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(envmap, cmap='cividis')
ax2.imshow(test_envmap, cmap='cividis')
plt.show()

'''------ 3 --------'''
def get_distance_map(mapval):
    value_indicies = np.argwhere(envmap==mapval)
    print("yo")
    print(value_indicies.shape)
    value_indicies_tree = KDTree(value_indicies)
    d, _ = value_indicies_tree.query(envmap_coor, distance_upper_bound=5)
    d = d.reshape(envmap.shape)
    d = np.where(d >= 5, 5, d)
    # return d¨
    np.save(f'EECS_Degree_Project/coding_task/maps/distance_map_{mapval}', d)
    np.save()

def main():
    values = list(range(7))
    with ProcessPoolExecutor(max_workers=8) as executor:
        executor.map(get_distance_map, values)

        # for v, d in enumerate(result):
        #     np.savetxt(f'coding_task\distance_map_{v}.txt', d)

if __name__ == '__main__':
    main()