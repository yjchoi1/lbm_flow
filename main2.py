import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from lbm_solver import LBMModel
import utils


# LBM geometry
lx = 320
ly = 84

# yc: assume node spacing in lbm and gns is the same.
nodes_lx = lx  # yc: 80
nodes_ly = ly  # yc: 21

# Compute derived parameters
nnodes = nodes_lx * nodes_ly  # Total number of nodes
ndims = 2  # Number of dimensions
ncells = (nodes_ly - 1) * (nodes_lx - 1)  # Total number of cells
ncells_row = nodes_lx - 1  # Number of cells per row

# Inlet velocity & viscosity
nu = 0.08
v_left_np = np.random.uniform(0.1, 0.3, ly)

# Define simulation parameters
max_step = 10000
save_step = 20
timesteps = int(max_step / save_step)
nnodes_per_cell = 4

LBM = LBMModel(lx=lx, ly=ly, nu=nu, v_left_np=v_left_np, timesteps=timesteps)

def run(max_step=max_step, save_step=save_step):
    velocity = np.zeros((timesteps, nnodes, ndims))
    pressure = np.zeros((timesteps, nnodes, 1))

    LBM.init_field()
    for step in tqdm(range(max_step)):
        LBM.collision()
        LBM.streaming()
        LBM.boundary_condition()

        # Save to npz
        if step % save_step == 0:
            current_save_step = step//save_step
            velocity, pressure = LBM.export_npz(current_save_step, velocity, pressure)
        # velocity, pressure = LBM.export_npz(step // save_step, velocity, pressure)

    return velocity, pressure


# Location and size of obstacles
# obs_x = [30, 32, 30, 31, 52, 50, 52, 51, 71, 70, 70, 68]  # obs_x = [70, 70, 70 ]
# obs_y = [10, 30, 50, 71, 10, 30, 50, 69, 10, 30, 50, 71]  # obs_y = [40, 40, 40]
# obs_r = [7, 7, 6, 7, 5, 7, 6, 7, 5, 6, 7, 7]  # obs_z = [12, 16, 20]
obs_x = [100]  # obs_x = [70, 70, 70 ]
obs_y = [40]  # obs_y = [40, 40, 40]
obs_r = [15]
# Location and size of obstacles
colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'pink']

# Plotting the rectangular graph
plt.figure(figsize=(8, 4), dpi=200)
plt.xlim(0, lx)
plt.ylim(0, ly)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)

# Plotting the circles' edges
spheres = []
for i, (x, y, r) in enumerate(zip(obs_x, obs_y, obs_r)):
    circle = plt.Circle((x, y), r, facecolor='none', edgecolor=colors[i % len(colors)], linewidth=2.5)
    plt.gca().add_patch(circle)
    plt.text(x, y + r, str(i), ha='center', va='bottom')
    spheres.append([x, y, r])


obs_info = "spheres"
data = LBM.initialize_npz(spheres)
data['velocity'], data['pressure'] = run(max_step=10000, save_step=20)
LBM.is_solid.fill(0)

LBM.parent_dict[obs_info] = data

np.savez('GNS_Obstacle.npz', **LBM.parent_dict)


timestep = 2
total_timestep = 500
# Assume x_range and y_range are given
x_range = [-0.01, 1.6]
y_range = [-0.02, 0.4]

# Call the function
for obs in LBM.parent_dict.keys():
    utils.make_animation(LBM.parent_dict, obs, total_timestep, nodes_lx, nodes_ly, x_range, y_range)
    # plot_field(parent_dict, obs, total_timestep, nodes_lx, nodes_ly, x_range, y_range)

a=1