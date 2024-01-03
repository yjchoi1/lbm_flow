import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from lbm_solver import LBMModel
import utils


# LBM geometry
lx = 80
ly = 80
lx_physical = 1.6
dx = dy = lx_physical/lx
ly_physical = dy * ly

# yc: assume node spacing in lbm and gns is the same.
nodes_lx = lx  # yc: 80
nodes_ly = ly  # yc: 21

# Compute derived parameters
nnodes = nodes_lx * nodes_ly  # Total number of nodes
ndims = 2  # Number of dimensions
ncells = (nodes_ly - 1) * (nodes_lx - 1)  # Total number of cells
ncells_row = nodes_lx - 1  # Number of cells per row

# Inlet velocity & viscosity
nu = 0.0001
v_left_np = np.random.uniform(0.008, 0.015, ly)

# Define simulation parameters
max_step = 10000
save_step = 20
timesteps = int(max_step / save_step)
nnodes_per_cell = 4

# Init LBM solver with the current domain setting
LBM = LBMModel(
    lx=lx, ly=ly, lx_physical=lx_physical, nu=nu, v_left_np=v_left_np, timesteps=timesteps)

# Define run LBM
def run(max_step=max_step, save_step=save_step):
    velocity = np.zeros((timesteps, nnodes, ndims))
    pressure = np.zeros((timesteps, nnodes, 1))

    LBM.init_field()
    for step in tqdm(range(max_step)):
        LBM.boundary_condition()
        LBM.collision()
        LBM.streaming()

        # Save to npz
        # if step == 1:
        #     velocity, pressure = LBM.export_npz(current_save_step, velocity, pressure)
        if step % save_step == 0:
            current_save_step = step//save_step
            velocity, pressure = LBM.export_npz(current_save_step, velocity, pressure)
        # velocity, pressure = LBM.export_npz(step // save_step, velocity, pressure)

    return velocity, pressure


# # Location and size of obstacles
obs_x = [30, 32, 30, 31, 52, 50, 52, 51, 71, 70, 70, 68]  # obs_x = [70, 70, 70 ]
obs_y = [10, 30, 50, 71, 10, 30, 50, 69, 10, 30, 50, 71]  # obs_y = [40, 40, 40]
obs_r = [7, 7, 6, 7, 5, 7, 6, 7, 5, 6, 7, 7]  # obs_z = [12, 16, 20]
# obs_x = [50]  # obs_x = [70, 70, 70 ]
# obs_y = [5]  # obs_y = [40, 40, 40]
# obs_r = [10]

# Plotting the rectangular graph
plt.figure(figsize=(8, 4), dpi=200)
plt.xlim(0, lx)
plt.ylim(0, ly)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)

# Plotting the circles' edges
spheres = []
for i, (x, y, r) in enumerate(zip(obs_x, obs_y, obs_r)):
    circle = plt.Circle((x, y), r, facecolor='none', linewidth=2.5)
    plt.gca().add_patch(circle)
    plt.text(x, y + r, str(i), ha='center', va='bottom')
    spheres.append([x, y, r])

# Run LBM
obs_info = "spheres"
data = LBM.initialize_npz(spheres)
data['velocity'], data['pressure'] = run(max_step=10000, save_step=20)
# Reset solid nodes after LBM solver ends
LBM.is_solid.fill(0)

# Export data
LBM.parent_dict[obs_info] = data
# np.savez_compressed('GNS_Obstacle.npz', **LBM.parent_dict)

#%%
# Plot specific timestep
vis_steps = [0, 1, 2, 10, 50, 100, 150, 200, 300, 400, 499]
x_range = [0, lx_physical]
y_range = [0, ly_physical]

for obs in LBM.parent_dict.keys():
    for vis_step in vis_steps:
        utils.plot_field(LBM.parent_dict, obs, vis_step, nodes_lx, nodes_ly, x_range, y_range)

# Make animation
total_timestep = timesteps
# Call the function
for obs in LBM.parent_dict.keys():
    utils.make_animation(LBM.parent_dict, obs, total_timestep, nodes_lx, nodes_ly, x_range, y_range)
    # plot_field(parent_dict, obs, total_timestep, nodes_lx, nodes_ly, x_range, y_range)

a=1