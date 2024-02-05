import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import curve_fit
import pandas as pd


MARKER_SIZE = 0.1
SIZE_FACTOR = 10


def vel_autogen(ly, shape_option, args):
    def quadratic(x, a, b, c):
        return a * x ** 2 + b * x + c

    # if sim_config["initial_vel"]["autogen"]:
    print("Generate velocity boundary condition randomly")

    if shape_option == "uniform":
        if isinstance(args, dict):
            peak = np.random.uniform(args['peak'][0], args['peak'][1])
            npoints = args['npoints']
            v_left_np = np.full(npoints, peak)
        else:
            if args[2] != ly:
                raise ValueError("Size of input initial velocity array should be same as `ly`")
            else:
                v_left_np = np.random.uniform(args[0], args[1], args[2])

        v_left_np[0] = 0
        v_left_np[-1] = 0

    elif shape_option == "normal":
        raise NotImplementedError

    elif shape_option == "quad":
        peak = np.random.uniform(args['peak'][0], args['peak'][1])
        npoints = args['npoints']
        x_data = np.array([0, ly / 2, ly])
        y_data = np.array([0, peak, 0])

        coefficients, _ = curve_fit(quadratic, x_data, y_data)
        x_values = np.linspace(x_data.min(), x_data.max(), npoints)

        v_left_np = quadratic(x_values, *coefficients)

    elif shape_option == "multi_quad":
        peak = np.random.uniform(args['peak'][0], args['peak'][1])
        npoints = args['npoints']
        if npoints % 4 == 0:
            # set x for 1/2 of the entire left boundary
            x_data = np.array([0, ly / 4, ly / 2])
            y_data = np.array([0, peak, 0])
            coefficients, _ = curve_fit(quadratic, x_data, y_data)
            x_values = np.linspace(x_data.min(), x_data.max(), int(npoints / 2))
            partial_v = quadratic(x_values, *coefficients)
            # Sample first half of the `partial_v_left`
            partial_v_half = partial_v[0:int(npoints / 4)]
            v_left_np = np.concatenate((np.flip(partial_v_half), partial_v, partial_v_half))
        else:
            raise NotImplementedError("ly should be divisible by 4")

    elif shape_option == "from_csv":
        raise NotImplementedError

    else:
        raise ValueError("Not implemented velocity option. Choose among `normal`, `uniform, `quad`")

    v_left_np[0] = 0
    v_left_np[-1] = 0

    return v_left_np
def vel_from_data(current_sim_id, ly, csv_file_path, n_simulations):
    print("Get velocity boundary condition from data")
    df = pd.read_csv(csv_file_path, header=None)
    if df.values.shape[0] != n_simulations or df.values.shape[1] != ly:
        raise ValueError("Check the size of the `velocity.csv`")
    else:
        # get i-th velocity in csv file.
        v_left_np = df.values[current_sim_id, :].reshape(-1)

    v_left_np[0] = 0
    v_left_np[-1] = 0

    return v_left_np

def gen_circles(n, x_range, y_range, radius_range, min_distance=0):

    circles = []

    def is_overlapping(new_circle):
        x_new, y_new, r_new = new_circle
        for x, y, r in circles:
            if np.sqrt((x_new - x) ** 2 + (y_new - y) ** 2) < (r_new + r + min_distance):
                return True
        return False

    i = 0
    while len(circles) < n:
        radius = np.random.randint(radius_range[0], radius_range[1])
        x = np.random.randint(x_range[0], x_range[1])
        y = np.random.randint(y_range[0], y_range[1])
        new_circle = (x, y, radius)

        # prevent infinite loop
        i += 1
        j = 0
        if i > 1e6:
            print(f"Try new non-overlapping barrier config")
            i = 0
            j += 1
            circles = []

            if j >= 10:
                raise Exception(f"Cannot generate non overlapping obstacles in {j} trial")

        # Check overlap of the new circle and append if pass
        if not is_overlapping(new_circle):
            circles.append(new_circle)

    return circles


def visualize_circles_and_grid(circles, x_max, y_max, grid_spacing=1):
    # Create the grid points
    x_grid = np.arange(0, x_max, grid_spacing)
    y_grid = np.arange(0, y_max, grid_spacing)
    grid_points = np.array([(x, y) for x in x_grid for y in y_grid])

    # Function to check if a point is inside any circle
    def is_inside_any_circle(point):
        for x, y, r in circles:
            if np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2) <= r:
                return True
        return False

    # Color points based on whether they are inside a circle or not
    colors = ['red' if is_inside_any_circle(point) else 'gray' for point in grid_points]
    sizes = [3 if is_inside_any_circle(point) else 1 for point in grid_points]

    # Plotting
    plt.figure(figsize=(10, 10))
    plt.scatter(grid_points[:, 0], grid_points[:, 1], c=colors, s=sizes)
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)
    plt.title("Grid Points and Circles Visualization")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()


# def plot_config(spheres, nodes_lx, nodes_ly):
#
#     fig, ax = plt.subplots(1, 1, figsize=(nodes_lx/SIZE_FACTOR, nodes_ly/SIZE_FACTOR))
#     for i, (x, y, r) in enumerate(spheres):
#         circle = plt.Circle((x, y), r, facecolor='none')
#         ax.add_patch(circle)
#         ax.text(x, y + r, str(i), ha='center', va='bottom')
#     ax.xlim(0, nodes_lx)
#     ax.ylim(0, nodes_ly)
#     ax.set_aspect('equal', 'box')
#     ax.grid(visible=True)


def make_animation(
        dict_input, obs_info, total_timestep, nodes_lx, nodes_ly, x_range, y_range,
        output_path
):

    fig = plt.figure(figsize=(nodes_lx/SIZE_FACTOR, nodes_ly/SIZE_FACTOR))

    # Calculate global min and max velocity magnitudes across all timesteps
    all_velocities = np.concatenate([dict_input[obs_info]["velocity"][timestep] for timestep in range(total_timestep)])
    vel_mag_all = np.sqrt(np.sum(all_velocities ** 2, axis=-1))
    global_vel_mag_min = vel_mag_all.min()
    global_vel_mag_max = vel_mag_all.max()

    def animate(timestep):
        print(f"Render step {timestep}/{total_timestep}")
        # Increase figure size and DPI for better resolution
        fig.clear()
        ax = fig.add_subplot(1, 1, 1, projection='rectilinear')

        # Extract node coordinates and velocity at the given timestep
        coords = dict_input[obs_info]["pos"][0]
        velocities = dict_input[obs_info]["velocity"][timestep]

        # Get the node_type and positions for the specific timestep
        node_type_timestep = dict_input[obs_info]['node_type'][0, :, 0]
        pos_timestep = dict_input[obs_info]['pos'][0]

        # Create masks for each node type
        mask0 = node_type_timestep == 0
        mask4 = node_type_timestep == 4
        mask5 = node_type_timestep == 5
        mask6 = node_type_timestep == 6

        # Calculate the magnitude of the velocity
        vel_magnitude = np.sqrt(np.sum(velocities ** 2, axis=-1))

        # Reshape the coordinates and velocity magnitude to 2D grid
        X = coords[:, 0].reshape(nodes_ly, nodes_lx)
        Y = coords[:, 1].reshape(nodes_ly, nodes_lx)
        Z = vel_magnitude.reshape(nodes_ly, nodes_lx)

        # Create filled contour plot with increased number of contours
        contour_levels = 50  # increase this for smoother contours
        cntr = ax.contourf(
            X, Y, Z,
            contour_levels, cmap='cividis', vmin=global_vel_mag_min, vmax=global_vel_mag_max)

        # Scatter plot for each node type with different color
        # ax.scatter(pos_timestep[mask0, 0], pos_timestep[mask0, 1], color='blue', s=MARKER_SIZE, label='Normal nodes')
        ax.scatter(pos_timestep[mask4, 0], pos_timestep[mask4, 1], color='orange', s=MARKER_SIZE*3, label='Inlet nodes')
        ax.scatter(pos_timestep[mask5, 0], pos_timestep[mask5, 1], color='green', s=MARKER_SIZE*3, label='Outlet nodes')
        ax.scatter(pos_timestep[mask6, 0], pos_timestep[mask6, 1], color='red', s=MARKER_SIZE*3, label='Wall boundary')

        # Create a colorbar and reduce its height
        cbar = fig.colorbar(cntr, ax=ax, shrink=0.4)
        cbar.set_label('Velocity magnitude')
        # Limit the colorbar's legend to three digits
        cbar.formatter = FormatStrFormatter('%.3e')
        cbar.update_ticks()

        ax.set_title('Velocity field at timestep {} with obstacle {}'.format(timestep, obs_info))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        # Add the legend
        ax.legend(fontsize="7", loc="best")

        # Set x and y axis range
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)

        # Set aspect ratio to match the aspect ratio of x and y ranges
        ax.set_aspect("equal")
        # plt.show()

    ani = FuncAnimation(
        fig, animate, frames=np.arange(0, total_timestep, 10), interval=20)

    ani.save(output_path, dpi=100, fps=10, writer='imagemagick')
    # print(f"Animation saved to: {animation_filename}")

def plot_field(
        dict_input, obs_info, timestep, nodes_lx, nodes_ly, x_range, y_range,
        output_path):

    fig = plt.figure(figsize=(nodes_lx/SIZE_FACTOR, nodes_ly/SIZE_FACTOR))

    # Increase figure size and DPI for better resolution
    ax = fig.add_subplot(1, 1, 1, projection='rectilinear')

    # Extract node coordinates and velocity at the given timestep
    coords = dict_input[obs_info]["pos"][0]
    velocities = dict_input[obs_info]["velocity"][timestep]

    # Get the node_type and positions for the specific timestep
    node_type_timestep = dict_input[obs_info]['node_type'][0, :, 0]
    pos_timestep = dict_input[obs_info]['pos'][0]

    # Create masks for each node type
    mask0 = node_type_timestep == 0
    mask4 = node_type_timestep == 4
    mask5 = node_type_timestep == 5
    mask6 = node_type_timestep == 6

    # Calculate the magnitude of the velocity
    vel_magnitude = np.sqrt(np.sum(velocities ** 2, axis=-1))
    vel_mag_min = vel_magnitude.min()
    vel_mag_max = vel_magnitude.max()

    # Reshape the coordinates and velocity magnitude to 2D grid
    X = coords[:, 0].reshape(nodes_ly, nodes_lx)
    Y = coords[:, 1].reshape(nodes_ly, nodes_lx)
    Z = vel_magnitude.reshape(nodes_ly, nodes_lx)

    # Create filled contour plot with increased number of contours
    contour_levels = 50  # increase this for smoother contours
    cntr = ax.contourf(
        X, Y, Z,
        contour_levels, cmap='cividis', vmin=vel_mag_min, vmax=vel_mag_max)

    # Scatter plot for each node type with different color
    # ax.scatter(pos_timestep[mask0, 0], pos_timestep[mask0, 1], color='blue', s=MARKER_SIZE, label='Normal nodes')
    ax.scatter(pos_timestep[mask4, 0], pos_timestep[mask4, 1], color='orange', s=MARKER_SIZE*3, label='Inlet nodes')
    ax.scatter(pos_timestep[mask5, 0], pos_timestep[mask5, 1], color='green', s=MARKER_SIZE*3, label='Outlet nodes')
    ax.scatter(pos_timestep[mask6, 0], pos_timestep[mask6, 1], color='red', s=MARKER_SIZE*3, label='Wall boundary')

    # Create a colorbar and reduce its height
    cbar = fig.colorbar(cntr, ax=ax, shrink=0.4)
    cbar.set_label('Velocity magnitude')
    # Limit the colorbar's legend to three digits
    cbar.formatter = FormatStrFormatter('%.3e')
    cbar.update_ticks()

    ax.set_title('Velocity field at timestep {} with obstacle {}'.format(timestep, obs_info))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # Add the legend
    ax.legend(fontsize="7", loc="best")

    # Set x and y axis range
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    # Set aspect ratio to match the aspect ratio of x and y ranges
    ax.set_aspect("equal")
    # plt.show()

    plt.savefig(output_path)
    # print(f"Animation saved to: {animation_filename}")

def plot_vleft(ly, data, v_left_np, save_path):
    timesteps_to_plot = [0, 1, 50, -1]
    line_colors = ['gray', 'g', 'green', 'red']

    node_pos = data['pos'][0]
    timesteps = len(data['velocity'])
    x0_index = np.where(node_pos[:, 0] == 0)
    for i, t in enumerate(timesteps_to_plot):
        left_velocity = data["velocity"][t][x0_index, 0]
        plt.plot(np.arange(ly), left_velocity[0], line_colors[i], label=f"t{timesteps_to_plot}")
    plt.plot(np.arange(ly), v_left_np, 'blue', label='v_left', ls='--')
    plt.legend()
    plt.savefig(save_path)
