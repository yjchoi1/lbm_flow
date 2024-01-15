import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter

MARKER_SIZE = 0.1
SIZE_FACTOR = 10


def gen_circles(n, x_range, y_range, radius_range):

    circles = []

    def is_overlapping(new_circle):
        x_new, y_new, r_new = new_circle
        for x, y, r in circles:
            if np.sqrt((x_new - x) ** 2 + (y_new - y) ** 2) < (r_new + r):
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
        if i > 1e6:
            break

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
        cbar.formatter = FormatStrFormatter('%.3f')
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
        fig, animate, frames=np.arange(0, total_timestep, 5), interval=20)

    ani.save(output_path, dpi=100, fps=15, writer='imagemagick')
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
    cbar.formatter = FormatStrFormatter('%.3f')
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

