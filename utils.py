import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter


def make_animation(
        dict_input, obs_info, total_timestep, nodes_lx, nodes_ly, x_range, y_range):

    fig = plt.figure(figsize=(40, 10))

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
        coords = dict_input[obs_info]["pos"][timestep]
        velocities = dict_input[obs_info]["velocity"][timestep]

        # Get the node_type and positions for the specific timestep
        node_type_timestep = dict_input[obs_info]['node_type'][timestep, :, 0]
        pos_timestep = dict_input[obs_info]['pos'][timestep]

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
        ax.scatter(pos_timestep[mask0, 0], pos_timestep[mask0, 1], color='blue', s=1, label='Normal nodes')
        ax.scatter(pos_timestep[mask4, 0], pos_timestep[mask4, 1], color='orange', s=1, label='Inlet nodes')
        ax.scatter(pos_timestep[mask5, 0], pos_timestep[mask5, 1], color='green', s=1, label='Outlet nodes')
        ax.scatter(pos_timestep[mask6, 0], pos_timestep[mask6, 1], color='red', s=1, label='Wall boundary')

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

    ani.save('ani.gif', dpi=100, fps=15, writer='imagemagick')
    # print(f"Animation saved to: {animation_filename}")

def plot_field(
        dict_input, obs_info, timestep, nodes_lx, nodes_ly, x_range, y_range):

    fig = plt.figure(figsize=(40, 10))

    # Increase figure size and DPI for better resolution
    ax = fig.add_subplot(1, 1, 1, projection='rectilinear')

    # Extract node coordinates and velocity at the given timestep
    coords = dict_input[obs_info]["pos"][timestep]
    velocities = dict_input[obs_info]["velocity"][timestep]

    # Get the node_type and positions for the specific timestep
    node_type_timestep = dict_input[obs_info]['node_type'][timestep, :, 0]
    pos_timestep = dict_input[obs_info]['pos'][timestep]

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
    ax.scatter(pos_timestep[mask0, 0], pos_timestep[mask0, 1], color='blue', s=1, label='Normal nodes')
    ax.scatter(pos_timestep[mask4, 0], pos_timestep[mask4, 1], color='orange', s=10, label='Inlet nodes')
    ax.scatter(pos_timestep[mask5, 0], pos_timestep[mask5, 1], color='green', s=10, label='Outlet nodes')
    ax.scatter(pos_timestep[mask6, 0], pos_timestep[mask6, 1], color='red', s=3, label='Wall boundary')

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

    plt.savefig(f'field_t{timestep}.png')
    # print(f"Animation saved to: {animation_filename}")

