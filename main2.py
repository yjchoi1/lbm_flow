import argparse
import numpy as np
import json
import os
from tqdm import tqdm
from lbm_solver import LBMModel
import utils


# Define run LBM
def run(lbm_timesteps, save_interval):

    # Get simulation information
    nnodes = LBM.lx * LBM.ly

    # Initiate vel and pressure array
    velocity = np.zeros((LBM.timesteps, nnodes, LBM.ndims))
    pressure = np.zeros((LBM.timesteps, nnodes, 1))
    LBM.init_field()

    # Start simulation
    for step in tqdm(range(lbm_timesteps)):
        LBM.boundary_condition()
        LBM.collision()
        LBM.streaming()

        # Save to npz
        if step % save_interval == 0:
            current_save_step = step//save_interval
            velocity, pressure = LBM.export_npz(current_save_step, velocity, pressure)

    return velocity, pressure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default="config.json", type=str, help="Input json file name")
    args = parser.parse_args()

    input_path = args.input_path
    follow_taichi_coord = True
    f = open(input_path)
    inputs = json.load(f)
    f.close()

    simulation_name = inputs["simulation_name"]
    simulation_ids = np.arange(inputs["simulation_id_range"][0], inputs["simulation_id_range"][1])

    # Set output directory
    output_dir = inputs["output_dir"]
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    # Set simulation config
    sim_config = inputs["sim_config"]
    # LBM geometry
    lx = sim_config["lx"]
    ly = sim_config["ly"]
    lx_physical = sim_config["lx_physical"]
    dx = dy = lx_physical / lx
    ly_physical = dy * ly
    # Inlet velocity & viscosity
    # TODO: make a table for proper range of nu & vel & n_circles
    nu = sim_config["nu"]
    # TODO: velocity distribution options
    if sim_config["initial_vel"]["option"] == "uniform":
        args = sim_config["initial_vel"]["args"]
        v_left_np = np.random.uniform(args[0], args[1], args[2])
    elif sim_config["initial_vel"]["option"] == "normal":
        pass
    elif sim_config["initial_vel"]["option"] == "quad":
        pass
    else:
        raise ValueError("Not implemented velocity option! Choose among `normal`, `uniform, `quad`")

    # Define simulation parameters
    lbm_timesteps = sim_config["lbm_timesteps"]
    save_interval = sim_config["save_interval"]
    npz_timesteps = int(lbm_timesteps / save_interval)

    for i in simulation_ids:
        simulation_name = f"{simulation_name}{i}"
        # Init LBM solver with the current domain setting
        LBM = LBMModel(
            lx=lx, ly=ly, lx_physical=lx_physical, nu=nu, v_left_np=v_left_np, timesteps=npz_timesteps)

        circles = utils.gen_circles(10, x_max=lx, y_max=ly, radius_range=[7, 10])
        utils.visualize_circles_and_grid(circles, lx, ly)

        # Run LBM
        # Place circular obstacles
        data = LBM.initialize_npz(circles)
        # Run lbm solver
        data['velocity'], data['pressure'] = run(lbm_timesteps=lbm_timesteps, save_interval=save_interval)
        # Reset solid nodes after LBM solver ends
        LBM.is_solid.fill(0)
        # Save result data
        LBM.result_dict[simulation_name] = data
        # Export to npz
        np.savez(f'{output_dir}/{simulation_name}.npz', **LBM.result_dict)

        # Visualization
        if i % inputs["vis_config"]["field_save_interval"] == 0:
            if inputs["vis_config"]["save_field"] == True:
                vis_steps = [0, 1, 2, 10, 50, 100, 150, 200, 300, 400, 499]
                x_range = [0, lx_physical]
                y_range = [0, ly_physical]

                for sim_data in LBM.result_dict.keys():
                    for vis_step in vis_steps:
                        utils.plot_field(
                            LBM.result_dict, sim_data, vis_step, lx, ly, x_range, y_range,
                            output_path=f"{output_dir}/{simulation_name}_t{vis_step}.png")

        if i % inputs["vis_config"]["ani_save_interval"] == 0:
            if inputs["vis_config"]["save_animation"] == True:
                # Make animation
                # Call the function
                for sim_data in LBM.result_dict.keys():
                    utils.make_animation(
                        LBM.result_dict, sim_data, npz_timesteps, lx, ly, x_range, y_range,
                        output_path=f"{output_dir}/{simulation_name}.gif")
