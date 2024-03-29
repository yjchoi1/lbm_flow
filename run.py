import argparse
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
from lbm_solver import LBMModel
import utils
from matplotlib import pyplot as plt


# Define run LBM
def run(lbm_timesteps, save_interval):
    # Get simulation information
    nnodes = LBM.lx * LBM.ly

    # Initiate vel and pressure array
    velocity = np.zeros((LBM.timesteps, nnodes, LBM.ndims), dtype=np.float32)
    pressure = np.zeros((LBM.timesteps, nnodes, 1), dtype=np.float32)
    LBM.init_field()

    # Start simulation
    for step in tqdm(range(lbm_timesteps)):
        LBM.boundary_condition()
        LBM.collision()
        LBM.streaming()

        # Save to npz
        if step % save_interval == 0:
            current_save_step = step // save_interval
            velocity, pressure = LBM.export_npz(current_save_step, velocity, pressure)

    return np.float32(velocity), np.float32(pressure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default="/work2/08264/baagee/frontera/gns-meshnet-data/lbm/dataset_trial4/config.json", type=str, help="Input json file name")
    args = parser.parse_args()

    # Get inputs from config json file
    input_path = args.input_path
    follow_taichi_coord = True
    f = open(input_path)
    inputs = json.load(f)
    f.close()

    # Simulation name and ids under which the output will be saved in order
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

    # Define timesteps
    lbm_timesteps = sim_config["lbm_timesteps"]
    save_interval = sim_config["save_interval"]  # downsampling rate for saving the lbm results as npz
    npz_timesteps = int(lbm_timesteps / save_interval)  # downsampled timesteps for saving npz

    # Start simulation corresponding to current sim id.
    for i in simulation_ids:
        current_sim_name = f"{simulation_name}{i}"

        # Velocity boundary condition
        if sim_config["initial_vel"]["autogen"]:
            print("Generate velocity boundary condition randomly")
            v_left_np = utils.vel_autogen(
                ly,
                shape_option=sim_config["initial_vel"]["option"],
                args=sim_config["initial_vel"]["args"])
        else:
            print("Get velocity boundary condition from data")
            v_left_np = utils.vel_from_data(
                current_sim_id=i, ly=ly,
                csv_file_path=sim_config["initial_vel"]["from_data"],
                n_simulations=len(simulation_ids))

        # Save current velocity boundary condition
        df = pd.DataFrame(v_left_np.reshape(1, -1))
        df.to_csv(f"{output_dir}/{current_sim_name}_velocity.csv", header=False, index=False)

        # Init LBM solver with the current domain setting
        LBM = LBMModel(
            lx=lx, ly=ly, lx_physical=lx_physical, nu=nu, v_left_np=v_left_np, timesteps=npz_timesteps)

        # Place obstacles
        if sim_config["circle"]["autogen"]:
            print("Generate obstacles randomly")
            circles = utils.gen_circles(
                n=np.random.randint(sim_config["circle"]['ncircles_range'][0], sim_config["circle"]['ncircles_range'][1]),
                x_range=sim_config["circle"]['x_range'],
                y_range=sim_config["circle"]['y_range'],
                radius_range=sim_config["circle"]['radius_range'],
                min_distance=sim_config["circle"]['min_distance'])
        else:
            print("Get obstacles from json data")
            f = open(sim_config["circle"]['from_data'])
            circle_data = json.load(f)['circle_data']
            circles = circle_data[i]

        # Save current circle configs
        with open(f"{output_dir}/{current_sim_name}_circle.json", "w") as circle_file:
            save_circle = {'circle_data': [circles]}
            json.dump(save_circle, circle_file, indent=2)

        # Run LBM
        # Place circular obstacles
        data = LBM.initialize_npz(circles)
        # Run lbm solver
        data['velocity'], data['pressure'] = run(lbm_timesteps=lbm_timesteps, save_interval=save_interval)
        # Reset solid nodes after LBM solver ends
        LBM.is_solid.fill(0)
        # Save result data
        if "npz_options" in inputs.keys() and inputs["npz_options"]["fixed_mesh"]:
            data['pos'] = np.float32(np.expand_dims(data['pos'][0], axis=0))
            data['node_type'] = np.expand_dims(data['node_type'][0], axis=0)
            data['cells'] = np.expand_dims(data['cells'][0], axis=0)
            del data['pressure']
            LBM.result_dict[current_sim_name] = data
        else:
            LBM.result_dict[current_sim_name] = data
        # Export to npz
        np.savez_compressed(f'{output_dir}/{current_sim_name}.npz', **LBM.result_dict)

        # Save config file being used
        with open(f"{output_dir}/config.json", "w") as input_file:
            json.dump(inputs, input_file, indent=4)

        # Visualization
        utils.plot_vleft(
            ly, data, v_left_np, save_path=f"{output_dir}/vleft_{current_sim_name}.png")

        if inputs["vis_config"]["save_field"]:
            x_range = [0, lx_physical]
            y_range = [0, ly_physical]

            for sim_data in LBM.result_dict.keys():
                for vis_step in inputs["vis_config"]["vis_steps"]:
                    utils.plot_field(
                        LBM.result_dict, sim_data, vis_step, lx, ly, x_range, y_range,
                        output_path=f"{output_dir}/{current_sim_name}_t{vis_step}.png")

        if i % inputs["vis_config"]["ani_save_interval"] == 0 and inputs["vis_config"]["save_animation"] == True:
            x_range = [0, lx_physical]
            y_range = [0, ly_physical]
            # Make animation
            # Call the function
            for sim_data in LBM.result_dict.keys():
                utils.make_animation(
                    LBM.result_dict, sim_data, npz_timesteps, lx, ly, x_range, y_range,
                    output_path=f"{output_dir}/{current_sim_name}.gif")

