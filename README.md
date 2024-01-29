# Fluid flow simulation with Lattice Boltzmann Method (LBM)
This code is for simulating fluid flow simulation through porous media
using Lattice Boltzmann Method (LBM). The LBM solver is based on [link](https://github.com/Amber1995/Taichi-MRT-MultiphaseLBM)

## Simulation example
```shell
python3 input_path="examples/random_gen/config-random_gen.json"
```
![Simulation example](examples/random_gen/outputs/porous0.gif)
```shell
python3 input_path="examples/from_data/config-from_data.json"
```
![image alt >](examples/from_data/outputs/porous1.gif)
Dots denotes lattice nodes.

## Run
```shell
python3 input_path="config.json"
```
An example of the config files can be found in `examples/`. The options of `config.json` is explained below.

## Velocity input options

### Uniform
* In `config["sim_config"]["initial_vel"]`, set `autogen` `true`.
* set `option` to `uniform`
* Set `args` based on `[min, max, nnodes_left]`. This follows numpy argument for 
[`numpy.random.uniform`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html#numpy-random-uniform).

### Normal
`args = [...]` (Not yet implemented)

### Quad or multi-quad 
* set `option` to `quad` or `multi_quad` depending on the purpose.  
* `args = [peak, array_length]` 




### From `csv`
* In `config["sim_config"]["initial_vel"]`, set `autogen` `false`.
* Specify user-defined velocity file path to `from_data` (e.g., `examples/from_data/velocity.csv`) 

Velocity file looks like as follows: 
`velocities.csv`
```
     v0, v1, v2, ...
sim0
sim1
sim2
.
.
.
```

The number of `sim{n}` should match `simulation_id_range`.

## Obstacle options
### Random
* In `config["sim_config"]["circle"]`, set `autogen` `true`.
* Set `ncircles_range` (e.g, `[8, 12]` will generate 10 to 12 circular obstacles randomly).
* Set `radius_range` (e.g, `[7, 10]` will generate circular obstacles with radius 7 to 10 randomly).
* Set `x_range` and `y_range`. 
This is a range `[lower_bound, upper_bound]` to place the center of circles.

### From `json`
* In `config["sim_config"]["circle"]`, set `autogen` `false`.
* Specify user-defined obstacle file path to `from_data` (e.g., `examples/from_data/circles.json`)

Obstacle file looks like as follows: 
```json
{
  "circle_data": [
    [[40, 40, 7], [55, 42, 5]],
    [[40, 40, 6]]
  ]
}
```
Each low corresponds to the circular obstacle information for the associated simulation id, 
where each entry corresponds to a circle's `x`, `y` and `r`. 
`len(circle_data)` should match `simulation_id_range`.

# Contributors
* Yongjin Choi
* Qiuyu Wang