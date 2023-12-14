import taichi as ti
import numpy as np
import math
from sympy import inverse_mellin_transform
from pyevtk.hl import gridToVTK
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import random
from tqdm import tqdm
from utils import plot_field, make_animation

# %%

ti.init(arch=ti.cpu)

dim = 2
Q = 9
real = ti.f32
i32_vec2d = ti.types.vector(2, ti.i32)
f32_vec2d = ti.types.vector(2, ti.f32)
scalar_real = lambda: ti.field(dtype=real)
scalar_int = lambda: ti.field(dtype=ti.i32)
vec = lambda: ti.Vector.field(dim, dtype=real)
vec_Q = lambda: ti.Vector.field(Q, dtype=real)

# Input parameters
lx = 80  # yc: 320
ly = 80  # yc: 84

is_solid = scalar_int()
v = vec()
temp_v = vec()
target_v = vec()
target_rho = scalar_real()
rho = scalar_real()
collide_f = vec_Q()
stream_f = vec_Q()

ti.root.dense(ti.ij, (lx, ly)).place(is_solid, target_v, target_rho, rho, v, temp_v, collide_f, stream_f)

# Definition of LBM parameters
half = (Q - 1) // 2

# LBM weights
w_np = np.array(
    [4.0 / 9.0, 1.0 / 36.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 9.0])
w = ti.field(ti.f32, shape=Q)
w.from_numpy(w_np)

# x and y components of predefined velocity in Q directions
e_xy_list = [[0, 0], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1]]
e_xy = ti.Vector.field(n=2, dtype=ti.i32, shape=Q)
e_xy.from_numpy(np.array(e_xy_list))

# reversed_e_xy_np stores the index of the opposite component to every component in e_xy_np
reversed_e_index = np.array([e_xy_list.index([-a for a in e]) for e in e_xy_list])

# MRT matrices
M_np = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 0, -1, 0, 1, -1, -1, 1], [0, 0, 1, 0, -1, 1, 1, -1, -1],
                 [0, 1, 1, 1, 1, 2, 2, 2, 2], [0, 1, -1, 1, -1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, -1, 1, -1],
                 [0, 0, 0, 0, 0, 1, 1, -1, -1], [0, 0, 0, 0, 0, 1, -1, -1, 1], [0, 0, 0, 0, 0, 1, 1, 1, 1]])
M_mat = ti.Matrix.field(Q, Q, ti.f32, shape=())
M_mat[None] = ti.Matrix(M_np)

inv_M_np = np.linalg.inv(M_np)
inv_M_mat = ti.Matrix.field(Q, Q, ti.f32, shape=())
inv_M_mat[None] = ti.Matrix(inv_M_np)

# Diagonal relaxation matrix for fluid 1
S_dig_vec = ti.Vector.field(Q, ti.f32, shape=())
S_dig_vec[None] = ti.Vector([1, 1.5, 1.4, 1, 1.5, 1, 1.5, 1., 1.])

# Fluid properties
nu = 0.08
v_left = 0.1

ti.static(e_xy)
ti.static(w)
ti.static(M_mat)
ti.static(inv_M_mat)
ti.static(S_dig_vec)

# %%

parent_dict = {}

# Define simulation parameters
timesteps = 600
nnodes_per_cell = 4

# yc: assume node spacing in lbm and gns is the same.
nodes_lx = lx  # yc: 80
nodes_ly = ly  # yc: 21

# Compute derived parameters
nnodes = nodes_lx * nodes_ly  # Total number of nodes
ndims = 2  # Number of dimensions
ncells = (nodes_ly - 1) * (nodes_lx - 1)  # Total number of cells
ncells_row = nodes_lx - 1  # Number of cells per row

# Define physical parameters
# yc: used to be dx = dy = 0.8 / nodes_lx
lx_physical = 0.8
# Based on lx_physical, compute physical spacing between nodes (assuming equal spacing in x and y)
dx = dy = lx_physical / nodes_lx

# Define Lattice-Boltzmann model parameters
lbm_dx = lx // nodes_lx  # Lattice spacing in x-direction
lbm_dy = ly // nodes_ly  # Lattice spacing in y-direction

j_indices = np.arange(nnodes)
lbm_x = (j_indices % nodes_lx) * lbm_dy
lbm_y = (j_indices // nodes_lx) * lbm_dy

# %%
# Writing input model (here we make a grain sample
solid_count = 0


def place_sphere(spheres):
    global solid_count

    for sphere in spheres:
        x = sphere[0]
        y = sphere[1]
        R = sphere[2]
        xmin = max(0, x - R)
        ymin = max(0, y - R)
        xmax = min(lx, x + R + 1)
        ymax = min(ly, y + R + 1)

        for px in range(xmin, xmax):
            for py in range(ymin, ymax):
                dx = px - x
                dy = py - y

                dist2 = dx * dx + dy * dy
                R2 = R * R

                if dist2 < R2 and is_solid[px, py] == 0:
                    is_solid[px, py] = 1
                    solid_count += 1


@ti.func
def periodic_index(i):
    iout = i
    if i[0] < 0:     iout[0] = lx - 1
    if i[0] > lx - 1:  iout[0] = 0
    if i[1] < 0:     iout[1] = ly - 1
    if i[1] > ly - 1:  iout[1] = 0

    return iout


@ti.func
def velocity_vec(local_pos) -> f32_vec2d:
    velocity_vec = ti.Vector([0., 0.])
    for i in ti.static(range(2)):
        for s in ti.static(range(Q)):
            velocity_vec[i] = velocity_vec[i] + (stream_f[local_pos][s] * e_xy[s][i])
        velocity_vec[i] = velocity_vec[i] / rho[local_pos]

    return velocity_vec


@ti.func
def feq(k, rho_local, u):  # anti bounce-back pressue bound
    eu = e_xy[k].dot(u)
    uv = u.dot(u)
    feqout = w[k] * rho_local * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv)
    return feqout


@ti.kernel
def init_field():
    for x, y in ti.ndrange(lx, ly):
        rho[x, y] = 1.0
        v[x, y] = ti.Vector([0., 0.])
        collide_f[x, y] = ti.Vector([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        stream_f[x, y] = ti.Vector([0., 0., 0., 0., 0., 0., 0., 0., 0.])

        if is_solid[x, y] <= 0:
            for q in ti.static(range(Q)):
                collide_f[x, y][q] = w[q] * rho[x, y]
                stream_f[x, y][q] = w[q] * rho[x, y]


@ti.kernel
def collision():
    for I in ti.grouped(collide_f):
        if (I.x < lx and I.y < ly and is_solid[I] <= 0):
            """MRT operator"""
            a = 1. / 36.
            v[I] = velocity_vec(I)

            e = -4 * collide_f[I][0] + 2 * collide_f[I][1] - collide_f[I][2] + 2 * collide_f[I][3] - collide_f[I][
                4] + 2 * collide_f[I][5] - collide_f[I][6] + 2 * collide_f[I][7] - collide_f[I][8]
            eps = 4 * collide_f[I][0] + collide_f[I][1] - 2 * collide_f[I][2] + collide_f[I][3] - 2 * collide_f[I][4] + \
                  collide_f[I][5] - 2 * collide_f[I][6] + collide_f[I][7] - 2 * collide_f[I][8]

            j_x = collide_f[I][5] + collide_f[I][6] + collide_f[I][7] - collide_f[I][1] - collide_f[I][2] - \
                  collide_f[I][3]
            q_x = -collide_f[I][1] + 2 * collide_f[I][2] - collide_f[I][3] + collide_f[I][5] - 2 * collide_f[I][6] + \
                  collide_f[I][7]
            j_y = collide_f[I][1] + collide_f[I][8] + collide_f[I][7] - collide_f[I][3] - collide_f[I][4] - \
                  collide_f[I][5]
            q_y = collide_f[I][1] - collide_f[I][3] + 2 * collide_f[I][4] - collide_f[I][5] + collide_f[I][7] - 2 * \
                  collide_f[I][8]
            p_xx = collide_f[I][2] - collide_f[I][4] + collide_f[I][6] - collide_f[I][8]
            p_xy = -collide_f[I][1] + collide_f[I][3] - collide_f[I][5] + collide_f[I][7]

            j_x2 = j_x * j_x
            j_y2 = j_y * j_y

            eO = e - S_dig_vec[None][1] * (e + 2 * rho[I] - 3 * (j_x2 + j_y2) / rho[I])
            epsO = eps - S_dig_vec[None][2] * (eps - rho[I] + 3 * (j_x2 + j_y2) / rho[I])
            q_xO = q_x - S_dig_vec[None][3] * (q_x + j_x)
            q_yO = q_y - S_dig_vec[None][6] * (q_y + j_y)
            p_xxO = p_xx - 1.0 / (3.0 * nu + 0.5) * (p_xx - (j_x2 - j_y2) / rho[I])
            p_xyO = p_xy - 1.0 / (3.0 * nu + 0.5) * (p_xy - j_x * j_y / rho[I])

            collide_f[I][0] = a * (4 * rho[I] - 4 * eO + 4 * epsO)
            collide_f[I][2] = a * (4 * rho[I] - eO - 2 * epsO - 6 * j_x + 6 * q_xO + 9 * p_xxO)
            collide_f[I][4] = a * (4 * rho[I] - eO - 2 * epsO - 6 * j_y + 6 * q_yO - 9 * p_xxO)
            collide_f[I][6] = a * (4 * rho[I] - eO - 2 * epsO + 6 * j_x - 6 * q_xO + 9 * p_xxO)
            collide_f[I][8] = a * (4 * rho[I] - eO - 2 * epsO + 6 * j_y - 6 * q_yO - 9 * p_xxO)
            collide_f[I][1] = a * (4 * rho[I] + 2 * eO + epsO - 6 * j_x - 3 * q_xO + 6 * j_y + 3 * q_yO - 9 * p_xyO)
            collide_f[I][3] = a * (4 * rho[I] + 2 * eO + epsO - 6 * j_x - 3 * q_xO - 6 * j_y - 3 * q_yO + 9 * p_xyO)
            collide_f[I][5] = a * (4 * rho[I] + 2 * eO + epsO + 6 * j_x + 3 * q_xO - 6 * j_y - 3 * q_yO - 9 * p_xyO)
            collide_f[I][7] = a * (4 * rho[I] + 2 * eO + epsO + 6 * j_x + 3 * q_xO + 6 * j_y + 3 * q_yO + 9 * p_xyO)


@ti.kernel
def boundary_condition():
    for I in ti.grouped(v):
        if (I.x < lx and I.y < ly and is_solid[I] <= 0):
            for s in ti.static(range(Q)):
                if I.x == 0:
                    if e_xy[s][0] == 1 and e_xy[s][1] == 0:
                        stream_f[I][s] = feq(s, rho[I], ti.Vector([v_left, v[I].y]))

    for I in ti.grouped(v):
        if (I.x < lx and I.y < ly and is_solid[I] <= 0):
            collide_f[I] = stream_f[I]
            rho[I] = collide_f[I].sum()


@ti.kernel
def streaming():
    for I in ti.grouped(collide_f):
        if (I.x < lx and I.y < ly and is_solid[I] <= 0):
            for s in ti.static(range(Q)):
                neighbor_pos = periodic_index(I + e_xy[s])
                if (is_solid[neighbor_pos] <= 0):
                    stream_f[neighbor_pos][s] = collide_f[I][s]
                else:
                    stream_f[I][reversed_e_index[s]] = collide_f[I][s]


def export_VTK(n):
    is_solid_3d = np.ascontiguousarray(is_solid.to_numpy()[0:lx, 0:ly]).reshape(lx, ly, 1)
    rho_3d = np.ascontiguousarray(rho.to_numpy()[0:lx, 0:ly]).reshape(lx, ly, 1)
    v_ = v.to_numpy()[0:lx, 0:ly, :]
    v_3d = v_[0:lx, 0:ly, np.newaxis, :]

    grid_x = np.linspace(0, lx, lx)
    grid_y = np.linspace(0, ly, ly)
    z = np.array([0.0])

    gridToVTK(
        "./LB_SingelPhase_" + str(n),
        grid_x,
        grid_y,
        z,
        pointData={"Solid": is_solid_3d,
                   "rho": rho_3d,
                   "velocity": (np.ascontiguousarray(v_3d[:, :, :, 0]),
                                np.ascontiguousarray(v_3d[:, :, :, 1]),
                                np.ascontiguousarray((v_3d[:, :, :, 0] ** 2 + v_3d[:, :, :, 1] ** 2) ** 0.5),
                                )
                   }
    )


@ti.kernel
def update_vel():
    for I in ti.grouped(v):
        v[I] = velocity_vec(I)


def export_npz(timestep, velocity, pressure):
    v_ = v.to_numpy()[0:lx, 0:ly, :]
    rho_ = rho.to_numpy()[0:lx, 0:ly]
    velocity[timestep, :, 0] = v_[lbm_x, lbm_y, 0]
    velocity[timestep, :, 1] = v_[lbm_x, lbm_y, 1]
    pressure[timestep, :, 0] = rho_[lbm_x, lbm_y]
    return velocity, pressure


def initialize_npz(spheres):
    for i in range(lx):
        is_solid[i, 0] = 1  # 1 is solid, 0 is fluid
        for j in range(ly - 1, ly):
            is_solid[i, j] = 1

    # tiff images to write is_solid in future

    place_sphere(spheres)
    # Node positions
    pos = np.zeros((timesteps, nnodes, ndims))
    pos[:, :, 0] = (j_indices % nodes_lx) * dx
    pos[:, :, 1] = (j_indices // nodes_lx) * dy

    # Node types
    node_type = np.zeros((timesteps, nnodes, 1), dtype=int)

    # Create lbm_x and lbm_y arrays with shape (timesteps, nnodes, 1)
    lbm_x_arr = np.broadcast_to(lbm_x[None, :, None], (timesteps, nnodes, 1))
    lbm_y_arr = np.broadcast_to(lbm_y[None, :, None], (timesteps, nnodes, 1))

    # Assuming 'is_solid' is a 2D array with shape (lx, ly)
    is_solid_ = is_solid.to_numpy()[0:lx, 0:ly]

    # Update node_type based on conditions
    node_type[is_solid_[lbm_x_arr, lbm_y_arr] == 1] = 6
    node_type[lbm_x_arr == 0] = 4
    node_type[lbm_x_arr == lx - lbm_dy] = 5

    # Cells
    j_indices_cells = np.arange(ncells)
    cell = np.zeros((timesteps, ncells, nnodes_per_cell), dtype=int)
    cell[:, :, 0] = j_indices_cells + j_indices_cells // ncells_row
    cell[:, :, 1] = cell[:, :, 0] + 1
    cell[:, :, 2] = cell[:, :, 0] + ncells_row + 2
    cell[:, :, 3] = cell[:, :, 0] + ncells_row + 1

    # create the data
    data = {
        "pos": pos,  # node coordinates
        "node_type": node_type,  # type of nodes encoded with integer
        "velocity": 0,  # fluid velocity
        "cells": cell,  # cells making the mesh
        "pressure": 0  # Optional dynamics properties
    }
    return data


def run(max_step=1000, compute_loss=True):
    velocity = np.zeros((timesteps, nnodes, ndims))
    pressure = np.zeros((timesteps, nnodes, 1))

    init_field()
    for step in tqdm(range(max_step)):
        collision()
        streaming()
        boundary_condition()
        velocity, pressure = export_npz(step // 20, velocity, pressure)

        # if step % 200 == 0:
        #     export_VTK(step // 200)

    solid_count = 0
    return velocity, pressure


# %%
# Location and size of obstacles
# Change the location and size of the obstacle(s)
##Note: lx = 320 ly = 84
obs_x = [30, 32, 30, 31, 52, 50, 52, 51, 71, 70, 70, 68]  # obs_x = [70, 70, 70 ]
obs_y = [10, 30, 50, 71, 10, 30, 50, 69, 10, 30, 50, 71]  # obs_y = [40, 40, 40]
obs_r = [7, 7, 6, 7, 5, 7, 6, 7, 5, 6, 7, 7]  # obs_z = [12, 16, 20]

# %%
# Location and size of obstacles
colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'pink']

# Plotting the rectangular graph
plt.figure(figsize=(8, 4), dpi=200)
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)

# Plotting the circles' edges
spheres = []
for i, (x, y, r) in enumerate(zip(obs_x, obs_y, obs_r)):
    circle = plt.Circle((x, y), r, facecolor='none', edgecolor=colors[i % len(colors)], linewidth=2.5)
    plt.gca().add_patch(circle)
    plt.text(x, y + r, str(i), ha='center', va='bottom')
    spheres.append([x, y, r])

# Setting the x-axis tick spacing
# plt.xticks(range(0, 80, 20))
# Displaying the graph
plt.show()

# %%
obs_info = "spheres"
data = initialize_npz(spheres)
data['velocity'], data['pressure'] = run(max_step=10000)
is_solid.fill(0)

parent_dict[obs_info] = data

# %%
np.savez('GNS_Obstacle.npz', **parent_dict)

# %%


timestep = 2
total_timestep = 500
# Assume x_range and y_range are given
x_range = [-0.01, 0.8]
y_range = [-0.02, 0.8]

# Call the function
for obs in parent_dict.keys():
    make_animation(parent_dict, obs, total_timestep, nodes_lx, nodes_ly, x_range, y_range)
    # plot_field(parent_dict, obs, total_timestep, nodes_lx, nodes_ly, x_range, y_range)
