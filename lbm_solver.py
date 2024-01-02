import taichi as ti
import numpy as np
from pyevtk.hl import gridToVTK
from tqdm import tqdm

@ti.data_oriented
class LBMModel:
    def __init__(self, lx, ly, nu, v_left_np, timesteps):

        # Initialize Taichi
        ti.init(arch=ti.gpu)

        # Define types and dimensions
        self.dim = 2
        self.Q = 9
        real = ti.f32
        i32_vec2d = ti.types.vector(2, ti.i32)
        f32_vec2d = ti.types.vector(2, ti.f32)
        scalar_real = lambda: ti.field(dtype=real)
        scalar_int = lambda: ti.field(dtype=ti.i32)
        vec = lambda: ti.Vector.field(self.dim, dtype=real)
        vec_Q = lambda: ti.Vector.field(self.Q, dtype=real)

        # Setting the input parameters
        self.lx = lx
        self.ly = ly
        self.nu = nu
        self.v_left_np = v_left_np
        self.timesteps = timesteps

        # Initialize solid_count
        self.solid_count = 0
        self.nnodes_per_cell = 4

        # Allocate fields
        self.is_solid = scalar_int()
        self.v = vec()
        self.temp_v = vec()
        self.target_v = vec()
        self.target_rho = scalar_real()
        self.rho = scalar_real()
        self.collide_f = vec_Q()
        self.stream_f = vec_Q()

        # Place fields in the memory layout
        ti.root.dense(ti.ij, (self.lx, self.ly)).place(self.is_solid, self.target_v, self.target_rho, self.rho, self.v, self.temp_v, self.collide_f, self.stream_f)

        # Initialize other parameters and fields
        self._initialize_parameters()

    def _initialize_parameters(self):
        # LBM weights
        w_np = np.array(
            [4.0 / 9.0, 1.0 / 36.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 9.0])
        self.w = ti.field(ti.f32, shape=self.Q)
        self.w.from_numpy(w_np)

        # x and y components of predefined velocity in Q directions
        e_xy_list = [[0, 0], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1]]
        self.e_xy = ti.Vector.field(n=2, dtype=ti.i32, shape=self.Q)
        self.e_xy.from_numpy(np.array(e_xy_list))

        # reversed_e_xy_np stores the index of the opposite component to every component in e_xy_np
        self.reversed_e_index = np.array([e_xy_list.index([-a for a in e]) for e in e_xy_list])

        # MRT matrices
        M_np = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 0, -1, 0, 1, -1, -1, 1], [0, 0, 1, 0, -1, 1, 1, -1, -1],
             [0, 1, 1, 1, 1, 2, 2, 2, 2], [0, 1, -1, 1, -1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, -1, 1, -1],
             [0, 0, 0, 0, 0, 1, 1, -1, -1], [0, 0, 0, 0, 0, 1, -1, -1, 1], [0, 0, 0, 0, 0, 1, 1, 1, 1]])
        self.M_mat = ti.Matrix.field(self.Q, self.Q, ti.f32, shape=())
        self.M_mat[None] = ti.Matrix(M_np)

        inv_M_np = np.linalg.inv(M_np)
        self.inv_M_mat = ti.Matrix.field(self.Q, self.Q, ti.f32, shape=())
        self.inv_M_mat[None] = ti.Matrix(inv_M_np)

        # Diagonal relaxation matrix for fluid 1
        S_dig_vec_np = [1, 1.5, 1.4, 1, 1.5, 1, 1.5, 1., 1.]
        self.S_dig_vec = ti.Vector.field(self.Q, ti.f32, shape=())
        self.S_dig_vec[None] = ti.Vector(S_dig_vec_np)

        # Fluid properties
        self.v_left = ti.field(ti.f32, shape=self.ly)
        self.v_left.from_numpy(self.v_left_np)

        # Static fields
        ti.static(self.e_xy)
        ti.static(self.w)
        ti.static(self.M_mat)
        ti.static(self.inv_M_mat)
        ti.static(self.S_dig_vec)

        # Compute derived parameters
        self.nnodes = self.lx * self.ly  # Total number of nodes
        self.ndims = 2  # Number of dimensions
        self.ncells = (self.ly - 1) * (self.lx - 1)  # Total number of cells
        self.ncells_row = self.lx - 1  # Number of cells per row

        # Define physical parameters
        self.lx_physical = 0.8
        # Compute physical spacing between nodes
        self.dx = self.dy = self.lx_physical / self.lx

        # Define Lattice-Boltzmann model parameters
        self.lbm_dx = self.lx // self.lx  # Lattice spacing in x-direction
        self.lbm_dy = self.ly // self.ly  # Lattice spacing in y-direction

        self.j_indices = np.arange(self.nnodes)
        self.lbm_x = (self.j_indices % self.lx) * self.lbm_dy
        self.lbm_y = (self.j_indices // self.lx) * self.lbm_dy

        # Initialize parent_dict
        self.parent_dict = {}

    def place_sphere(self, spheres):
        for sphere in spheres:
            x, y, R = sphere
            xmin = max(0, x - R)
            ymin = max(0, y - R)
            xmax = min(self.lx, x + R + 1)
            ymax = min(self.ly, y + R + 1)

            for px in range(xmin, xmax):
                for py in range(ymin, ymax):
                    dx = px - x
                    dy = py - y

                    dist2 = dx * dx + dy * dy
                    R2 = R * R

                    if dist2 < R2 and self.is_solid[px, py] == 0:
                        self.is_solid[px, py] = 1
                        self.solid_count += 1

    @ti.func
    def periodic_index(self, i):
        iout = i
        if i[0] < 0:
            iout[0] = self.lx - 1
        if i[0] >= self.lx:
            iout[0] = 0
        if i[1] < 0:
            iout[1] = self.ly - 1
        if i[1] >= self.ly:
            iout[1] = 0

        return iout

    @ti.func
    def velocity_vec(self, local_pos):
        velocity_vec = ti.Vector([0.0, 0.0])
        for i in ti.static(range(2)):
            for s in ti.static(range(self.Q)):
                velocity_vec[i] += self.stream_f[local_pos][s] * self.e_xy[s][i]
            velocity_vec[i] /= self.rho[local_pos]

        return velocity_vec

    @ti.func
    def feq(self, k, rho_local, u):
        eu = self.e_xy[k].dot(u)
        uv = u.dot(u)
        feq_out = self.w[k] * rho_local * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv)
        return feq_out

    @ti.kernel
    def init_field(self):
        for x, y in ti.ndrange(self.lx, self.ly):
            self.rho[x, y] = 1.0
            self.v[x, y] = ti.Vector([0.0, 0.0])
            self.collide_f[x, y] = ti.Vector([0.0 for _ in range(self.Q)])
            self.stream_f[x, y] = ti.Vector([0.0 for _ in range(self.Q)])

            if self.is_solid[x, y] <= 0:
                for q in ti.static(range(self.Q)):
                    self.collide_f[x, y][q] = self.w[q] * self.rho[x, y]
                    self.stream_f[x, y][q] = self.w[q] * self.rho[x, y]

    @ti.kernel
    def collision(self):
        for I in ti.grouped(self.collide_f):
            if (I.x < self.lx and I.y < self.ly and self.is_solid[I] <= 0):
                # MRT operator
                v_I = self.velocity_vec(I)

                e = -4 * self.collide_f[I][0] + 2 * self.collide_f[I][1] - self.collide_f[I][2] + 2 * self.collide_f[I][
                    3] - self.collide_f[I][4] + 2 * self.collide_f[I][5] - self.collide_f[I][6] + 2 * self.collide_f[I][
                        7] - self.collide_f[I][8]
                eps = 4 * self.collide_f[I][0] + self.collide_f[I][1] - 2 * self.collide_f[I][2] + self.collide_f[I][
                    3] - 2 * self.collide_f[I][4] + self.collide_f[I][5] - 2 * self.collide_f[I][6] + self.collide_f[I][
                          7] - 2 * self.collide_f[I][8]

                j_x = self.collide_f[I][5] + self.collide_f[I][6] + self.collide_f[I][7] - self.collide_f[I][1] - \
                      self.collide_f[I][2] - self.collide_f[I][3]
                q_x = -self.collide_f[I][1] + 2 * self.collide_f[I][2] - self.collide_f[I][3] + self.collide_f[I][
                    5] - 2 * self.collide_f[I][6] + self.collide_f[I][7]
                j_y = self.collide_f[I][1] + self.collide_f[I][8] + self.collide_f[I][7] - self.collide_f[I][3] - \
                      self.collide_f[I][4] - self.collide_f[I][5]
                q_y = self.collide_f[I][1] - self.collide_f[I][3] + 2 * self.collide_f[I][4] - self.collide_f[I][5] + \
                      self.collide_f[I][7] - 2 * self.collide_f[I][8]
                p_xx = self.collide_f[I][2] - self.collide_f[I][4] + self.collide_f[I][6] - self.collide_f[I][8]
                p_xy = -self.collide_f[I][1] + self.collide_f[I][3] - self.collide_f[I][5] + self.collide_f[I][7]

                j_x2 = j_x * j_x
                j_y2 = j_y * j_y

                eO = e - self.S_dig_vec[None][1] * (e + 2 * self.rho[I] - 3 * (j_x2 + j_y2) / self.rho[I])
                epsO = eps - self.S_dig_vec[None][2] * (eps - self.rho[I] + 3 * (j_x2 + j_y2) / self.rho[I])
                q_xO = q_x - self.S_dig_vec[None][3] * (q_x + j_x)
                q_yO = q_y - self.S_dig_vec[None][6] * (q_y + j_y)
                p_xxO = p_xx - 1.0 / (3.0 * self.nu + 0.5) * (p_xx - (j_x2 - j_y2) / self.rho[I])
                p_xyO = p_xy - 1.0 / (3.0 * self.nu + 0.5) * (p_xy - j_x * j_y / self.rho[I])

                # Update the collision function based on the MRT model
                a = 1. / 36.
                self.collide_f[I][0] = a * (4 * self.rho[I] - 4 * eO + 4 * epsO)
                self.collide_f[I][1] = a * (4 * self.rho[I] + 2 * eO + epsO - 6 * j_x - 3 * q_xO + 6 * j_y + 3 * q_yO - 9 * p_xyO)
                self.collide_f[I][2] = a * (4 * self.rho[I] - eO - 2 * epsO - 6 * j_x + 6 * q_xO + 9 * p_xxO)
                self.collide_f[I][3] = a * (4 * self.rho[I] + 2 * eO + epsO - 6 * j_x - 3 * q_xO - 6 * j_y - 3 * q_yO + 9 * p_xyO)
                self.collide_f[I][4] = a * (4 * self.rho[I] - eO - 2 * epsO - 6 * j_y + 6 * q_yO - 9 * p_xxO)
                self.collide_f[I][5] = a * (4 * self.rho[I] + 2 * eO + epsO + 6 * j_x + 3 * q_xO - 6 * j_y - 3 * q_yO - 9 * p_xyO)
                self.collide_f[I][6] = a * (4 * self.rho[I] - eO - 2 * epsO + 6 * j_x - 6 * q_xO + 9 * p_xxO)
                self.collide_f[I][7] = a * (4 * self.rho[I] + 2 * eO + epsO + 6 * j_x + 3 * q_xO + 6 * j_y + 3 * q_yO + 9 * p_xyO)
                self.collide_f[I][8] = a * (4 * self.rho[I] - eO - 2 * epsO + 6 * j_y - 6 * q_yO - 9 * p_xxO)

    @ti.kernel
    def boundary_condition(self):
        for I in ti.grouped(self.v):
            if (I.x < self.lx and I.y < self.ly and self.is_solid[I] <= 0):
                for s in ti.static(range(self.Q)):
                    if I.x == 0:
                        if self.e_xy[s][0] == 1 and self.e_xy[s][1] == 0:
                            self.stream_f[I][s] = self.feq(s, self.rho[I], ti.Vector([self.v_left, self.v[I].y]))

        for I in ti.grouped(self.v):
            if (I.x < self.lx and I.y < self.ly and self.is_solid[I] <= 0):
                self.collide_f[I] = self.stream_f[I]
                self.rho[I] = self.collide_f[I].sum()


    @ti.kernel
    def streaming(self):
        for I in ti.grouped(self.collide_f):
            if (I.x < self.lx and I.y < self.ly and self.is_solid[I] <= 0):
                for s in ti.static(range(self.Q)):
                    neighbor_pos = self.periodic_index(I + self.e_xy[s])
                    if self.is_solid[neighbor_pos] <= 0:
                        self.stream_f[neighbor_pos][s] = self.collide_f[I][s]
                    else:
                        reverse_direction = self.reversed_e_index[s]
                        self.stream_f[I][reverse_direction] = self.collide_f[I][s]

    @ti.kernel
    def update_vel(self):
        for I in ti.grouped(self.v):
            self.v[I] = self.velocity_vec(I)

    def export_npz(self, timestep, velocity, pressure):
        v_ = self.v.to_numpy()[0:self.lx, 0:self.ly, :]
        rho_ = self.rho.to_numpy()[0:self.lx, 0:self.ly]
        velocity[timestep, :, 0] = v_[self.lbm_x, self.lbm_y, 0]
        velocity[timestep, :, 1] = v_[self.lbm_x, self.lbm_y, 1]
        pressure[timestep, :, 0] = rho_[[self.lbm_x, self.lbm_y]]
        return velocity, pressure

    def initialize_npz(self, spheres):
        for i in range(self.lx):
            self.is_solid[i, 0] = 1  # Bottom boundary
            for j in range(self.ly - 1, self.ly):
                self.is_solid[i, j] = 1  # Top boundary

        self.place_sphere(spheres)

        # Node positions
        pos = np.zeros((self.timesteps, self.nnodes, self.ndims))
        pos[:, :, 0] = (self.j_indices % self.lx) * self.dx
        pos[:, :, 1] = (self.j_indices // self.lx) * self.dy

        # Node types
        node_type = np.zeros((self.timesteps, self.nnodes, 1), dtype=int)

        # Create lbm_x and lbm_y arrays with shape (timesteps, nnodes, 1)
        lbm_x_arr = np.broadcast_to(self.lbm_x[None, :, None], (self.timesteps, self.nnodes, 1))
        lbm_y_arr = np.broadcast_to(self.lbm_y[None, :, None], (self.timesteps, self.nnodes, 1))

        # Assuming 'is_solid' is a 2D array with shape (lx, ly)
        is_solid_ = self.is_solid.to_numpy()[0:self.lx, 0:self.ly]

        # Update node_type based on conditions
        node_type[is_solid_[lbm_x_arr, lbm_y_arr] == 1] = 6
        node_type[lbm_x_arr == 0] = 4
        node_type[lbm_x_arr == self.lx - self.lbm_dy] = 5

        # Cells
        j_indices_cells = np.arange(self.ncells)
        cell = np.zeros((self.timesteps, self.ncells, self.nnodes_per_cell), dtype=int)
        cell[:, :, 0] = j_indices_cells + j_indices_cells // self.ncells_row
        cell[:, :, 1] = cell[:, :, 0] + 1
        cell[:, :, 2] = cell[:, :, 0] + self.ncells_row + 2
        cell[:, :, 3] = cell[:, :, 0] + self.ncells_row + 1

        # Create the data structure
        data = {
            "pos": pos,
            "node_type": node_type,
            "velocity": np.zeros((self.timesteps, self.nnodes, self.ndims)),  # To be filled during simulation
            "cells": cell,
            "pressure": np.zeros((self.timesteps, self.nnodes, 1))  # To be filled during simulation
        }

        return data