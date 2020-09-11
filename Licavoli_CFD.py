#!/usr/bin/env python

"""
Author: Adam Licavoli
Date: 8/23/2020


Engine Inlet Design:
    o B = Boundary
    o x = Interior cell

    |<--LS---->|<------------- 3.2 m ------------------->|<1x>|

    B  B  B ...
-----------------------------.                       - - - - - -.
|   B  x  x ...               ------.                          /
|   B  x                             ------.                  /
|   .                                       -----.        Theta = 10.94
|   .  |                                          -----.    /
1.0m   22                                               -------
|   .  |                                               ... x  B
|   .
|   . 
|   B  x
|   B  x  x  ...         --42--                        ... x  B
--  -----------------------------------------------------------
    B  B  B  ...                                       ... B  B
    O



Data storage and naming scheme:
  
              ^                    ^
              |                    |
            NS_0,2              NS_1,2
              |                    |
----|------SS_0,2------|-------SS_1,2-----|----
    |                  |                  |
    |                  |                  |
  SW_0,1       .      SW_1,1      .      SW_2,1 -NW_2,1->
    |       U_0,1      |       U_1,1      |
    |                  |                  |
    |                  |                  |
----|------SS_0,1------|-------SS_1,1-----|----
    |                  |                  |
    |                  |                  |
  SW_0,0       .      SW_1,0      .      SW_2,0 -NW_2,0->
    |       U_0,0      |       U_1,0      |
    |                  |                  |
    |                  |                  |
----O------SS_0,0------|------SS_1,0------|----
    |                  |                  |
    
SS - side length south
SW - side length west

NS - normal vector south
NW - normal vector west

norm_west, norm_south
Dim 0: Spacial X
Dim 1: Spacial Y
Dim 3: [ihat, jhat]

U Array
Dim 0: [rho, u*rho, v*rho, e]
Dim 1: Spacial X
Dim 2: Spacial Y
Dim 3: Temporal (Current, Next) or (Next, Current) depending on state

V Array
Dim 0: [rho, u, v, p]
Dim 1: Spacial X
Dim 2: Spacial Y

All other spacially distributed quantities
Dim 0: Spacial X
Dim 1: Spacial Y
"""

from os import path, system
from collections import defaultdict
import numpy as np
import h5py
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

plt.close("all")
DEBUG = False


#
# Define User Controls
#
MAX_NUM_T_STEP = 1500
CONVERGEANCE_TOL = 1.0e-5
N = 1  # Grid refinement integer
T_DIM_SIZE = 2
NU = 0.8
PROJECT_FILE_PREFIX = "Licavoli_Project_"
FILE_SPEC = "licavoli_n%d_project.hdf5"
SAVE_FILE = FILE_SPEC % N
FIG_SIZE = (11, 5.5)
FIG_RES = 500

#
# Define Gas Constants
#
GAMMA = 1.4
BETA = GAMMA - 1.0
ALPHA = (GAMMA + 1.0) / (GAMMA - 1)
R = 287.1
DEG2RAD = np.pi / 180.0


def conservative_to_primative(U):
    """
    Convert from conservative state vectors to primitive state vectors

    Parameters
    ----------
    U : np.array
        Conservative state vector.

    Returns
    -------
    V : np.array
        Primitive state vector.

    """
    V = np.zeros(U.shape) * np.nan
    V[0] = U[0]  # rho
    V[1] = U[1] / U[0]  # u
    V[2] = U[2] / U[0]  # v
    V[3] = (U[3] - 0.5 * (U[1] ** 2 + U[2] ** 2) / U[0]) * BETA
    return V


def primative_to_conservative(density, pressure, x_velocity, y_velocity):
    """
    Create a conservative state array from primatives

    Parameters
    ----------
    density : double
    pressure : double
    x_velocity : double
    y_velocity : double

    Returns
    -------
    np.array
        Conservative state array.

    """
    energy = pressure / BETA + density * (x_velocity ** 2.0 + y_velocity ** 2.0) / 2.0
    x_momentum = x_velocity * density
    y_momentum = y_velocity * density
    U = np.array([density, x_momentum, y_momentum, energy]).reshape(4, 1, 1, 1)
    return U


#
# Define Inlet Conditions
#
P1 = 1.0e5  # N/m^2
RHO1 = 1.0  # kg/m^3
M1 = 2.9  # nd
U1 = M1 * ((GAMMA * P1 / RHO1) ** 0.5)
U_INLET = primative_to_conservative(RHO1, P1, U1, 0.0)

#
# Inlet Design
#
IL = 42 * N  # nd
IS = 6 * N  # nd
JL = 22 * N  # nd
H = 1.0  # m
L = 3.2  # m
LS = IS * L / IL
TOTAL_L = L + LS
THETA = 10.94 * DEG2RAD  # rad

#
# Define ENUMS
#
SOUTH = 0
EAST = 1
NORTH = 2
WEST = 3

#
# Main function
#
def do_project():
    """
    Setup and converge a Stager Warming flux vector scheme to compute steady
    state conditions for a supersonic engine inlet

    Returns
    -------
    None.

    """

    #
    # Mesh the engine inlet. Using algebraic initial mesh
    # followed by elliptical smoothing
    #
    mesh = gen_engine_inlet_mesh()

    U = init_state()

    U[:, :, :, 0] = update_boundary_conditions(U[:, :, :, 0])
    U[:, :, :, 1] = update_boundary_conditions(U[:, :, :, 1])

    V = conservative_to_primative(U[:, :, :, 0])

    #
    # Perform onetime mesh properties calc. Calculate cell volume
    # cell wall lengths, and cell wall normal vectors
    #
    side_south, side_west, norm_south, norm_west, vol, dx, dy = calc_mesh_props(mesh)

    #
    # Track quantities to study
    #
    study_data = defaultdict(list)

    #
    # Loop through time
    #
    time = 0.0
    rel_pressure_delta = 525600 + 24601  # Just a large number
    t_step = 0
    while t_step <= MAX_NUM_T_STEP and rel_pressure_delta > CONVERGEANCE_TOL:

        #
        # Alternate between two U time dimensions
        #
        t_idx = t_step % T_DIM_SIZE
        next_t_idx = (t_step + 1) % T_DIM_SIZE

        #
        # Calc speed of sound, do not calc C^2 twice
        #
        C, C_2 = calc_speed_of_sound(V)

        #
        # Calc a stable time step and track total time
        #
        dt = calc_stable_dt(V, dx, dy, C)

        #
        # Step the solution in time
        #
        U = steger_warming_scheme(
            U,
            dt,
            vol,
            t_idx,
            next_t_idx,
            V,
            C,
            C_2,
            side_south,
            side_west,
            norm_south,
            norm_west,
        )

        #
        # Evaluate the convergence error
        #
        pressures = V[3]

        #
        # Enforce boundary conditions
        #
        U[:, :, :, next_t_idx] = update_boundary_conditions(U[:, :, :, next_t_idx])

        V = conservative_to_primative(U[:, :, :, next_t_idx])
        rel_pressure_delta = np.max(np.abs(pressures - V[3]) / V[3])
        print(
            "t step = %4d  Max pressure time delta = %12.5f, time = %6f, dt = %6f"
            % (t_step, rel_pressure_delta, time, dt)
        )

        #
        # Step the clocks
        #
        t_step += 1
        time += dt

        #
        # Track the study data
        #
        study_data["rel_pressure_delta"].append(rel_pressure_delta)
        study_data["dt"].append(dt)
        study_data["time"].append(time)
        study_data["total_system_energy"].append(np.sum(U[3, :, :, t_idx] * vol))

    #
    # Track the static data
    #
    study_data["convergence_tol"] = CONVERGEANCE_TOL
    study_data["n"] = N
    study_data["x_mesh"] = mesh["center_nodes_x"]
    study_data["y_mesh"] = mesh["center_nodes_y"]
    study_data["x_border"] = mesh["border_nodes_x"]
    study_data["y_border"] = mesh["border_nodes_y"]
    study_data["vertical_normals"] = norm_south
    study_data["horizontal_normals"] = norm_west

    #
    # Track the final states
    #
    study_data["U_tf"] = U[:, :, :, t_idx]

    if not np.isnan(rel_pressure_delta):
        #
        # Save progress
        #
        with h5py.File(SAVE_FILE, "w") as f:
            for item in study_data:
                f.create_dataset(item, data=study_data[item])

        #
        # Generate Plots
        #
        plot_project()
    else:
        print("Unstable Solution")

    system('say -v Samantha "Yo! This program is done, come check it"')

##############################################################################
# Problem Specific Functions
##############################################################################

def init_state():
    """
    Initialize the conservative state array. Utilize lower resolution solutions
    if available.  If not, tile the inlet condition across the spacial 
    dimensions

    Returns
    -------
    U : np.array
        Conservative state array.

    """
    #
    # Init the state array
    #
    U = np.tile(U_INLET, (1, IL, JL, T_DIM_SIZE)).squeeze()

    #
    # Use a lower resolution starting condition if available
    #
    if N > 1 and path.exists(FILE_SPEC % 1):
        with h5py.File(FILE_SPEC % 1, "r") as f:
            for xi in range(0, N):
                for eta in range(0, N):
                    U[:, xi::N, eta::N, 0] = f["U_tf"][:]
    return U


def update_boundary_conditions(U):
    """
    Counter flow out of the top and bottom walls to enforce a zero flow.
    Repeat the outlet state.

    Parameters
    ----------
    U : np.array
        Conservative state array.

    Returns
    -------
    U : np.array
        Conservative state array.

    """

    # Inlet
    U[:, 0, 1 : JL - 2] = np.tile(U_INLET, U[0, 0, 1 : JL - 2].shape).squeeze()

    # Outlet
    U[:, IL - 1, 1 : (JL - 1)] = U[:, IL - 2, 1 : (JL - 1)]

    # Lower wall
    for xi in range(0, IL):
        U[:, xi, 0] = np.array([U[0, xi, 1], U[1, xi, 1], -U[2, xi, 1], U[3, xi, 1]])

    # Top wall before initial shock
    for xi in range(0, IS):
        U[:, xi, JL - 1] = np.array(
            [U[0, xi, JL - 2], U[1, xi, JL - 2], -U[2, xi, JL - 2], U[3, xi, JL - 2],]
        )

    # Top wall after initial shock
    for xi in range(IS, IL):
        U[:, xi, JL - 1] = np.array(
            [
                U[0, xi, JL - 2],
                (U[1, xi, (JL - 2)] * np.cos(-2.0 * THETA))
                + (U[2, xi, (JL - 2)] * np.sin(-2.0 * THETA)),
                (U[1, xi, (JL - 2)] * np.sin(-2.0 * THETA))
                - (U[2, xi, (JL - 2)] * np.cos(-2.0 * THETA)),
                U[3, xi, JL - 2],
            ]
        )

    return U

def gen_engine_inlet_mesh():
    """
        Project specific method to generate a mesh

    Returns
    -------
    mesh : dictionary
        Dictionary of border node and cell center coordinates.

    """

    #
    # Algebraic mesh
    #
    border_x = np.zeros((IL + 1, JL + 1)) * np.nan
    border_y = border_x.copy()
    center_x = np.zeros((IL, JL)) * np.nan
    center_y = center_x.copy()

    dx = L / (IL - IS - 1.0)
    for xi in range(0, IL + 1):
        if xi < IS:
            dy = H / (JL - 2.0)
        else:
            dy = (H - np.arctan(THETA) * (xi - IS) * dx) / (JL - 2.0)

        for eta in range(0, JL + 1):
            border_y[xi, eta] = (eta * dy) - dy
            border_x[xi, eta] = dx * xi

    border_x, border_y = solve_eliptical(border_x, border_y)

    for xi in range(0, IL):
        for eta in range(0, JL):
            #
            # CG method
            #
            center_x[xi, eta] = (
                border_x[xi + 1, eta]
                + border_x[xi, eta]
                + border_x[xi + 1, eta + 1]
                + border_x[xi, eta + 1]
            ) / 4.0
            center_y[xi, eta] = (
                border_y[xi + 1, eta]
                + border_y[xi, eta]
                + border_y[xi + 1, eta + 1]
                + border_y[xi, eta + 1]
            ) / 4.0

        mesh = {
            "border_nodes_x": border_x,
            "border_nodes_y": border_y,
            "center_nodes_x": center_x,
            "center_nodes_y": center_y,
        }

    return mesh

##############################################################################
# General Functions
##############################################################################

def steger_warming_scheme(
        U,
        dt,
        vol,
        t_idx,
        next_t_idx,
        V,
        C,
        C_2,
        side_south,
        side_west,
        norm_south,
        norm_west,
):
    """
    

    Parameters
    ----------
    U : TYPE
        Conservative state vector.
    dt : TYPE
        Time step.
    vol : TYPE
        Cell volumes.
    t_idx : TYPE
        Current time index.
    next_t_idx : TYPE
        Next time index.
    V : TYPE
        Primitive state vector.
    C : TYPE
        Speed of sound.
    C_2 : TYPE
        Speed of sound squared.
    side_south : TYPE
        Southern cell wall lengths.
    side_west : TYPE
        Western cell wall lengths.
    norm_south : TYPE
        Southern wall normal vectors.
    norm_west : TYPE
        Western wall normal vectors.

    Returns
    -------
    U : np.array
        Updated conservative state vector.

    """

    for xi in range(1, IL - 1):
        for eta in range(1, JL - 1):

            north_wall = (norm_south[xi, eta + 1, :], side_south[xi, eta + 1])
            east_wall = (norm_west[xi + 1, eta, :], side_west[xi + 1, eta])
            south_wall = (norm_south[xi, eta, :], side_south[xi, eta])
            west_wall = (norm_west[xi, eta, :], side_west[xi, eta])

            f_plus_north = calc_split_flux2(
                V[:, xi, eta + 1],
                U[:, xi, eta + 1, t_idx],
                C[xi, eta + 1],
                C_2[xi, eta + 1],
                north_wall,
                -1.0,
            )

            f_minus_north = calc_split_flux2(
                V[:, xi, eta],
                U[:, xi, eta, t_idx],
                C[xi, eta],
                C_2[xi, eta],
                north_wall,
                1.0,
            )

            f_minus_east = calc_split_flux2(
                V[:, xi + 1, eta],
                U[:, xi + 1, eta, t_idx],
                C[xi + 1, eta],
                C_2[xi + 1, eta],
                east_wall,
                -1.0,
            )

            f_plus_east = calc_split_flux2(
                V[:, xi, eta],
                U[:, xi, eta, t_idx],
                C[xi, eta],
                C_2[xi, eta],
                east_wall,
                1.0,
            )

            f_plus_south = calc_split_flux2(
                V[:, xi, eta],
                U[:, xi, eta, t_idx],
                C[xi, eta],
                C_2[xi, eta],
                south_wall,
                -1.0,
            )

            f_minus_south = calc_split_flux2(
                V[:, xi, eta - 1],
                U[:, xi, eta - 1, t_idx],
                C[xi, eta - 1],
                C_2[xi, eta - 1],
                south_wall,
                1.0,
            )

            f_minus_west = calc_split_flux2(
                V[:, xi, eta],
                U[:, xi, eta, t_idx],
                C[xi, eta],
                C_2[xi, eta],
                west_wall,
                -1.0,
            )

            f_plus_west = calc_split_flux2(
                V[:, xi - 1, eta],
                U[:, xi - 1, eta, t_idx],
                C[xi - 1, eta],
                C_2[xi - 1, eta],
                west_wall,
                1.0,
            )

            U[:, xi, eta, next_t_idx] = U[:, xi, eta, t_idx] - (
                (
                    (
                        + (f_plus_north + f_minus_north)
                        + (f_minus_east + f_plus_east)
                        - (f_plus_south + f_minus_south)
                        - (f_minus_west + f_plus_west)
                    )
                    * dt
                )
                / vol[xi, eta]
            )

    return U


#
# Calculate P and P_inv matrix constants
#
P_MODEL = np.ones((4, 4))
P_MODEL[0, 2] = 0.0

P_INV_MODEL = np.ones((4, 4))
P_INV_MODEL[1, 3] = BETA
P_INV_MODEL[2, 3] = 0.0
P_INV_MODEL[3, 3] = BETA

HALF_OVER_BETA = 0.5 / BETA


def calc_split_flux2(V, U, C, C_2, wall_props, polarity):
    """
    Calculate the split fluxes accross a single cell wall. Pre multiply with
    cell wall length

    Parameters
    ----------
    V : TYPE
        Primative state vector.
    U : TYPE
        Conservative state vector.
    C : TYPE
        Speed of sound.
    C_2 : TYPE
        Speed of sound squred.
    wall_props : TYPE
        Tuple containing wall normal vector and length.
    polarity : float
        -1 or 1 depending on split flux eigen vector assignment.

    Returns
    -------
    split_F : TYPE
        Split flux * wall length.

    """

    normal_vec, side_length = wall_props[:]
    #
    # Only calculate 1/(2*c^2) and 1/(2*C) once
    #
    one_over_2_c_2 = 0.5 / C_2
    one_over_2_c = 0.5 / C

    P = P_MODEL

    P_inv = P_INV_MODEL

    #
    # Calculate P and P_inv direction independent constants
    #
    alpha = ((V[1] ** 2.0) + (V[2] ** 2.0)) / 2.0

    P[0, 1] = one_over_2_c_2
    P[0, 3] = one_over_2_c_2

    P[1, 0] = V[1]
    P[2, 0] = V[2]
    P[3, 0] = alpha

    P_inv[0, 0] = 1.0 - (2.0 * alpha * BETA * one_over_2_c_2)
    P_inv[0, 1] = 2.0 * V[1] * BETA * one_over_2_c_2
    P_inv[0, 2] = 2.0 * V[2] * BETA * one_over_2_c_2
    P_inv[0, 3] = -2.0 * BETA * one_over_2_c_2

    kx = normal_vec[0]
    ky = normal_vec[1]

    u_prime = (kx * V[1]) + (ky * V[2])
    v_prime = (-ky * V[1]) + (kx * V[2])

    #
    # Calculate remaining P and P_inv components
    #
    P[1, 1] = (V[1] * one_over_2_c_2) + kx * one_over_2_c
    P[1, 2] = -ky * V[0]
    P[1, 3] = V[1] * one_over_2_c_2 - kx * one_over_2_c

    P[2, 1] = V[2] * one_over_2_c_2 + ky * one_over_2_c
    P[2, 2] = kx * V[0]
    P[2, 3] = V[2] * one_over_2_c_2 - ky * one_over_2_c

    P[3, 1] = alpha * one_over_2_c_2 + u_prime * one_over_2_c + HALF_OVER_BETA
    P[3, 2] = V[0] * v_prime
    P[3, 3] = alpha * one_over_2_c_2 - u_prime * one_over_2_c + HALF_OVER_BETA

    P_inv[1, 0] = alpha * BETA - u_prime * C
    P_inv[1, 1] = -V[1] * BETA + kx * C
    P_inv[1, 2] = -V[2] * BETA + ky * C

    P_inv[2, 0] = -v_prime / V[0]
    P_inv[2, 1] = -ky / V[0]
    P_inv[2, 2] = kx / V[0]

    P_inv[3, 0] = alpha * BETA + u_prime * C
    P_inv[3, 1] = -V[1] * BETA - kx * C
    P_inv[3, 2] = -V[2] * BETA - ky * C

    if DEBUG:
        if np.linalg.norm(np.matmul(P, P_inv) - np.eye(4)) > 10e-6:
            print("Error, P*Pinv != eye(4)")
    #
    # Calculate split A matrix
    #
    tent = np.eye(4)
    tent[0, 0] = (u_prime + (polarity * np.abs(u_prime))) / 2.0
    tent[1, 1] = (u_prime + C + (polarity * np.abs(u_prime + C))) / 2.0
    tent[2, 2] = tent[0, 0]
    tent[3, 3] = (u_prime - C + (polarity * np.abs(u_prime - C))) / 2.0

    #
    # Calculate split flux
    #
    return np.matmul(np.matmul(P, np.matmul(tent, P_inv)), U) * side_length


def calc_mesh_props(mesh):
    """
    General method to derive side lengths and normal vectors from a mesh

    Parameters
    ----------
    mesh : dictionary
        Dictionary of border node and cell center coordinates.

    Returns
    -------
    side_south : np.array
        Horizontal Side Lengths.
    side_west : np.array
        Vertical Side Lengths.
    norm_west : np.array
        Western Normal Vectors.
    norm_south : np.array
        Northern Normal Vectors.
    vol : np.array
        Cell volumes.
    center_dx : np.array
        Cell center dx.
    center_dy : np.array
        Cell center dy.

    """
    border_x = mesh["border_nodes_x"]
    border_y = mesh["border_nodes_y"]
    center_x = mesh["center_nodes_x"]
    center_y = mesh["center_nodes_y"]

    side_south = np.zeros((border_y.shape[0] - 1, border_y.shape[1] - 1)) * np.nan
    side_west = side_south.copy()
    vol = side_south.copy()
    norm_west = np.zeros((border_y.shape[0] - 1, border_y.shape[1] - 1, 2)) * np.nan
    norm_south = norm_west.copy()
    center_dx = side_south.copy()
    center_dy = side_south.copy()

    for xi in range(0, border_x.shape[0] - 1):
        for eta in range(0, border_x.shape[1] - 1):
            vert_surf = np.array(
                [
                    border_x[xi, eta + 1] - border_x[xi, eta],
                    border_y[xi, eta + 1] - border_y[xi, eta],
                ]
            )

            horz_surf = np.array(
                [
                    border_x[xi + 1, eta] - border_x[xi, eta],
                    border_y[xi + 1, eta] - border_y[xi, eta],
                ]
            )

            side_south[xi, eta] = np.linalg.norm(horz_surf)
            side_west[xi, eta] = np.linalg.norm(vert_surf)

            norm_south[xi, eta, 0] = -horz_surf[1]
            norm_south[xi, eta, 1] = horz_surf[0]
            norm_south[xi, eta, :] = norm_south[xi, eta, :] / side_south[xi, eta]

            norm_west[xi, eta, 0] = vert_surf[1]
            norm_west[xi, eta, 1] = -vert_surf[0]
            norm_west[xi, eta, :] = norm_west[xi, eta, :] / side_west[xi, eta]

            # Volume Calc
            vol[xi, eta] = 0.5 * np.abs(
                (
                    (border_x[xi + 1, eta + 1] - border_x[xi, eta])
                    * (border_y[xi, eta + 1] - border_y[xi, eta])
                )
                - (
                    (border_x[xi, eta + 1] - border_x[xi, eta])
                    * (border_y[xi + 1, eta + 1] - border_y[xi, eta])
                )
            )

            vol[xi, eta] = vol[xi, eta] + 0.5 * np.abs(
                (
                    (border_x[xi + 1, eta] - border_x[xi, eta])
                    * (border_y[xi + 1, eta + 1] - border_y[xi, eta])
                )
                - (
                    (border_x[xi + 1, eta + 1] - border_x[xi, eta])
                    * (border_y[xi + 1, eta] - border_y[xi, eta])
                )
            )

            if xi < border_x.shape[0] - 2:
                center_dx[xi, eta] = center_x[xi + 1, eta] - center_x[xi, eta]

            if eta < border_x.shape[1] - 2:
                center_dy[xi, eta] = center_y[xi, eta + 1] - center_y[xi, eta]

    return side_south, side_west, norm_south, norm_west, vol, center_dx, center_dy


def calc_speed_of_sound(V):
    """
    General method to calc the speed of sound from a primitive variable array

    Parameters
    ----------
    V : np.array
        Primitive Variable Array.

    Returns
    -------
    C : np.array
        Speed of sound.
    C_2 : np.array
        Speed of sound squared.

    """
    C_2 = GAMMA * V[3] / V[0]
    C = C_2 ** 0.5
    return C, C_2


def calc_stable_dt(V, dx, dy, C):
    """
    General method to calculate a dt for enforce the CFL condition

    Parameters
    ----------
    V : np.array
        Primitive state vectors.
    dx : np.array
        Cell center dx.
    dy : np.array
        Cell center dy.
    C : np.array
        Speed of sound.

    Returns
    -------
    double
        time step.

    """
    dt = NU / (
        np.abs(V[1] / dx)
        + np.abs(V[2] / dy)
        + C * (1.0 / (dx ** 2.0) + 1.0 / (dy ** 2.0)) ** 0.5
    )

    return np.nanmin(dt)


def solve_eliptical(grid_X_initial, grid_Y_initial):
    """
    General method to shift grid nodes to satisfy an elliptical differential equation

    Parameters
    ----------
    grid_X_initial : np.array
        Initial grid node x positions.
    grid_Y_initial : np.array
        Initial grid node y positions.

    Returns
    -------
    grid_x : np.array
        Elliptical grid x positions.
    grid_y : np.array
        Elliptical grid y positions.
    """
    grid_x = grid_X_initial.copy()
    grid_y = grid_Y_initial.copy()
    grid_max_error = 100.0
    jacobian = grid_x.copy() * np.nan

    while abs(grid_max_error) > 1e-5:
        grid_max_error = 0.0
        for xi in range(IL + 1):
            for eta in range(JL + 1):

                # Do not recalculate boundary position
                if eta == 0:
                    x_eta = grid_x[xi, eta + 1] - grid_x[xi, eta]
                    y_eta = grid_y[xi, eta + 1] - grid_y[xi, eta]

                if xi == 0:
                    x_xi = grid_x[xi + 1, eta] - grid_x[xi, eta]
                    y_xi = grid_y[xi + 1, eta] - grid_y[xi, eta]

                if eta == JL:
                    x_eta = grid_x[xi, eta] - grid_x[xi, eta - 1]
                    y_eta = grid_y[xi, eta] - grid_y[xi, eta - 1]
                    
                if xi == IL:
                    x_xi = grid_x[xi, eta] - grid_x[xi - 1, eta]
                    y_xi = grid_y[xi, eta] - grid_y[xi - 1, eta]

                if (eta > 0) and (eta < JL) and (xi > 0) and (xi < IL):

                    #
                    # Compute Partial Derivatives with central difference
                    #
                    x_xi = (grid_x[xi + 1, eta] - grid_x[xi - 1, eta]) / 2
                    x_eta = (grid_x[xi, eta + 1] - grid_x[xi, eta - 1]) / 2
                    x_xi_xi = grid_x[xi + 1, eta] - 2 * grid_x[xi, eta] + grid_x[xi - 1, eta]

                    x_xi_eta = (
                        ((grid_x[xi + 1, eta + 1] - grid_x[xi - 1, eta + 1]) / 2)
                        - ((grid_x[xi + 1, eta - 1] - grid_x[xi - 1, eta - 1]) / 2)
                    ) / 2
                    x_eta_eta = (
                        grid_x[xi, eta + 1] - 2 * grid_x[xi, eta] + grid_x[xi, eta - 1]
                    )

                    y_xi = (grid_y[xi + 1, eta] - grid_y[xi - 1, eta]) / 2
                    y_eta = (grid_y[xi, eta + 1] - grid_y[xi, eta - 1]) / 2
                    y_xi_xi = grid_y[xi + 1, eta] - 2 * grid_y[xi, eta] + grid_y[xi - 1, eta]
                    
                    y_xi_eta = (
                        ((grid_y[xi + 1, eta + 1] - grid_y[xi - 1, eta + 1]) / 2)
                        - ((grid_y[xi + 1, eta - 1] - grid_y[xi - 1, eta - 1]) / 2)
                    ) / 2
                    y_eta_eta = grid_y[xi, eta + 1] - 2 * grid_y[xi, eta] + grid_y[xi, eta - 1]
                    

                    #
                    # Compute common quantities
                    #
                    alpha = x_eta ** 2 + y_eta ** 2
                    beta = x_xi * x_eta + y_xi * y_eta
                    gamma = x_xi ** 2 + y_xi ** 2

                    alpha2_gamma2 = 2.0 * alpha + 2.0 * gamma

                    #
                    # Solve elliptical equation via Gauss-Seidel method
                    #
                    dx = (
                        -(-alpha * x_xi_xi + 2.0 * beta * x_xi_eta - gamma * x_eta_eta)
                        / alpha2_gamma2
                    )
                    dy = (
                        -(-alpha * y_xi_xi + 2.0 * beta * y_xi_eta - gamma * y_eta_eta)
                        / alpha2_gamma2
                    )

                    #
                    # Track max updates for convergence criteria
                    #
                    grid_max_error = max([grid_max_error, dx, dy])

                    #
                    # Update the physical dimensions of the mesh
                    #
                    grid_x[xi, eta] += dx
                    grid_y[xi, eta] += dy

                jacobian[xi, eta] = x_xi * y_eta - x_eta * y_xi

    return grid_x, grid_y


def calc_v_truth(x_domain, y_scalar):
    """
    Calculate the primative state for a line traversing the inlet for 
    a given y position.
    
    Flow states calculated using
    
    https://www.engineering.com/calculators/oblique_flow_relations.htm

    Parameters
    ----------
    x_domain : TYPE
        DESCRIPTION.
    y_scalar : TYPE
        DESCRIPTION.

    Returns
    -------
    V : TYPE
        DESCRIPTION.

    """

    # Y must be a scalar

    beta_1 =  28.9996490 * DEG2RAD
    beta_2 =  34.2188729 * DEG2RAD - THETA
    beta_3 =  41.6006055 * DEG2RAD

    m2 = 2.37808978
    p2 = 2.141e5
    rho2 = RHO1 * 1.69993936
    u_mag_2 = m2 * ((GAMMA * p2 / rho2) ** 0.5)
    u2 = np.cos(THETA) * u_mag_2
    v2 = -np.sin(THETA) * u_mag_2

    rho3 = rho2 * 1.58073445
    m3 = 1.94244995
    p3 = 4.109e5
    u3 = m3 * ((GAMMA * p3 / rho3) ** 0.5)

    rho4 = rho3 * 1.49767216
    m4 = 1.55163830
    p4 = 7.287e5
    u_mag_4 = m4 * ((GAMMA * p4 / rho4) ** 0.5)
    u4 = np.cos(THETA) * u_mag_4
    v4 = -np.sin(THETA) * u_mag_4

    V_1 = np.array([RHO1, U1, 0.0, P1])
    V_2 = np.array([rho2, u2, v2, p2])
    V_3 = np.array([rho3, u3, 0.0, p3])
    V_4 = np.array([rho4, u4, v4, p4])

    #
    # Calc shock wall intercepts
    # Points:
    # a - top wall, start of turn
    # b - Shock intercept with bottom wall
    # c - reflection 1 intercept with top wall
    # d - reflection 2 intercept with bottom wall (theoretical)
    #
    x_a = LS
    y_a = H

    x_b = x_a + (y_a / np.tan(beta_1))
    y_b = 0.0

    x_c = (np.tan(beta_2) * x_b + y_a + np.tan(THETA) * x_a) / (
        np.tan(beta_2) + np.tan(THETA)
    )
    y_c = (x_c - x_b) * np.tan(beta_2)

    x_d = x_c + (y_c / np.tan(beta_3))
    y_d = 0.0

    shock_1_2_intercept_y_target = np.interp(y_scalar, [y_b, y_a], [x_b, x_a])
    shock_2_3_intercept_y_target = np.interp(y_scalar, [y_b, y_c], [x_b, x_c])
    shock_3_4_intercept_y_target = np.interp(y_scalar, [y_d, y_c], [x_d, x_c])

    V = np.zeros((4, x_domain.shape[0])) * np.nan

    for x_idx, x in enumerate(x_domain):
        if x <= shock_1_2_intercept_y_target:
            V[:, x_idx] = V_1
        elif (x > shock_1_2_intercept_y_target) and (x <= shock_2_3_intercept_y_target):
            V[:, x_idx] = V_2
        elif (x > shock_2_3_intercept_y_target) and (x <= shock_3_4_intercept_y_target):
            V[:, x_idx] = V_3
        elif x > shock_3_4_intercept_y_target:
            V[:, x_idx] = V_4

    return V

def interp_V(x,y,V,xq, yq):
    V_interp = []
    for state in V:
        V_interp.append(griddata((x.reshape((-1)), y.reshape((-1))), state.reshape((-1)), (xq, yq)))
                        
    return V_interp


#
# Execute the project if script is executed.  Do not run if imported
#
if __name__ == "__main__":
    do_project()
