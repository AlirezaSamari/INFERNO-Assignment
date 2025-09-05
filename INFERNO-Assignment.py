# Author: Alireza Samari
import numpy as np

def solver_fdm(Pe, nx=300, ny=30, l_star = 10, tol=1e-6, shape_diff_tol=1e-2, max_iterations=500000):
    if Pe > 10: tol=1e-8
    # setup dimensionless grid
    h = 0.03
    l = l_star*h
    h_star = 1.0
    dx = l_star/(nx - 1)
    dy = h_star/(ny - 1)
    x = np.linspace(0, l_star, nx)
    y = np.linspace(0, h_star, ny)
    X, Y = np.meshgrid(x, y)

    # initialize dimensionless temperature and velocity fields
    l_plate = 0.02 # hot and cold plate length
    T_star = np.zeros((ny, nx)) # temperature field initialized to 0
    l_plate_star_idx = int((l_plate / h) / dx) # finding the index of physical end of the hot and cold plates

    T_star[:, 0] = 0.
    T_star[0, :l_plate_star_idx] = 0. # cold plate temp
    T_star[-1, :l_plate_star_idx] = 1. # hot plate temp

    u_star = np.zeros((ny, nx)) # build the zero matrix of the velocity field
    for j in range(ny): u_star[j, :] = 6. * y[j] * (h_star - y[j]) / (h_star**2)

    # setup solver
    error = 1.
    iter = 0
    omega = 0.05 if Pe > 50 else 0.8
    while error > tol and iter < max_iterations:
        T_old = T_star.copy()
        # I used vectorized update to enhance solver
        conv_term = Pe * u_star[1:-1, 1:-1] * (T_old[1:-1, 2:] - T_old[1:-1, 0:-2]) / (2 * dx) #advection/convection terem
        # diffusion term
        diff_x = (T_old[1:-1, 2:] + T_old[1:-1, 0:-2]) / dx**2
        diff_y = (T_old[2:, 1:-1] + T_old[0:-2, 1:-1]) / dy**2
        T_new_interior = (diff_x + diff_y - conv_term) / ((2/dx**2) + (2/dy**2))

        T_star[1:-1, 1:-1] = (1 - omega) * T_old[1:-1, 1:-1] + omega * T_new_interior # under-relaxation for stability;
                                                                                    # for higher Pe it is better to consider lower value.
        # Neumann boundary conditions (adiabatic walls & outlet)
        T_star[0, l_plate_star_idx:] = T_star[1, l_plate_star_idx:]
        T_star[-1, l_plate_star_idx:] = T_star[-2, l_plate_star_idx:]
        T_star[:, -1] = T_star[:, -2]

        error = np.max(np.abs(T_star - T_old))
        iter += 1
        if iter % 50000 == 0: print(f"Iteration: {iter}, Error: {error:.2e}")
    if iter < max_iterations: print(f"Solver converged after {iter} iterations for Pe={Pe}.")
    else: print(f"Max iterations reached for Pe={Pe}.")

    # development location of the temp. profile:
    x_coords = X[0, :]
    prev_norm_profile = None
    t0 = 1.0 # hot wall temp as ref. temp
    for i in range(1, T_star.shape[1]):
      current_profile_T_star = T_star[:, i]
      current_profile_u_star = u_star[:, i]
      # mean temp at this cross-section
      tm = np.sum(current_profile_u_star * current_profile_T_star) / np.sum(current_profile_u_star)
      # normalized profile
      current_norm_profile = (t0 - current_profile_T_star) / (t0 - tm + 1e-9)

      if prev_norm_profile is not None:
          shape_difference = np.sum(np.abs(current_norm_profile - prev_norm_profile))
          if shape_difference < shape_diff_tol:
              entrance_length = x_coords[i]
              print(f"Profile is fully developed at x/H = {entrance_length:.3f} (idx={i})")
              return T_star, X, Y, i, entrance_length

      # update the previous profile
      prev_norm_profile = current_norm_profile

    print("Profile did not fully develop within the channel length.")
    last_index = T_star.shape[1] - 1
    return T_star, X, Y, last_index, x_coords[-1]

def find_location_by_sensitivity(Pe, Pe_ratio=1.05):
    Pe_prime = Pe * Pe_ratio
    T_star_Pe, X_grid, _, _, _= solver_fdm(Pe)
    T_star_Pe_prime, _, _,_,_ = solver_fdm(Pe_prime)
    # calculate the sensitivity at each location
    sensitivity = []
    for i in range(1, X_grid.shape[1] - 1):
        profile1 = T_star_Pe[:, i]
        profile2 = T_star_Pe_prime[:, i]
        diff = np.sum(np.abs(profile1 - profile2))# sensitivity is equal to sum of the absolute differences.
        sensitivity.append(diff)

    best_loc_idx = np.argmax(sensitivity) + 1
    best_x_pos = X_grid[0, best_loc_idx]

    print(f"Sensitive location is x/h = {best_x_pos:.3f}, (idx={best_loc_idx})")
    return best_loc_idx, best_x_pos