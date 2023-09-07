'''Author: Damian Navarro
    Final project for Math104B
'''

import numpy as np
import matplotlib.pyplot as plt
import zipfile

with zipfile.ZipFile('Math104Bfinal_codes.zip', 'w') as myzip:
    myzip.write('main.py')


def euler_method(t, x0, v0, k, m, dt):
    num_steps = len(t)
    x = np.zeros(num_steps)
    v = np.zeros(num_steps)
    x[0] = x0
    v[0] = v0

    for i in range(1, num_steps):
        x[i] = x[i-1] + dt * v[i-1]
        v[i] = v[i-1] + dt * (-k/m * x[i-1])

    return x, v


def rk4_method(t, x0, v0, k, m, dt):
    num_steps = len(t)
    x = np.zeros(num_steps)
    v = np.zeros(num_steps)
    x[0] = x0
    v[0] = v0

    for i in range(1, num_steps):
        k1x = dt * v[i-1]
        k1v = dt * (-k/m * x[i-1])

        k2x = dt * (v[i-1] + k1v/2)
        k2v = dt * (-k/m * (x[i-1] + k1x/2))

        k3x = dt * (v[i-1] + k2v/2)
        k3v = dt * (-k/m * (x[i-1] + k2x/2))

        k4x = dt * (v[i-1] + k3v)
        k4v = dt * (-k/m * (x[i-1] + k3x))

        x[i] = x[i-1] + (k1x + 2*k2x + 2*k3x + k4x)/6
        v[i] = v[i-1] + (k1v + 2*k2v + 2*k3v + k4v)/6

    return x, v

# Parameters
k = 1.0  # Spring constant
m = 1.0  # Mass

# Initial conditions
x0 = 1.0  # Initial displacement
v0 = 0.0  # Initial velocity

# Time parameters
t_start = 0.0
t_end = 10.0
dt_values = [0.1, 0.05, 0.025, 0.0125]  # Different time step sizes

# Exact solution function for comparison


def exact_solution(t):
    return x0 * np.cos(np.sqrt(k/m) * t)


# Calculate and print the arrays and absolute error
for dt in dt_values:
    t = np.arange(t_start, t_end, dt)
    num_steps = len(t)

    # Solve using Euler's Method
    x_euler, v_euler = euler_method(t, x0, v0, k, m, dt)

    # Solve using RK4 method
    x_rk4, v_rk4 = rk4_method(t, x0, v0, k, m, dt)

    # Calculate the exact solution
    x_exact = exact_solution(t)

    # Calculate the absolute error
    error_x = np.abs(x_euler - x_rk4)
    error_v = np.abs(v_euler - v_rk4)

    print(f"\nTime step size (dt): {dt}")
    print("Euler's Method")
    print("Displacement:", ', '.join([f"{x:.4f}" for x in x_euler[:10]]))
    print("Velocity:", ', '.join([f"{v:.4f}" for v in v_euler[:10]]))
    print("=============================================================================================")
    print("RK4 Method")
    print("Displacement:", ', '.join([f"{x:.4f}" for x in x_rk4[:10]]))
    print("Velocity:", ', '.join([f"{v:.4f}" for v in v_rk4[:10]]))
    print("=============================================================================================")
    print("Exact Solution")
    print("Displacement:", ', '.join([f"{x:.4f}" for x in x_exact[:10]]))
    print("=============================================================================================")
    print("Absolute Error (Displacement):", error_x[:10])
    print("Absolute Error (Velocity):", error_v[:10])
    print()
    print()
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(t, x_euler, label="Euler's Method")
    plt.plot(t, x_rk4, label="RK4 Method")
    plt.plot(t, exact_solution(t), '--', label="Exact Solution")
    plt.xlabel('Time')
    plt.ylabel('Displacement')
    plt.title(f'Harmonic Oscillator: Displacement vs. Time (dt = {dt})')
    plt.legend()
    plt.grid(True)
    plt.show()