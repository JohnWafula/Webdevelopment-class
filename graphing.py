import numpy as np
import matplotlib.pyplot as plt

# Constants
V = 1
C = 1
R = 2

# Time span
t = np.arange(0, 10, 0.01)  # Adjust the time span as needed

# Initial conditions
Q0 = 0
Q0_prime = 1/2

# Different values of L
L_values = [0, 1, 10]

# Initialize a dictionary to store Q(t) for different L values
Q_values = {}

# Solve for Q(t) for each L
for L in L_values:
    # Define the differential equation
    def equation(t, Q):
        return [Q[1], (1 / (L * C)) * (V - R * Q[1] - Q[0])]

    # Numerically solve the differential equation using Euler's method
    Q_solution = np.zeros((len(t), 2))
    Q_solution[0] = [Q0, Q0_prime]
    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        Q_solution[i] = Q_solution[i - 1] + h * np.array(equation(t[i - 1], Q_solution[i - 1]))

    # Store the Q(t) values
    Q_values[f'L = {L}'] = Q_solution[:, 0]

# Plot Q(t) for different L values
plt.figure(figsize=(10, 6))
for label, Q in Q_values.items():
    plt.semilogy(t, Q, label=label)

plt.xlabel('Time (t)')
plt.ylabel('Charge (Q(t))')
plt.title('Charge on Capacitor vs. Time')
plt.legend()
plt.grid(True)
plt.show()
