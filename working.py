import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
M, m, L, a, I_v, F_0, g = 5, 1, 1, 0.25, 0.01, 200, 9.81
I_o = M * ((L**2)/12 + (L**2 - 4*L*a + 4*a**2)/4)

# Initial conditions
r_0 = 0.3 * L
phi_0 = -np.pi / 32

# System of differential equations
def system(t, y):
    r, v, phi, omega = y
    F = F_0
    drdt = v
    dvdt = -g * np.sin(phi) + r * omega**2
    dphidt = omega
    domegadt = (np.sin(phi) * F * a - np.sin(phi) * np.cos(phi) * M * g * (L / 2 - a) - 
                2 * m * r * v * omega * np.sin(phi) - m*r**2*omega**2*np.cos(phi) + m*r*np.cos(phi)*dvdt) / ((I_o + I_v + m*r**2)*np.sin(phi))

    return [drdt, dvdt, dphidt, domegadt]

# Event function to stop the solver when r > L - a
def event1(t, y):
    r = y[0] 
    return r - (L - a)
event1.terminal = True

# Event function to stop the solver when the normal force becomes negative ( Not needed though since event 1 happens before)
def event2(t, y):
    r, v, phi, omega = y
    drdt, dvdt, dphidt, domegadt = system(t, y)  # get the right variables
    normal_force = m * (g * np.cos(phi) + 2 * drdt * dphidt + r * domegadt) # calculated before
    return normal_force
event2.terminal = True


# Time span for the solution
t_span = [0, 0.5]

# Initial conditions vector, distance, velocity, angle, angular velocity
y_0 = [r_0, 0, phi_0, 0]

# Time points and how many "stops"
t_eval = np.linspace(t_span[0], t_span[1], 100)

# Solve the system of differential equations with the event function which if true stops the solver
sol = solve_ivp(system, t_span, y_0, events=[event1, event2], t_eval=t_eval)

# Define the labels for the plots
labels = [('r (m)', 'r as a function of time'), 
          ("r' (m/s)", "r' as a function of time"), 
          ('φ (rad)', 'φ as a function of time'), 
          ("φ' (rad/s)", "φ' as a function of time")]

plt.figure(figsize=(12,6))

# Loop over the solutions and create plots
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(sol.t, sol.y[i])
    plt.xlabel('Time (s)')
    plt.ylabel(labels[i][0])
    plt.title(labels[i][1])

# Plot the cart's motion in the XY plane. Sol.y[2] represents the value phi
x = sol.y[0] * np.cos(sol.y[2])
y = sol.y[0] * np.sin(sol.y[2])

plt.figure(figsize=(6,6))
plt.plot(x, y)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Carts motion in the XY plane')
plt.axis('equal')
plt.show()

if sol.t_events[0].size > 0:
    print(f"The cart went off the edge at t = {sol.t_events[0][0]} s")
if sol.t_events[1].size > 0:
    print(f"The normal force became negative at t = {sol.t_events[1][0]} s")
    
# Hastighetsvektor, y[1] represents the velocity, y[3] represents the angular velocity
v_x = sol.y[1] * np.cos(sol.y[2]) - sol.y[0] * sol.y[3] * np.sin(sol.y[2])  # Velocity "längs stången" projicerat on the x-axis, - Tangential velocity projicerat on the x-axis 
v_y = sol.y[1] * np.sin(sol.y[2]) + sol.y[0] * sol.y[3] * np.cos(sol.y[2])  # Velocity "längs stången" projicerat on the y-axis, + Tangential velocity projicerat on the y-axis



# Initial conditions for the projectile motion
x_0 = x[-1] # The carts x position at the last "tidspunkt" from the solutions above
y_0 = y[-1] # The carts y position at the last "tidspunk" from the solutions above
v_x_0 = v_x[-1] # The carts x velocity at the last "tidspunkt" from the solutions above
v_y_0 = v_y[-1] # The carts y velocity at the last "tidspunkt" from the solutions above

# Time step
dt = 0.001

# Projectile motion until it hits the ground (y=0)
while y_0 > 0:
    x_0 += v_x_0 * dt #updates the x position 
    y_0 += v_y_0 * dt #updates the y position
    v_y_0 -= g * dt #simulates the carts velocity in the y direction

print(f"The landing spot is at x = {x_0} m")
