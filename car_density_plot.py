import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
v_max, u_max = 22.2, 0.2  # From part b)
L, T = 2000, 150
display_time = 10
dx, dt = 5, 0.01
Nx, Nt = int(L/dx) + 1, int(T/dt) + 1
x = np.linspace(-2000, 1000, Nx)

saved_frames = {"p1":[],"p2":[],"p5":[]}

# Flux function
def J(u, p):
    return u * v_max * (1 - (u/u_max)**p)

# Lax-Friedrichs solver
def solve_lax_friedrichs(p):
    u = np.where(x<0, u_max, 0)  # Initial condition
    u_new = u.copy()
    frames = [u.copy()]
    for n in range(Nt):
        # u[0], u[-1] = 0, 0  # Boundary conditions
        u[0], u[-1] = u[1], 0  # Boundary conditions
        # u[-1] = 0  # Boundary conditions
        for j in range(1, Nx-1):
            u_new[j] = 0.5 * (u[j+1] + u[j-1]) - (dt/(2*dx)) * (J(u[j+1], p) - J(u[j-1], p))
        u = u_new.copy()
        if n % 10 == 0:  # Save every 10 steps
            frames.append(u.copy())
        # saved_frames["p"+str(p)].append(u.copy())
    return frames

# Compute solutions for p = 1, 2, 5
frames_p1 = solve_lax_friedrichs(1)
frames_p2 = solve_lax_friedrichs(2)
frames_p5 = solve_lax_friedrichs(5)

# Set up figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.suptitle("Traffic Density Evolution for Different p Values")
ax1.set_title("p = 1")
ax2.set_title("p = 2")
ax3.set_title("p = 5")
for ax in (ax1, ax2, ax3):
    ax.set_xlim(-2000, 1000)
    ax.set_ylim(0, u_max * 1.1)
    ax.set_ylabel("Density (cars/m)")
ax3.set_xlabel("Position (m)")

# Initialize lines
line1, = ax1.plot(x, frames_p1[0], color="blue")
line2, = ax2.plot(x, frames_p2[0], color="green")
line3, = ax3.plot(x, frames_p5[0], color="red")

# Animation update function
def update(frame):
    line1.set_ydata(frames_p1[frame])
    line2.set_ydata(frames_p2[frame])
    line3.set_ydata(frames_p5[frame])
    return line1, line2, line3

# Animation
ani = FuncAnimation(fig, update, frames=len(frames_p1), interval=display_time, blit=True)

# Pause functionality
paused = False
def toggle_pause(event):
    global paused
    if event.key == " ":  # Spacebar to toggle pause
        paused = not paused
        if paused:
            print(list(ani.frame_seq)[0]/1501*T, "s") # Print the time of the current frame
            ani.event_source.stop()
        else:
            ani.event_source.start()

fig.canvas.mpl_connect("key_press_event", toggle_pause)
plt.tight_layout()
plt.show()

# p = 1
with open("csv/saved_info_1.csv", "w") as f:
    t_total = 0
    for i, frame in enumerate(saved_frames["p1"]):
        t_total += dt
        f.write(str(round(t_total, 5)))
        f.write(",")
        for n, point in enumerate(frame):
            f.write(str(point))
            if n != len(frame)-1:
                f.write(",")
        f.write(";\n")

# p = 2
with open("csv/saved_info_2.csv", "w") as f:
    t_total = 0
    for i, frame in enumerate(saved_frames["p2"]):
        t_total += dt
        f.write(str(round(t_total, 5)))
        f.write(",")
        for n, point in enumerate(frame):
            f.write(str(point))
            if n != len(frame)-1:
                f.write(",")
        f.write(";\n")

# p = 5
with open("csv/saved_info_5.csv", "w") as f:
    t_total = 0
    for i, frame in enumerate(saved_frames["p5"]):
        t_total += dt
        f.write(str(round(t_total, 5)))
        f.write(",")
        for n, point in enumerate(frame):
            f.write(str(point))
            if n != len(frame)-1:
                f.write(",")
        f.write(";\n")
