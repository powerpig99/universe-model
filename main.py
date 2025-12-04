import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d

# --- CONFIGURATION ---
GRID_SIZE = 1024
RADIUS = 3
KERNEL_WIDTH = 2 * RADIUS + 1
# The Uncertainty Principle: Divisor is slightly > range to prevent P=0 or P=1.
DIVISOR = (2 * (KERNEL_WIDTH ** 2)) + 1e-5 
INTERVAL = 50

def initialize_universe(size):
    return np.random.choice([-1, 1], size=(size, size))

def update(frame, img, grid, kernel, divisor):
    # 1. THE ACTIVE VOID (The Heat Bath)
    # Pad the universe with a random "Quantum Foam" (+/- 1)
    padded_grid = np.pad(grid, pad_width=RADIUS, mode='constant', constant_values=0)
    noise_halo = np.random.choice([-1, 1], size=padded_grid.shape)
    
    # Apply noise only to the boundary (The Event Horizon)
    padded_grid[:RADIUS, :] = noise_halo[:RADIUS, :]
    padded_grid[-RADIUS:, :] = noise_halo[-RADIUS:, :]
    padded_grid[:, :RADIUS] = noise_halo[:, :RADIUS]
    padded_grid[:, -RADIUS:] = noise_halo[:, -RADIUS:]
    
    # 2. LOGIC (Convolution)
    neighbor_sum = convolve2d(padded_grid, kernel, mode='valid')
    
    # 3. PROBABILITY (The Field)
    probability = 0.5 + (neighbor_sum / divisor)
    
    # 4. COLLAPSE (The State)
    random_roll = np.random.random(grid.shape)
    np.copyto(grid, np.where(random_roll < probability, 1, -1))
    
    img.set_data(grid)
    return img,

def run_simulation():
    kernel = np.ones((KERNEL_WIDTH, KERNEL_WIDTH))
    grid = initialize_universe(GRID_SIZE)
    print(f"--- UNIVERSE STARTED ---\nSize: {GRID_SIZE}x{GRID_SIZE}\nBoundary: Active Void (0 Bias)")
    
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
    ax.axis('off')
    img = ax.imshow(grid, cmap='Greys', interpolation='nearest', vmin=-1, vmax=1)
    
    ani = animation.FuncAnimation(
        fig, update, fargs=(img, grid, kernel, DIVISOR),
        interval=INTERVAL, blit=True
    )
    plt.show()

if __name__ == "__main__":
    run_simulation()