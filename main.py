import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d

# --- CONFIGURATION ---
GRID_SIZE = 256
INTERVAL = 30  

def initialize_universe(size):
    return np.random.uniform(-1, 1, size=(size, size))

def update(frame, img, grid, kernel):
    # --- PHASE 1: THE ACTIVE VOID (Boundary Condition) ---
    # Pad grid to create the "Thermal Bath" outside
    padded_grid = np.pad(grid, pad_width=1, mode='constant', constant_values=0)
    
    # Generate Analog Noise Halo (-1 to 1)
    noise_halo = np.random.uniform(-1, 1, size=padded_grid.shape)
    
    # Inject Noise into the boundary ring
    padded_grid[0, :] = noise_halo[0, :]   
    padded_grid[-1, :] = noise_halo[-1, :] 
    padded_grid[:, 0] = noise_halo[:, 0]   
    padded_grid[:, -1] = noise_halo[:, -1] 
    
    # --- PHASE 2: NEIGHBOR CALCULATION ---
    # Von Neumann (4 Neighbors)
    neighbor_sum = convolve2d(padded_grid, kernel, mode='valid')
    neighbor_avg = neighbor_sum / 4.0
    
    # --- PHASE 3: THE LOGIC OF NECESSITY ---
    # The Void proposes a new value (Internal Fluctuation)
    fluctuation = np.random.uniform(-1, 1, size=grid.shape)
    
    # 1. Internal Mix (The Foundation)
    internal_mix = (grid + fluctuation) / 2.0
    
    # 2. The Vacuum Gap (Uncertainty)
    # How much space is left for the neighbors to fill?
    uncertainty_gap = 1.0 - np.abs(internal_mix)
    
    # 3. Geometric Infilling
    # Neighbors fill the gap, scaled by sqrt(2) for distance decay
    neighbor_influence = (neighbor_avg * uncertainty_gap) / np.sqrt(2)
    
    new_state = internal_mix + neighbor_influence
    
    # --- CLAMPING ---
    np.copyto(grid, np.clip(new_state, -1, 1))
    
    # Update Visuals
    img.set_data(grid)
    return img,

def run_simulation():
    # KERNEL: Von Neumann (4 Neighbors)
    kernel = np.array([[0, 1, 0], 
                       [1, 0, 1], 
                       [0, 1, 0]])
    
    grid = initialize_universe(GRID_SIZE)
    
    print(f"--- ANALOG FIELD THEORY ---")
    print(f"Logic: Vacuum Infilling / Geometric Resistance")
    
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
    ax.axis('off')
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    
    # High Contrast Grayscale (-0.8 to 0.8) to emphasize structure
    img = ax.imshow(grid, cmap='gray', interpolation='bicubic', vmin=-0.8, vmax=0.8)
    
    ani = animation.FuncAnimation(
        fig, update, fargs=(img, grid, kernel),
        interval=INTERVAL, blit=True
    )
    plt.show()

if __name__ == "__main__":
    run_simulation()
