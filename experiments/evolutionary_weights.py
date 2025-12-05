import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import zoom
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
DIGIT_NEG = 3  # The Force of -1
DIGIT_POS = 8  # The Force of +1
SCALE_FACTOR = 1 # Keep it 28x28 for speed, or 2 for resolution
GRID_SIZE = 28 * SCALE_FACTOR
ENTROPY_RATE = 0.5 # Maximum Entropy (The Edge of Chaos)

class PhysicalResonator:
    def __init__(self):
        # Initialize the Universe of Weights (The Bridge) as neutral Chaos
        self.weights = np.random.uniform(-0.1, 0.1, size=(GRID_SIZE, GRID_SIZE))
        self.best_accuracy = 0.0
        
        # Physics Kernel
        self.kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    def predict(self, images):
        # RESONATION: Dot Product
        # Flatten images and weights to calculate interaction
        # If Image aligns with Weights, score is high.
        
        # Reshape weights to match flattened image vector
        w_flat = self.weights.flatten()
        
        # Dot product: (N_samples x N_pixels) dot (N_pixels)
        scores = images.dot(w_flat)
        return scores

    def evaluate(self, X, y):
        scores = self.predict(X)
        # Decision Threshold is 0 (The Center)
        predictions = np.where(scores > 0, 1, -1)
        
        accuracy = np.mean(predictions == y)
        return accuracy

    def physics_step(self):
        # 1. MUTATION: The Void proposes a change
        # We don't change everything, just a subtle shift
        fluctuation = np.random.uniform(-1, 1, size=self.weights.shape)
        
        # 2. INERTIA: Mix History with Fluctuation
        # This keeps the weights "Heavy" (Memorable)
        internal_mix = (self.weights * (1.0 - ENTROPY_RATE)) + (fluctuation * ENTROPY_RATE)
        
        # 3. VACUUM INFILLING: Coherence
        # This forces the weights to form contiguous shapes, not static noise.
        # This acts as a regularizer!
        neighbor_sum = convolve2d(self.weights, self.kernel, mode='same', boundary='wrap')
        neighbor_avg = neighbor_sum / 4.0
        
        uncertainty_gap = 1.0 - np.abs(internal_mix)
        neighbor_influence = (neighbor_avg * uncertainty_gap) / np.sqrt(2)
        
        candidate_weights = internal_mix + neighbor_influence
        
        # Clamp to physical limits
        return np.clip(candidate_weights, -1, 1)

    def evolve(self, X_train, y_train, generations=100):
        print(f"--- EVOLUTION STARTING ---")
        print(f"Task: Discriminate {DIGIT_NEG} (-1) vs {DIGIT_POS} (+1)")
        
        history = []
        
        for gen in range(generations):
            # A. Generate a Parallel Universe (Candidate)
            candidate_weights = self.physics_step()
            
            # B. Test the Candidate (Natural Selection)
            # Temporarily swap weights
            old_weights = self.weights.copy()
            self.weights = candidate_weights
            
            acc = self.evaluate(X_train, y_train)
            
            # C. Selection Logic
            if acc >= self.best_accuracy:
                # Survival of the Fittest
                self.best_accuracy = acc
                status = "IMPROVED"
            else:
                # Death: Revert to previous state
                self.weights = old_weights
                status = "REJECTED"
            
            history.append(self.best_accuracy)
            
            if status == "IMPROVED":
                print(f"Gen {gen}: Acc {self.best_accuracy:.4f} [{status}]")
                
        return history

# --- DATA LOADING ---
def load_data():
    print("Loading MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X_raw, y_raw = mnist.data, mnist.target
    
    # Filter for our two warring factions
    mask = (y_raw == str(DIGIT_NEG)) | (y_raw == str(DIGIT_POS))
    X = X_raw[mask]
    y = y_raw[mask]
    
    # Normalize Inputs to (-1, 1) to match ToE
    # 0 -> -1 (Void), 255 -> 1 (Matter)
    X = (X / 127.5) - 1.0
    
    # Labels: Neg -> -1, Pos -> 1
    y = np.where(y == str(DIGIT_POS), 1, -1)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# --- EXECUTION ---
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    
    # Instantiate the Universe
    universe = PhysicalResonator()
    
    # Run Evolution
    acc_history = universe.evolve(X_train, y_train, generations=10000)
    
    # Validation
    test_acc = universe.evaluate(X_test, y_test)
    print(f"--- FINAL RESULT ---")
    print(f"Test Set Accuracy: {test_acc*100:.2f}%")
    
    # --- VISUALIZATION ---
    # Show the "Evolved Structure" of the Weights
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title(f"The Evolved Bridge ({DIGIT_NEG} vs {DIGIT_POS})")
    # Red = Votes for 8, Blue = Votes for 3
    plt.imshow(universe.weights, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.title("Evolution of Accuracy")
    plt.plot(acc_history)
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    
    plt.show()
