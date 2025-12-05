import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.signal import convolve2d
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- CONFIGURATION ---
GRID_SIZE = 28
ENTROPY_RATE = 0.5
GENERATIONS = 10000 

class PhysicalNode:
    def __init__(self, target_digit):
        self.target_digit = target_digit
        self.grid = np.random.uniform(-0.1, 0.1, size=(GRID_SIZE, GRID_SIZE))
        self.kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        self.best_acc = 0.0
        self.max_signal = 1.0 

    def physics_step(self):
        current_grid = self.grid
        fluctuation = np.random.uniform(-1, 1, size=current_grid.shape)
        internal_mix = (current_grid * (1.0 - ENTROPY_RATE)) + (fluctuation * ENTROPY_RATE)
        
        neighbor_sum = convolve2d(current_grid, self.kernel, mode='same', boundary='wrap')
        neighbor_avg = neighbor_sum / 4.0
        
        uncertainty_gap = 1.0 - np.abs(internal_mix)
        neighbor_influence = (neighbor_avg * uncertainty_gap) / np.sqrt(2)
        
        candidate = internal_mix + neighbor_influence
        return np.clip(candidate, -1, 1)

    def evolve_balanced_recall(self, X_target, X_rest):
        candidate = self.physics_step()
        w_flat = candidate.flatten()
        
        scores_t = np.dot(X_target, w_flat)
        scores_r = np.dot(X_rest, w_flat)
        
        # Balanced Accuracy (Threshold 0)
        acc_t = np.mean(scores_t > 0)
        acc_r = np.mean(scores_r < 0)
        balanced_acc = (acc_t + acc_r) / 2.0
        
        if balanced_acc >= self.best_acc:
            self.grid = candidate
            self.best_acc = balanced_acc
            
            # Calibration: 99th percentile peak
            peak_t = np.percentile(np.abs(scores_t), 99)
            peak_r = np.percentile(np.abs(scores_r), 99)
            self.max_signal = max(peak_t, peak_r, 1e-9)
            return True
        return False

class CouncilOfTruth:
    def __init__(self):
        self.nodes = {d: PhysicalNode(d) for d in range(10)}
        self.data_map = {}
        # We track Training Accuracy to decide when to lock
        self.best_train_acc = 0.0
        self.best_nodes_state = None

    def train(self, X, y, X_test, y_test):
        self.data_map = {d: X[y == d] for d in range(10)}
        
        print(f"--- COUNCIL OF TRUTH (HONEST MODE) ---")
        print("Locking state based on TRAINING Accuracy only.")
        
        history_train = []
        history_test = []
        
        # Pre-select validation subsets for speed
        idx_train = np.random.choice(len(X), 10000, replace=False) # Large sample for stability
        idx_test = np.random.choice(len(X_test), 10000, replace=False)
        
        X_val_train = X[idx_train]
        y_val_train = y[idx_train]
        X_val_test = X_test[idx_test]
        y_val_test = y_test[idx_test]
        
        for gen in tqdm(range(GENERATIONS), desc="Evolving"):
            # Evolve each universe
            for d in range(10):
                X_rest_local = np.concatenate([self.data_map[r] for r in range(10) if r != d])
                self.nodes[d].evolve_balanced_recall(self.data_map[d], X_rest_local)

            # Global Check every 100 generations
            if gen % 100 == 0:
                # 1. Measure Global Performance
                preds_train = self.predict(X_val_train)
                acc_train = np.mean(preds_train == y_val_train)
                
                preds_test = self.predict(X_val_test)
                acc_test = np.mean(preds_test == y_val_test)
                
                history_train.append(acc_train)
                history_test.append(acc_test)
                
                log_msg = (f"Gen {gen}: Train {acc_train*100:.2f}% | Test {acc_test*100:.2f}%")
                
                # 2. Honest Snapshot Strategy
                # We only save if TRAINING accuracy improves.
                # We do not look at Test accuracy to make this decision.
                if acc_train > self.best_train_acc:
                    self.best_train_acc = acc_train
                    self.best_nodes_state = copy.deepcopy(self.nodes)
                    log_msg += " [NEW TRAIN BEST]"
                
                tqdm.write(log_msg)

        # Restore the Golden Age of Training
        print(f"\nRestoring Best Training State (Train Acc: {self.best_train_acc*100:.2f}%)")
        self.nodes = self.best_nodes_state
        
        return history_train, history_test

    def predict(self, images):
        scores = np.zeros((len(images), 10))
        for d in range(10):
            node = self.nodes[d]
            w_flat = node.grid.flatten()
            raw_scores = np.dot(images, w_flat)
            scores[:, d] = raw_scores / node.max_signal
        return np.argmax(scores, axis=1)

def load_data():
    print("Loading Full MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    X = (X / 127.5) - 1.0 
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    
    council = CouncilOfTruth()
    h_train, h_test = council.train(X_train, y_train, X_test, y_test)
    
    print("\n--- FINAL EVALUATION (HONEST LOCK) ---")
    preds_train = council.predict(X_train)
    final_train = np.mean(preds_train == y_train)
    
    preds_test = council.predict(X_test)
    final_test = np.mean(preds_test == y_test)
    
    print(f"Final Train Accuracy: {final_train*100:.2f}%")
    print(f"Final Test Accuracy:  {final_test*100:.2f}%")
    
    # VISUALIZATION 1: The Ghosts
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f"The Calibrated Council (Final Test Acc: {final_test*100:.2f}%)")
    
    for i in range(10):
        ax = axes.flat[i]
        ax.imshow(council.nodes[i].grid, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title(f"Uni {i}\nIndiv Acc: {council.nodes[i].best_acc:.2f}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # VISUALIZATION 2: The Tracking
    plt.figure(figsize=(10, 6))
    plt.plot(h_train, label='Training Accuracy', linewidth=2)
    plt.plot(h_test, label='Test Accuracy', linestyle='--', linewidth=2)
    plt.xlabel('Generations (x100)')
    plt.ylabel('Global Accuracy')
    plt.title('Subjective vs Objective Truth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()