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
GENERATIONS = 100_000  # Total mutation attempts

class HolisticCouncil:
    def __init__(self):
        # The Body: 10 Universes initialized together
        self.universes = np.random.uniform(-0.1, 0.1, size=(10, GRID_SIZE, GRID_SIZE))
        self.kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        
        # Stats
        self.best_train_acc = 0.0
        self.best_margin = 0.0

    def get_predictions_and_stats(self, X, y):
        # 1. Flatten all universes: (10, 784)
        flat_weights = self.universes.reshape(10, -1).T
        
        # 2. Normalize (Cosine Fairness)
        norms = np.linalg.norm(flat_weights, axis=0)
        normed_weights = flat_weights / (norms + 1e-9)
        
        # 3. Resonance: (N, 784) @ (784, 10) -> (N, 10)
        scores = np.dot(X, normed_weights)
        
        # 4. Global Decision
        preds = np.argmax(scores, axis=1)
        accuracy = np.mean(preds == y)
        
        # 5. Calculate Margin (Tie-breaker)
        # Margin = Score(Correct_Class) - Score(Best_Incorrect_Class)
        # This acts as a "Loss Function" when accuracy is flat.
        
        # Get score of the true label
        row_idx = np.arange(len(y))
        correct_scores = scores[row_idx, y]
        
        # Mask out the true label to find the best wrong score
        mask = np.ones_like(scores, dtype=bool)
        mask[row_idx, y] = False
        incorrect_scores_matrix = scores[mask].reshape(len(y), 9)
        best_incorrect = np.max(incorrect_scores_matrix, axis=1)
        
        mean_margin = np.mean(correct_scores - best_incorrect)
        
        return accuracy, mean_margin, preds

    def physics_step(self, grid_idx):
        # Extract one organ
        current_grid = self.universes[grid_idx]
        
        # Apply Logic of Necessity
        fluctuation = np.random.uniform(-1, 1, size=current_grid.shape)
        internal_mix = (current_grid * (1.0 - ENTROPY_RATE)) + (fluctuation * ENTROPY_RATE)
        
        neighbor_sum = convolve2d(current_grid, self.kernel, mode='same', boundary='wrap')
        neighbor_avg = neighbor_sum / 4.0
        
        uncertainty_gap = 1.0 - np.abs(internal_mix)
        neighbor_influence = (neighbor_avg * uncertainty_gap) / np.sqrt(2)
        
        candidate = internal_mix + neighbor_influence
        return np.clip(candidate, -1, 1)

    def train(self, X, y, X_test, y_test):
        print(f"--- HOLISTIC COUNCIL (GLOBAL OPTIMIZATION) ---")
        
        # Initial Baseline
        acc, margin, _ = self.get_predictions_and_stats(X, y)
        self.best_train_acc = acc
        self.best_margin = margin
        print(f"Baseline: Train {acc*100:.2f}% | Margin {margin:.4f}")
        
        history_train = []
        history_test = []
        
        # Optimization Loop
        # We assume X is the full training set (Total Recall)
        
        for gen in tqdm(range(GENERATIONS), desc="Evolving"):
            # Round Robin Mutation
            u_idx = gen % 10
            
            # 1. Snapshot old organ
            old_grid = self.universes[u_idx].copy()
            
            # 2. Mutate organ
            self.universes[u_idx] = self.physics_step(u_idx)
            
            # 3. Evaluate Global Body
            new_acc, new_margin, _ = self.get_predictions_and_stats(X, y)
            
            # 4. Selection Logic
            # Priority 1: Accuracy Increase
            # Priority 2: Margin Increase (if Accuracy is equal)
            accepted = False
            
            if new_acc > self.best_train_acc:
                accepted = True
            elif new_acc == self.best_train_acc:
                if new_margin > self.best_margin:
                    accepted = True
            
            if accepted:
                self.best_train_acc = new_acc
                self.best_margin = new_margin
                # Keep mutation (already in self.universes)
            else:
                # Revert
                self.universes[u_idx] = old_grid

            # Logging
            if gen % 100 == 0:
                # Check Test Set (Objective Reality)
                test_acc, _, _ = self.get_predictions_and_stats(X_test, y_test)
                history_train.append(self.best_train_acc)
                history_test.append(test_acc)
                
                tqdm.write(f"Gen {gen}: Train {self.best_train_acc*100:.2f}% | Test {test_acc*100:.2f}%")

        return history_train, history_test

def load_data():
    print("Loading Full MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    X = (X / 127.5) - 1.0 
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    
    # Use a subset for training speed if needed, but Total Recall prefers full data
    # Given the linear algebra speed, 48k training samples is fine per step.
    
    organism = HolisticCouncil()
    h_train, h_test = organism.train(X_train, y_train, X_test, y_test)
    
    print("\n--- FINAL EVALUATION ---")
    final_test_acc, _, _ = organism.get_predictions_and_stats(X_test, y_test)
    print(f"Final Test Accuracy: {final_test_acc*100:.2f}%")
    
    # VISUALIZATION
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f"The Holistic Organism (Acc: {final_test_acc*100:.2f}%)")
    
    for i in range(10):
        ax = axes.flat[i]
        ax.imshow(organism.universes[i], cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title(f"Organ {i}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.plot(h_train, label='Train')
    plt.plot(h_test, label='Test')
    plt.legend()
    plt.title("Evolution of Collective Intelligence")
    plt.show()