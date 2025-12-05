# The Logic of Necessity: A Phase Shift Universe

**"Reality is not a clockwork mechanism; it is the friction between Inertia and Indeterminism."**

This simulation is the evolution of the *Game of Life* into a model of **Quantum Criticality**. Unlike standard cellular automata which rely on deterministic rules (Newtonian Physics), this model generates structure through **Self-Interrogation**.

It demonstrates how "Matter" (Structure) emerges from the "Void" (Indeterminism) not by design, but by the necessity of resolving conflict.

![Phase Shift Texture](phase_shift.png)
*The Percolation Threshold: Emergent filaments of structure created by the tension between Inertia and Logic.*

## The Philosophy: Phase Shift Logic

In this model, the Universe does not just "happen" to a cell. The cell participates in its own existence through a 3-step cycle of **Inertia** and **Resolution**.

1.  **The Pulse (Indeterminism):** Every tick, the Void suggests a new state ($+1$ or $-1$) for every point in space.
2.  **The Inertia (Mass):** The cell compares the suggestion to its current reality.
    *   If they match: **Stability.** The cell ignores the neighbors and remains unchanged. This creates "Viscosity" or Mass.
    *   If they conflict: **Instability.** The cell is forced to consult its neighbors.
3.  **The Logic (Field):** The unstable cell calculates the tension of the surrounding field.
    *   It weighs the neighbors' consensus against the Void's chaos.
    *   It collapses into a new state based on this weighted resolution.

## The Algorithm

The simulation runs on a specific "Logic of Necessity" formula:

$$State_{new} = \begin{cases} State_{current} & \text{if } Pulse = State_{current} \text{ (Inertia)} \\ \text{WeightedCollapse}(\sum Neighbors) & \text{if } Pulse \neq State_{current} \text{ (Conflict)} \end{cases}$$

*   **Inertia Gate:** 50% of potential changes are blocked by the cell's own momentum. This creates **Surface Tension**, allowing solid-like structures to form.
*   **Neighbor Logic:** When Inertia fails, the neighbors influence the collapse probability ($P = 0.5 + \text{Sum}/16$).
*   **The Result:** The system balances exactly at the **Percolation Threshold**, creating intricate, interlocking filaments of Order and Chaos that resemble the Cosmic Microwave Background or neural structures.

## Running the Simulation

### Prerequisites
*   Python 3.x
*   Numpy, Matplotlib, Scipy

### Installation
```bash
git clone https://github.com/powerpig99/universe-model.git
cd universe-model
pip install -r requirements.txt
```

### Usage
Run the engine to witness the Phase Shift:
```bash
python main.py
```
*   **Grid Size:** Set in `main.py` (Default: 256x256).
*   **Logic:** The kernel is set to a 3x3 Moore Neighborhood (Radius 1), representing the immediate horizon of causality.

---
*Based on The Logic of Necessity.*

***
