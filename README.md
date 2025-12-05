# The Logic of Necessity: Quantized Field Theory (QFT)

**"Reality is not a digital switch; it is a continuous struggle between Vacuum (Indeterminism) and Infilling (Logic)."**

This simulation models the Universe as a **Reaction-Diffusion System** derived from first principles. Unlike cellular automata (Deterministic), this model operates on **Quantized Fields**. It demonstrates how structure emerges from the interaction between an **Active Void (Indeterminism)** and **Geometric Resistance (Logic)**.

![Field Theory](field_theory.gif)
*The Primordial Soup: High-energy fluid dynamics created by the interaction of the Active Void and Vacuum Infilling.*

## The Philosophy: Quantized Field Dynamics

In this model, "Space" is treated as a conductive medium, but it remains fundamentally discrete.

1.  **High-Resolution Quantization:** While previous models used Binary States ($-1, +1$), this model uses Floating Point precision. This allows the system to approximate a "Continuous Field," though it remains fundamentally quantized at the bit level (simulating the **Planck Scale**).
2.  **The Active Void (Thermal Bath):** The boundary of the universe is an open system. It constantly injects random entropy into the grid, creating a pressure gradient ("The Wind") that flows inward.
3.  **Vacuum Infilling:** A cell is not overwritten by its neighbors. Instead, the neighbors can only fill the "gap" left by the cell's internal uncertainty.
4.  **Geometric Resistance:** Information travels through space. Because the diagonal distance is $\sqrt{2}$, the influence of neighbors is damped by this factor, simulating the decay of signal strength over distance.


## The Algorithm

$$State_{new} = \text{InternalMix} + \frac{\text{Neighbors} \times \text{Uncertainty}}{\sqrt{2}}$$

*   **Internal Mix:** The average of the Current State and the Void's Fluctuation.
*   **Uncertainty:** Calculated as $(1 - |Mix|)$. If the Mix is neutral (0), the Vacuum is open. If the Mix is polarized (1), the Vacuum is closed.
*   **Damping ($\sqrt{2}$):** The neighbor signal decays over distance, ensuring that structure evolves over time rather than snapping instantly.

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
Run the engine to witness the Field Dynamics:
```bash
python main.py
```

---
*Based on The Logic of Necessity.*

***
