# DeepXVA: Accelerating Dynamic Initial Margin via Deep Learning

## Overview
Implementation of a **Deep Surrogate** model to accelerate the computation of Dynamic Initial Margin (DIM) under a G2++ interest rate model.

Traditional **Nested Monte-Carlo** approaches are computationally prohibitive for XVA calculations (Inner Loop for risk metrics × Outer Loop for exposure). This project replaces the Inner Loop with a trained Neural Network, achieving massive speedups while maintaining accuracy.

## Results
Benchmark on 1,000 risk scenarios (Inner simulations: 1,000 paths):

| Method | Compute Time | Speedup |
| :--- | :--- | :--- |
| Nested Monte-Carlo | ~45.0s | 1x |
| **Deep Neural Net** | **0.02s** | **~2200x** |

## Structure
- `src/g2_model.py`: Vectorized G2++ engine.
- `src/nested_mc.py`: Baseline implementation (Nested VaR).
- `src/neural_dim.py`: PyTorch architecture for function approximation $f(t, x, y) \to DIM$.
- `benchmark.py`: Runs the comparison.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run benchmark: `python benchmark.py`