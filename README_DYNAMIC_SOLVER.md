# Dynamic ODE Solver with Tkinter GUI

A comprehensive Python application for solving and visualizing power system and electrical machine problems using real-time ODE solvers.

## Features

### 1. **ODE Solvers**
- **Euler Method**: First-order numerical integration
- **RK45 (Runge-Kutta 4th Order)**: Higher accuracy numerical integration
- Real-time solver selection and comparison

### 2. **Economic Dispatch Optimization**
Solves optimal power allocation for three plants:
- **Plant 1**: C₁ = 0.06P² + 30P + 10 (30 ≤ P ≤ 150 MW)
- **Plant 2**: C₂ = 0.10P² + 40P + 15 (20 ≤ P ≤ 100 MW)
- **Plant 3**: C₃ = 0.075P² + 10P + 20 (50 ≤ P ≤ 250 MW)

Features:
- Lambda iteration method for optimal dispatch
- Dynamic load changes with real-time response
- Cost curve visualization
- Incremental cost analysis

### 3. **Single-Phase Induction Motor Analysis**
Analyzes a 230V, 50Hz, 8-pole, 1/2 hp motor:
- Output power and torque calculations
- Efficiency analysis
- Dynamic speed response
- Characteristic curves (torque-speed, power-speed, efficiency-speed)

### 4. **Interactive Controls**
- **Sliders** for real-time parameter adjustment:
  - Total load (MW) for economic dispatch
  - Target speed (rpm) for motor
  - Voltage (V) for motor
- Dynamic simulation with animation
- Automatic window resizing

### 5. **Visualization**
- Real-time dynamic graphs
- Multiple plot types:
  - Power distribution
  - Cost analysis
  - Time-series evolution
  - Motor characteristics
  - Comparative analysis

## Requirements

```bash
pip install numpy matplotlib
```

Python 3.x with tkinter (usually included by default)

## Usage

### Run the Application

```bash
python3 dynamic_ode_solver.py
```

### Economic Dispatch Tab

1. **Static Analysis**:
   - Adjust "Total Load" slider (200-400 MW)
   - Click "Calculate Optimal" to solve
   - View power distribution, costs, and curves

2. **Dynamic Simulation**:
   - Select ODE solver (Euler or RK45)
   - Click "Start Dynamic Sim" to see load changes over time
   - Observe how plants respond to varying demand
   - Click "Stop" to halt animation

### Induction Motor Tab

1. **Static Analysis**:
   - Adjust "Target Speed" slider (600-850 rpm)
   - Adjust "Voltage" slider (180-250 V)
   - Click "Calculate Performance" to analyze
   - View torque, power, efficiency, and current

2. **Dynamic Simulation**:
   - Select ODE solver (Euler or RK45)
   - Set target speed
   - Click "Start Dynamic Sim" to see transient response
   - Observe acceleration/deceleration behavior
   - Click "Stop" to halt animation

## Problem Solutions

### Problem 1: Economic Dispatch

**Given**:
- Total system load: 310 MW
- Three plants with quadratic cost functions

**Solution Method**:
- Uses lambda iteration (Lagrange multipliers)
- Equal incremental cost principle: dC₁/dP₁ = dC₂/dP₂ = dC₃/dP₃ = λ
- Respects power generation limits

**Expected Results**:
- Plant 1: ~128 MW
- Plant 2: ~32 MW
- Plant 3: ~150 MW
- Total Cost: ~$13,000/hr
- Lambda (λ): ~45 $/MWh

### Problem 2: Induction Motor

**Given**:
- 230V, 50Hz, 8-pole, 1/2 hp single-phase motor
- R₁ = 2Ω, X₁ = 1Ω, R₂' = 4Ω, X₂' = 1.5Ω, Xm = 180Ω
- Operating speed: 720 rpm
- Losses: 55W

**Solution Method**:
- Equivalent circuit analysis
- Slip calculation: s = (Ns - N)/Ns
- Power flow: Input → Air gap → Mechanical → Output
- Torque: T = P_out/ω

**Expected Results**:
- Synchronous speed: 750 rpm
- Slip: 0.04 (4%)
- Output power: ~250-300W
- Output torque: ~3-4 N·m
- Efficiency: ~70-75%

## Features Explained

### Automatic Resizing
- Window and all widgets resize proportionally
- Grid layout with weight configuration
- Maintains aspect ratio for plots

### ODE Solver Comparison
- **Euler**: Fast but less accurate, good for smooth systems
- **RK45**: More accurate but slower, better for complex dynamics
- Compare both methods visually in real-time

### Dynamic Simulation
- Models transient behavior
- Economic dispatch: Load changes trigger plant adjustments
- Motor: Speed changes show acceleration dynamics
- Time constants and delays visible

### Real-Time Visualization
- Live updating plots during simulation
- Multiple synchronized graphs
- Time-series and static analysis views

## Technical Details

### ODE Formulations

**Economic Dispatch**:
```
dP₁/dt = (P₁_optimal - P₁)/τ
dP₂/dt = (P₂_optimal - P₂)/τ
dP₃/dt = (P₃_optimal - P₃)/τ
```
Where τ is the time constant (5s) representing plant response time.

**Induction Motor**:
```
dN/dt = α × (60/2π)
α = (T_motor - T_load)/J
```
Where J is moment of inertia, α is angular acceleration.

### Code Structure

- `ODESolver`: Implements Euler and RK45 methods
- `EconomicDispatch`: Optimization and dynamics
- `InductionMotor`: Motor calculations and dynamics
- `DynamicSimulationGUI`: Main application interface

## Troubleshooting

### Issue: Window doesn't resize properly
**Solution**: Ensure grid weights are set correctly. Restart application.

### Issue: Animation is too slow/fast
**Solution**: Adjust `self.animation_frame += 2` in the animate methods (line ~720 and ~820).

### Issue: Plots are cluttered
**Solution**: Resize window larger. Adjust DPI in Figure creation.

### Issue: Motor calculations seem incorrect
**Solution**: Check parameter values match the problem. Verify voltage and speed sliders.

## Extensions

Potential enhancements:
1. Add more ODE solvers (RK23, adaptive step size)
2. Save/load simulation results
3. Export plots to files
4. Add more motor types (3-phase, synchronous)
5. Include transmission line losses
6. Real-time data input from files

## References

- Economic Dispatch: Equal Incremental Cost Method
- Induction Motor: Equivalent Circuit Model
- ODE Solvers: Numerical Methods for Engineers
- Tkinter: Python GUI Programming

## Author

Created for advanced power systems and electrical machines analysis.

## License

Educational use only.
