"""
Dynamic ODE Solver with Tkinter GUI
Solves:
1. Economic Dispatch for Power Plants (Optimization)
2. Single-Phase Induction Motor Analysis
Features: RK45 and Euler methods, real-time visualization, dynamic sliders
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import math


class ODESolver:
    """Implements ODE solvers: Euler and RK45"""

    @staticmethod
    def euler(f, y0, t_span, dt):
        """
        Euler method for solving ODEs
        f: derivative function dy/dt = f(t, y)
        y0: initial condition
        t_span: (t_start, t_end)
        dt: time step
        """
        t_start, t_end = t_span
        t = np.arange(t_start, t_end + dt, dt)
        y = np.zeros((len(t), len(y0) if hasattr(y0, '__len__') else 1))

        if hasattr(y0, '__len__'):
            y[0] = y0
        else:
            y[0] = y0

        for i in range(1, len(t)):
            y[i] = y[i-1] + dt * f(t[i-1], y[i-1])

        return t, y

    @staticmethod
    def rk45(f, y0, t_span, dt):
        """
        Runge-Kutta 4th order method (RK4) - commonly referred to as RK45 in practice
        f: derivative function dy/dt = f(t, y)
        y0: initial condition
        t_span: (t_start, t_end)
        dt: time step
        """
        t_start, t_end = t_span
        t = np.arange(t_start, t_end + dt, dt)
        y = np.zeros((len(t), len(y0) if hasattr(y0, '__len__') else 1))

        if hasattr(y0, '__len__'):
            y[0] = y0
        else:
            y[0] = y0

        for i in range(1, len(t)):
            k1 = f(t[i-1], y[i-1])
            k2 = f(t[i-1] + dt/2, y[i-1] + dt*k1/2)
            k3 = f(t[i-1] + dt/2, y[i-1] + dt*k2/2)
            k4 = f(t[i-1] + dt, y[i-1] + dt*k3)

            y[i] = y[i-1] + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

        return t, y


class EconomicDispatch:
    """
    Economic Dispatch Solver for Power Plants
    Optimizes power allocation across multiple plants
    """

    def __init__(self):
        # Plant 1: C1 = 0.06*P^2 + 30*P + 10, 30 <= P <= 150
        self.a1, self.b1, self.c1 = 0.06, 30, 10
        self.p1_min, self.p1_max = 30, 150

        # Plant 2: C2 = 0.10*P^2 + 40*P + 15, 20 <= P <= 100
        self.a2, self.b2, self.c2 = 0.10, 40, 15
        self.p2_min, self.p2_max = 20, 100

        # Plant 3: C3 = 0.075*P^2 + 10*P + 20, 50 <= P <= 250
        self.a3, self.b3, self.c3 = 0.075, 10, 20
        self.p3_min, self.p3_max = 50, 250

        self.total_load = 310  # MW

    def incremental_cost(self, P, plant_num):
        """Calculate dC/dP for a given plant"""
        if plant_num == 1:
            return 2 * self.a1 * P + self.b1
        elif plant_num == 2:
            return 2 * self.a2 * P + self.b2
        else:
            return 2 * self.a3 * P + self.b3

    def cost(self, P, plant_num):
        """Calculate total cost for a given plant"""
        if plant_num == 1:
            return self.a1 * P**2 + self.b1 * P + self.c1
        elif plant_num == 2:
            return self.a2 * P**2 + self.b2 * P + self.c2
        else:
            return self.a3 * P**2 + self.b3 * P + self.c3

    def solve_optimal_dispatch(self):
        """
        Solve economic dispatch using lambda iteration method
        Returns: (P1, P2, P3, lambda, total_cost)
        """
        # Initial guess for lambda (Lagrange multiplier)
        lambda_val = 50
        tolerance = 0.001
        max_iterations = 1000

        for iteration in range(max_iterations):
            # Calculate power for each plant given lambda
            # dC/dP = lambda => 2*a*P + b = lambda => P = (lambda - b)/(2*a)
            P1 = (lambda_val - self.b1) / (2 * self.a1)
            P2 = (lambda_val - self.b2) / (2 * self.a2)
            P3 = (lambda_val - self.b3) / (2 * self.a3)

            # Apply constraints
            P1 = max(self.p1_min, min(P1, self.p1_max))
            P2 = max(self.p2_min, min(P2, self.p2_max))
            P3 = max(self.p3_min, min(P3, self.p3_max))

            # Check if power balance is satisfied
            P_total = P1 + P2 + P3
            error = P_total - self.total_load

            if abs(error) < tolerance:
                break

            # Adjust lambda based on error
            # If total power is too high, decrease lambda
            # If total power is too low, increase lambda
            if error > 0:
                lambda_val -= 0.1
            else:
                lambda_val += 0.1

        # Calculate total cost
        total_cost = self.cost(P1, 1) + self.cost(P2, 2) + self.cost(P3, 3)

        return P1, P2, P3, lambda_val, total_cost

    def dynamic_load_change(self, t, y, target_load):
        """
        ODE for dynamic load changes
        y = [P1, P2, P3]
        Models power plants responding to load changes
        """
        P1, P2, P3 = y
        current_total = P1 + P2 + P3

        # Time-varying target load (can be modified)
        self.total_load = target_load

        # Solve for optimal dispatch at current load
        P1_opt, P2_opt, P3_opt, _, _ = self.solve_optimal_dispatch()

        # Rate of change proportional to error (simple controller)
        tau = 5.0  # Time constant
        dP1 = (P1_opt - P1) / tau
        dP2 = (P2_opt - P2) / tau
        dP3 = (P3_opt - P3) / tau

        return np.array([dP1, dP2, dP3])


class InductionMotor:
    """
    Single-Phase Induction Motor Analysis
    Calculates output power, torque, and efficiency
    """

    def __init__(self):
        # Motor parameters
        self.V = 230  # Voltage (V)
        self.f = 50   # Frequency (Hz)
        self.poles = 8
        self.hp = 0.5  # Horsepower

        # Circuit parameters
        self.R1 = 2.0    # Stator resistance (Ω)
        self.X1 = 1.0    # Stator reactance (Ω)
        self.R2_prime = 4.0    # Rotor resistance referred to stator (Ω)
        self.X2_prime = 1.5    # Rotor reactance referred to stator (Ω)
        self.Xm = 180.0  # Magnetizing reactance (Ω)

        # Operating conditions
        self.N = 720  # Speed (rpm)
        self.Pfw = 55  # Friction, windage, and core loss (W)

        # Synchronous speed
        self.Ns = 120 * self.f / self.poles  # rpm

    def calculate_slip(self):
        """Calculate slip"""
        return (self.Ns - self.N) / self.Ns

    def calculate_performance(self):
        """
        Calculate motor performance using equivalent circuit
        Returns: (P_out, T_out, efficiency, slip, current)
        """
        s = self.calculate_slip()

        # Forward and backward impedances for single-phase motor
        # Simplified analysis using forward field only

        # Rotor impedance
        Z_rotor = complex(self.R2_prime/s, self.X2_prime)

        # Parallel combination of Xm and rotor impedance
        Z_parallel = (complex(0, self.Xm) * Z_rotor) / (complex(0, self.Xm) + Z_rotor)

        # Total impedance
        Z_total = complex(self.R1, self.X1) + Z_parallel

        # Stator current
        I1 = self.V / abs(Z_total)

        # Input power
        P_in = self.V * I1 * np.cos(np.angle(Z_total))

        # Air gap power
        P_ag = I1**2 * abs(Z_parallel)**2 * self.R2_prime / (s * abs(Z_total)**2)

        # Rotor copper loss
        P_rcu = s * P_ag

        # Mechanical power developed
        P_mech = P_ag - P_rcu

        # Output power
        P_out = P_mech - self.Pfw

        # Output torque
        omega_m = 2 * np.pi * self.N / 60  # Mechanical angular velocity (rad/s)
        T_out = P_out / omega_m if omega_m > 0 else 0

        # Efficiency
        efficiency = (P_out / P_in * 100) if P_in > 0 else 0

        return P_out, T_out, efficiency, s, I1

    def dynamic_speed_change(self, t, y, target_speed):
        """
        ODE for dynamic speed changes
        y = [N] (speed in rpm)
        Models motor acceleration/deceleration
        """
        N = y[0] if hasattr(y, '__len__') else y
        self.N = N

        # Calculate torque
        _, T_out, _, _, _ = self.calculate_performance()

        # Load torque (simplified constant load)
        T_load = 2.0  # Nm

        # Inertia (estimated for 0.5 hp motor)
        J = 0.01  # kg·m²

        # Angular acceleration
        omega_m = 2 * np.pi * N / 60
        alpha = (T_out - T_load) / J if J > 0 else 0

        # Rate of change of speed
        dN_dt = alpha * 60 / (2 * np.pi)

        return np.array([dN_dt])


class DynamicSimulationGUI:
    """Main GUI application with dynamic simulation"""

    def __init__(self, root):
        self.root = root
        self.root.title("Dynamic ODE Solver - Power Systems & Motors")
        self.root.geometry("1200x800")

        # Make window resizable
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Solver selection
        self.solver_method = tk.StringVar(value="RK45")

        # Create notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        # Initialize solvers
        self.ode_solver = ODESolver()
        self.econ_dispatch = EconomicDispatch()
        self.motor = InductionMotor()

        # Create tabs
        self.create_economic_dispatch_tab()
        self.create_motor_tab()

        # Animation state
        self.animating = False
        self.animation_id = None

    def create_economic_dispatch_tab(self):
        """Create Economic Dispatch tab"""
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="Economic Dispatch")

        # Make tab resizable
        tab1.rowconfigure(1, weight=1)
        tab1.columnconfigure(0, weight=1)

        # Control frame
        control_frame = ttk.LabelFrame(tab1, text="Controls", padding=10)
        control_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)

        # Solver selection
        ttk.Label(control_frame, text="ODE Solver:").grid(row=0, column=0, sticky='w')
        solver_combo = ttk.Combobox(control_frame, textvariable=self.solver_method,
                                     values=["Euler", "RK45"], state='readonly', width=10)
        solver_combo.grid(row=0, column=1, padx=5)

        # Total load slider
        ttk.Label(control_frame, text="Total Load (MW):").grid(row=0, column=2, padx=(20, 0))
        self.load_var = tk.DoubleVar(value=310)
        self.load_slider = ttk.Scale(control_frame, from_=200, to=400,
                                      variable=self.load_var, orient='horizontal', length=200)
        self.load_slider.grid(row=0, column=3, padx=5)
        self.load_label = ttk.Label(control_frame, text="310.0 MW")
        self.load_label.grid(row=0, column=4)

        # Update load display
        self.load_var.trace('w', self.update_load_display)

        # Buttons
        ttk.Button(control_frame, text="Calculate Optimal",
                   command=self.calculate_dispatch).grid(row=0, column=5, padx=5)
        ttk.Button(control_frame, text="Start Dynamic Sim",
                   command=self.start_dispatch_animation).grid(row=0, column=6, padx=5)
        ttk.Button(control_frame, text="Stop",
                   command=self.stop_animation).grid(row=0, column=7, padx=5)

        # Results frame
        results_frame = ttk.LabelFrame(tab1, text="Results", padding=10)
        results_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)

        # Create matplotlib figure
        self.fig1 = Figure(figsize=(10, 6), dpi=100)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=results_frame)
        self.canvas1.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        # Results text
        self.results_text1 = tk.Text(results_frame, height=8, width=40)
        self.results_text1.grid(row=0, column=1, sticky='nsew', padx=5)

    def create_motor_tab(self):
        """Create Induction Motor tab"""
        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text="Induction Motor")

        # Make tab resizable
        tab2.rowconfigure(1, weight=1)
        tab2.columnconfigure(0, weight=1)

        # Control frame
        control_frame = ttk.LabelFrame(tab2, text="Controls", padding=10)
        control_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)

        # Solver selection
        ttk.Label(control_frame, text="ODE Solver:").grid(row=0, column=0, sticky='w')
        solver_combo = ttk.Combobox(control_frame, textvariable=self.solver_method,
                                     values=["Euler", "RK45"], state='readonly', width=10)
        solver_combo.grid(row=0, column=1, padx=5)

        # Speed slider
        ttk.Label(control_frame, text="Target Speed (rpm):").grid(row=0, column=2, padx=(20, 0))
        self.speed_var = tk.DoubleVar(value=720)
        self.speed_slider = ttk.Scale(control_frame, from_=600, to=850,
                                       variable=self.speed_var, orient='horizontal', length=200)
        self.speed_slider.grid(row=0, column=3, padx=5)
        self.speed_label = ttk.Label(control_frame, text="720 rpm")
        self.speed_label.grid(row=0, column=4)

        # Update speed display
        self.speed_var.trace('w', self.update_speed_display)

        # Voltage slider
        ttk.Label(control_frame, text="Voltage (V):").grid(row=1, column=0, sticky='w', pady=5)
        self.voltage_var = tk.DoubleVar(value=230)
        voltage_slider = ttk.Scale(control_frame, from_=180, to=250,
                                    variable=self.voltage_var, orient='horizontal', length=200)
        voltage_slider.grid(row=1, column=1, columnspan=2, padx=5, pady=5)
        self.voltage_label = ttk.Label(control_frame, text="230 V")
        self.voltage_label.grid(row=1, column=3)
        self.voltage_var.trace('w', self.update_voltage_display)

        # Buttons
        ttk.Button(control_frame, text="Calculate Performance",
                   command=self.calculate_motor).grid(row=0, column=5, padx=5)
        ttk.Button(control_frame, text="Start Dynamic Sim",
                   command=self.start_motor_animation).grid(row=0, column=6, padx=5)
        ttk.Button(control_frame, text="Stop",
                   command=self.stop_animation).grid(row=0, column=7, padx=5)

        # Results frame
        results_frame = ttk.LabelFrame(tab2, text="Results", padding=10)
        results_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)

        # Create matplotlib figure
        self.fig2 = Figure(figsize=(10, 6), dpi=100)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=results_frame)
        self.canvas2.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        # Results text
        self.results_text2 = tk.Text(results_frame, height=8, width=40)
        self.results_text2.grid(row=0, column=1, sticky='nsew', padx=5)

    def update_load_display(self, *args):
        """Update load display label"""
        self.load_label.config(text=f"{self.load_var.get():.1f} MW")

    def update_speed_display(self, *args):
        """Update speed display label"""
        self.speed_label.config(text=f"{self.speed_var.get():.0f} rpm")

    def update_voltage_display(self, *args):
        """Update voltage display label"""
        self.voltage_label.config(text=f"{self.voltage_var.get():.0f} V")

    def calculate_dispatch(self):
        """Calculate and display optimal economic dispatch"""
        self.econ_dispatch.total_load = self.load_var.get()
        P1, P2, P3, lambda_val, total_cost = self.econ_dispatch.solve_optimal_dispatch()

        # Update results text
        self.results_text1.delete(1.0, tk.END)
        self.results_text1.insert(tk.END, "=== ECONOMIC DISPATCH RESULTS ===\n\n")
        self.results_text1.insert(tk.END, f"Total Load: {self.econ_dispatch.total_load:.2f} MW\n\n")
        self.results_text1.insert(tk.END, f"Plant 1 Power: {P1:.2f} MW\n")
        self.results_text1.insert(tk.END, f"Plant 2 Power: {P2:.2f} MW\n")
        self.results_text1.insert(tk.END, f"Plant 3 Power: {P3:.2f} MW\n\n")
        self.results_text1.insert(tk.END, f"Total: {P1+P2+P3:.2f} MW\n\n")
        self.results_text1.insert(tk.END, f"Lambda (λ): {lambda_val:.2f} $/MWh\n\n")
        self.results_text1.insert(tk.END, f"Total Cost: ${total_cost:.2f}/hr\n\n")

        # Individual costs
        C1 = self.econ_dispatch.cost(P1, 1)
        C2 = self.econ_dispatch.cost(P2, 2)
        C3 = self.econ_dispatch.cost(P3, 3)
        self.results_text1.insert(tk.END, f"Plant 1 Cost: ${C1:.2f}/hr\n")
        self.results_text1.insert(tk.END, f"Plant 2 Cost: ${C2:.2f}/hr\n")
        self.results_text1.insert(tk.END, f"Plant 3 Cost: ${C3:.2f}/hr\n")

        # Plot static results
        self.fig1.clear()

        # Create subplots
        ax1 = self.fig1.add_subplot(2, 2, 1)
        ax2 = self.fig1.add_subplot(2, 2, 2)
        ax3 = self.fig1.add_subplot(2, 2, 3)
        ax4 = self.fig1.add_subplot(2, 2, 4)

        # Power distribution
        plants = ['Plant 1', 'Plant 2', 'Plant 3']
        powers = [P1, P2, P3]
        ax1.bar(plants, powers, color=['blue', 'green', 'red'])
        ax1.set_ylabel('Power (MW)')
        ax1.set_title('Power Distribution')
        ax1.grid(True, alpha=0.3)

        # Cost distribution
        costs = [C1, C2, C3]
        ax2.bar(plants, costs, color=['blue', 'green', 'red'])
        ax2.set_ylabel('Cost ($/hr)')
        ax2.set_title('Cost Distribution')
        ax2.grid(True, alpha=0.3)

        # Cost curves
        P_range = np.linspace(0, 250, 100)
        C1_curve = self.econ_dispatch.a1 * P_range**2 + self.econ_dispatch.b1 * P_range + self.econ_dispatch.c1
        C2_curve = self.econ_dispatch.a2 * P_range**2 + self.econ_dispatch.b2 * P_range + self.econ_dispatch.c2
        C3_curve = self.econ_dispatch.a3 * P_range**2 + self.econ_dispatch.b3 * P_range + self.econ_dispatch.c3

        ax3.plot(P_range, C1_curve, 'b-', label='Plant 1', linewidth=2)
        ax3.plot(P_range, C2_curve, 'g-', label='Plant 2', linewidth=2)
        ax3.plot(P_range, C3_curve, 'r-', label='Plant 3', linewidth=2)
        ax3.scatter([P1, P2, P3], [C1, C2, C3], s=100, c=['blue', 'green', 'red'], zorder=5)
        ax3.set_xlabel('Power (MW)')
        ax3.set_ylabel('Cost ($/hr)')
        ax3.set_title('Cost Curves')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Incremental cost curves
        IC1 = 2 * self.econ_dispatch.a1 * P_range + self.econ_dispatch.b1
        IC2 = 2 * self.econ_dispatch.a2 * P_range + self.econ_dispatch.b2
        IC3 = 2 * self.econ_dispatch.a3 * P_range + self.econ_dispatch.b3

        ax4.plot(P_range, IC1, 'b-', label='Plant 1', linewidth=2)
        ax4.plot(P_range, IC2, 'g-', label='Plant 2', linewidth=2)
        ax4.plot(P_range, IC3, 'r-', label='Plant 3', linewidth=2)
        ax4.axhline(y=lambda_val, color='k', linestyle='--', label=f'λ = {lambda_val:.2f}')
        ax4.set_xlabel('Power (MW)')
        ax4.set_ylabel('Incremental Cost ($/MWh)')
        ax4.set_title('Incremental Cost Curves')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        self.fig1.tight_layout()
        self.canvas1.draw()

    def calculate_motor(self):
        """Calculate and display motor performance"""
        self.motor.N = self.speed_var.get()
        self.motor.V = self.voltage_var.get()

        P_out, T_out, efficiency, slip, current = self.motor.calculate_performance()

        # Update results text
        self.results_text2.delete(1.0, tk.END)
        self.results_text2.insert(tk.END, "=== MOTOR PERFORMANCE RESULTS ===\n\n")
        self.results_text2.insert(tk.END, f"Voltage: {self.motor.V:.1f} V\n")
        self.results_text2.insert(tk.END, f"Frequency: {self.motor.f} Hz\n")
        self.results_text2.insert(tk.END, f"Poles: {self.motor.poles}\n\n")
        self.results_text2.insert(tk.END, f"Synchronous Speed: {self.motor.Ns:.0f} rpm\n")
        self.results_text2.insert(tk.END, f"Actual Speed: {self.motor.N:.0f} rpm\n")
        self.results_text2.insert(tk.END, f"Slip: {slip:.4f} ({slip*100:.2f}%)\n\n")
        self.results_text2.insert(tk.END, f"Output Power: {P_out:.2f} W\n")
        self.results_text2.insert(tk.END, f"Output Power: {P_out/746:.3f} hp\n\n")
        self.results_text2.insert(tk.END, f"Output Torque: {T_out:.3f} N·m\n\n")
        self.results_text2.insert(tk.END, f"Efficiency: {efficiency:.2f}%\n\n")
        self.results_text2.insert(tk.END, f"Stator Current: {current:.2f} A\n")

        # Plot motor characteristics
        self.fig2.clear()

        # Create subplots
        ax1 = self.fig2.add_subplot(2, 2, 1)
        ax2 = self.fig2.add_subplot(2, 2, 2)
        ax3 = self.fig2.add_subplot(2, 2, 3)
        ax4 = self.fig2.add_subplot(2, 2, 4)

        # Speed vs torque curve
        speed_range = np.linspace(0, self.motor.Ns * 0.99, 50)
        torques = []
        powers = []
        efficiencies = []
        currents = []

        for speed in speed_range:
            self.motor.N = speed
            p, t, eff, s, i = self.motor.calculate_performance()
            torques.append(t)
            powers.append(p)
            efficiencies.append(eff)
            currents.append(i)

        # Restore original speed
        self.motor.N = self.speed_var.get()

        # Torque-Speed curve
        ax1.plot(speed_range, torques, 'b-', linewidth=2)
        ax1.scatter([self.speed_var.get()], [T_out], s=100, c='red', zorder=5)
        ax1.set_xlabel('Speed (rpm)')
        ax1.set_ylabel('Torque (N·m)')
        ax1.set_title('Torque-Speed Characteristic')
        ax1.grid(True, alpha=0.3)

        # Power-Speed curve
        ax2.plot(speed_range, powers, 'g-', linewidth=2)
        ax2.scatter([self.speed_var.get()], [P_out], s=100, c='red', zorder=5)
        ax2.set_xlabel('Speed (rpm)')
        ax2.set_ylabel('Power (W)')
        ax2.set_title('Power-Speed Characteristic')
        ax2.grid(True, alpha=0.3)

        # Efficiency-Speed curve
        ax3.plot(speed_range, efficiencies, 'r-', linewidth=2)
        ax3.scatter([self.speed_var.get()], [efficiency], s=100, c='red', zorder=5)
        ax3.set_xlabel('Speed (rpm)')
        ax3.set_ylabel('Efficiency (%)')
        ax3.set_title('Efficiency-Speed Characteristic')
        ax3.grid(True, alpha=0.3)

        # Current-Speed curve
        ax4.plot(speed_range, currents, 'm-', linewidth=2)
        ax4.scatter([self.speed_var.get()], [current], s=100, c='red', zorder=5)
        ax4.set_xlabel('Speed (rpm)')
        ax4.set_ylabel('Current (A)')
        ax4.set_title('Current-Speed Characteristic')
        ax4.grid(True, alpha=0.3)

        self.fig2.tight_layout()
        self.canvas2.draw()

    def start_dispatch_animation(self):
        """Start dynamic simulation for economic dispatch"""
        self.stop_animation()
        self.animating = True

        # Initial conditions
        P1_init, P2_init, P3_init, _, _ = self.econ_dispatch.solve_optimal_dispatch()
        y0 = np.array([P1_init, P2_init, P3_init])

        # Time parameters
        t_span = (0, 50)  # 50 seconds
        dt = 0.1

        # Define time-varying load
        def load_profile(t):
            # Load varies sinusoidally
            base_load = self.load_var.get()
            return base_load + 30 * np.sin(2 * np.pi * t / 20)

        # Solve ODE
        def f(t, y):
            target = load_profile(t)
            return self.econ_dispatch.dynamic_load_change(t, y, target)

        if self.solver_method.get() == "Euler":
            t, y = self.ode_solver.euler(f, y0, t_span, dt)
        else:
            t, y = self.ode_solver.rk45(f, y0, t_span, dt)

        # Animate
        self.animate_dispatch(t, y, load_profile)

    def animate_dispatch(self, t, y, load_profile):
        """Animate economic dispatch results"""
        self.animation_frame = 0
        self.animation_data = (t, y, load_profile)

        def update_frame():
            if not self.animating or self.animation_frame >= len(t):
                self.stop_animation()
                return

            idx = self.animation_frame
            current_t = t[:idx+1]
            current_y = y[:idx+1]

            # Clear and plot
            self.fig1.clear()

            ax1 = self.fig1.add_subplot(2, 2, 1)
            ax2 = self.fig1.add_subplot(2, 2, 2)
            ax3 = self.fig1.add_subplot(2, 2, 3)
            ax4 = self.fig1.add_subplot(2, 2, 4)

            # Power vs time
            ax1.plot(current_t, current_y[:, 0], 'b-', label='Plant 1', linewidth=2)
            ax1.plot(current_t, current_y[:, 1], 'g-', label='Plant 2', linewidth=2)
            ax1.plot(current_t, current_y[:, 2], 'r-', label='Plant 3', linewidth=2)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Power (MW)')
            ax1.set_title('Power Output vs Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, t[-1])

            # Total power vs load
            total_power = current_y[:, 0] + current_y[:, 1] + current_y[:, 2]
            load_demand = [load_profile(ti) for ti in current_t]
            ax2.plot(current_t, total_power, 'b-', label='Total Generation', linewidth=2)
            ax2.plot(current_t, load_demand, 'r--', label='Load Demand', linewidth=2)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Power (MW)')
            ax2.set_title('Generation vs Demand')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, t[-1])

            # Current power distribution (bar chart)
            if idx > 0:
                plants = ['Plant 1', 'Plant 2', 'Plant 3']
                powers = current_y[idx]
                ax3.bar(plants, powers, color=['blue', 'green', 'red'])
                ax3.set_ylabel('Power (MW)')
                ax3.set_title(f'Current Distribution (t={current_t[idx]:.1f}s)')
                ax3.grid(True, alpha=0.3)

            # Cost over time
            costs = []
            for i in range(len(current_t)):
                C1 = self.econ_dispatch.cost(current_y[i, 0], 1)
                C2 = self.econ_dispatch.cost(current_y[i, 1], 2)
                C3 = self.econ_dispatch.cost(current_y[i, 2], 3)
                costs.append(C1 + C2 + C3)

            ax4.plot(current_t, costs, 'k-', linewidth=2)
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Total Cost ($/hr)')
            ax4.set_title('Total Cost vs Time')
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(0, t[-1])

            self.fig1.tight_layout()
            self.canvas1.draw()

            # Update results text
            if idx > 0:
                P1, P2, P3 = current_y[idx]
                total = P1 + P2 + P3
                cost = costs[idx]

                self.results_text1.delete(1.0, tk.END)
                self.results_text1.insert(tk.END, "=== DYNAMIC SIMULATION ===\n\n")
                self.results_text1.insert(tk.END, f"Time: {current_t[idx]:.1f} s\n\n")
                self.results_text1.insert(tk.END, f"Load Demand: {load_demand[idx]:.2f} MW\n\n")
                self.results_text1.insert(tk.END, f"Plant 1: {P1:.2f} MW\n")
                self.results_text1.insert(tk.END, f"Plant 2: {P2:.2f} MW\n")
                self.results_text1.insert(tk.END, f"Plant 3: {P3:.2f} MW\n\n")
                self.results_text1.insert(tk.END, f"Total Gen: {total:.2f} MW\n\n")
                self.results_text1.insert(tk.END, f"Total Cost: ${cost:.2f}/hr\n\n")
                self.results_text1.insert(tk.END, f"Solver: {self.solver_method.get()}\n")

            self.animation_frame += 2  # Speed up animation
            self.animation_id = self.root.after(50, update_frame)

        update_frame()

    def start_motor_animation(self):
        """Start dynamic simulation for motor"""
        self.stop_animation()
        self.animating = True

        # Initial conditions
        y0 = np.array([600])  # Start from 600 rpm

        # Time parameters
        t_span = (0, 30)  # 30 seconds
        dt = 0.05

        target_speed = self.speed_var.get()

        # Solve ODE
        def f(t, y):
            return self.motor.dynamic_speed_change(t, y, target_speed)

        if self.solver_method.get() == "Euler":
            t, y = self.ode_solver.euler(f, y0, t_span, dt)
        else:
            t, y = self.ode_solver.rk45(f, y0, t_span, dt)

        # Animate
        self.animate_motor(t, y, target_speed)

    def animate_motor(self, t, y, target_speed):
        """Animate motor results"""
        self.animation_frame = 0
        self.animation_data = (t, y, target_speed)

        def update_frame():
            if not self.animating or self.animation_frame >= len(t):
                self.stop_animation()
                return

            idx = self.animation_frame
            current_t = t[:idx+1]
            current_y = y[:idx+1]

            # Calculate motor parameters at each time step
            speeds = current_y.flatten()
            torques = []
            powers = []
            efficiencies = []

            for speed in speeds:
                self.motor.N = speed
                p, tor, eff, s, i = self.motor.calculate_performance()
                torques.append(tor)
                powers.append(p)
                efficiencies.append(eff)

            # Clear and plot
            self.fig2.clear()

            ax1 = self.fig2.add_subplot(2, 2, 1)
            ax2 = self.fig2.add_subplot(2, 2, 2)
            ax3 = self.fig2.add_subplot(2, 2, 3)
            ax4 = self.fig2.add_subplot(2, 2, 4)

            # Speed vs time
            ax1.plot(current_t, speeds, 'b-', linewidth=2)
            ax1.axhline(y=target_speed, color='r', linestyle='--', label=f'Target: {target_speed:.0f} rpm')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Speed (rpm)')
            ax1.set_title('Speed vs Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, t[-1])

            # Torque vs time
            ax2.plot(current_t, torques, 'g-', linewidth=2)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Torque (N·m)')
            ax2.set_title('Torque vs Time')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, t[-1])

            # Power vs time
            ax3.plot(current_t, powers, 'r-', linewidth=2)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Power (W)')
            ax3.set_title('Power vs Time')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, t[-1])

            # Efficiency vs time
            ax4.plot(current_t, efficiencies, 'm-', linewidth=2)
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Efficiency (%)')
            ax4.set_title('Efficiency vs Time')
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(0, t[-1])

            self.fig2.tight_layout()
            self.canvas2.draw()

            # Update results text
            if idx > 0:
                self.motor.N = speeds[idx]
                p_out, t_out, eff, slip, curr = self.motor.calculate_performance()

                self.results_text2.delete(1.0, tk.END)
                self.results_text2.insert(tk.END, "=== DYNAMIC SIMULATION ===\n\n")
                self.results_text2.insert(tk.END, f"Time: {current_t[idx]:.2f} s\n\n")
                self.results_text2.insert(tk.END, f"Current Speed: {speeds[idx]:.1f} rpm\n")
                self.results_text2.insert(tk.END, f"Target Speed: {target_speed:.1f} rpm\n\n")
                self.results_text2.insert(tk.END, f"Slip: {slip:.4f}\n\n")
                self.results_text2.insert(tk.END, f"Torque: {t_out:.3f} N·m\n")
                self.results_text2.insert(tk.END, f"Power: {p_out:.2f} W\n")
                self.results_text2.insert(tk.END, f"Efficiency: {eff:.2f}%\n\n")
                self.results_text2.insert(tk.END, f"Solver: {self.solver_method.get()}\n")

            self.animation_frame += 2  # Speed up animation
            self.animation_id = self.root.after(50, update_frame)

        update_frame()

    def stop_animation(self):
        """Stop animation"""
        self.animating = False
        if self.animation_id:
            self.root.after_cancel(self.animation_id)
            self.animation_id = None


def main():
    """Main entry point"""
    root = tk.Tk()
    app = DynamicSimulationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
