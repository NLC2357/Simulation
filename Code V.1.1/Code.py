import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from tqdm import tqdm
import os
import pickle

### ====================== PARAMETERS ======================

Lx, Ly = 2e-3, 2e-3 # [m]
Lslot, Lcoflow = 5e-4, 5e-4 # [m^-3]
Vslot, Vcoflow = 1.0, 0.2 # [m.s^-1]
Tslot = 300.0 # [K]

rho = 1.1614 # Fluid density [kg.m^-3]
nu = 15e-6 # Kinematic viscosity [m^2.s^-1]
cp = 1200.0 # Mass heat capacity [J.kg^-1.K^-1]
D = a = nu #  Schmidt and Prandtl numbers [0]
Ta = 10000.0 # Activation temperature [K]
A = 1.1e8 # Arrhenius factor 

i=0; # Index of figure

### ====================== CONSTANTES ======================

# Molar masses [kg.mol^-1]
W_CH4 = 0.01604 ; W_O2  = 0.03200 ; W_H2O = 0.01802 ; W_CO2 = 0.04401

# Enthalpy of formation [J.mol^-1]
h_CH4 = -74.9e3 ; h_O2  = 0.0 ; h_H2O = -241.818e3 ; h_CO2 = -393.52e3

# Stoechiometric coefficients [0]
nu_CH4 = -1 ; nu_O2 = -2 ; nu_H2O = 2 ; nu_CO2 = 1

### ====================== GEOMETRY ======================

nx, ny = 201, 201 # Number of spatial steps
dx = Lx / (nx - 1) # Length of spatial step for x
dy = Ly / (ny - 1) # Length of spatial step for y

Lslot_i = int(Lslot / dx) # Number of cells in slot
Lcoflow_i = int(Lcoflow / dx)

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)  

### ====================== DISPLAY FUNCTION ======================

class Display :
   i=0
   def display(cls, value, title):
        plt.figure(figsize=(10, 6))
        plt.contourf(X, Y, value.T, 50, cmap='viridis')   # .T = transposition -> for matplotlib
        plt.colorbar()
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim([0, Lx])
        plt.ylim([0, Ly])
        plt.savefig(f"Figure {cls.i}")
        plt.show()
        cls.i+=1
   display = classmethod(display)

### ====================== BOUNDARY FUNCTION ======================

@njit(fastmath=True, parallel=True)
def bord_u(u):
    u[0, :] = 0
    u[-1, :] = u[-2, :]
    u[:, 0] = 0 
    u[:, -1] = 0
    return u

@njit(fastmath=True, parallel=True)
def bord_v(v, Lslot_i, Lcoflow_i):
    v[0:Lslot_i, 0] = Vslot
    v[Lslot_i:Lslot_i + Lcoflow_i, 0] = Vcoflow
    v[0:Lslot_i, -1] = -Vslot
    v[Lslot_i:Lslot_i + Lcoflow_i, -1] = -Vcoflow
    v[0, :] = v[1, :]
    v[-1, :] = v[-2, :]
    return v

@njit(fastmath=True, parallel=True)
def bord_p(p):
    p[0, :] = p[1, :]
    p[-1, :] = 0
    p[:, 0] = p[:, 1]
    p[:, -1] = p[:, -2]
    return p

@njit(fastmath=True, parallel=True)
def bord_T(T):
    T[:, 0] = Tslot
    T[:, -1] = Tslot
    return T

@njit(fastmath=True, parallel=True)
def bord_Y(Y_CH4, Y_O2, Y_N2, Y_H2O, Y_CO2):
    Y_CH4[0:Lslot_i, -1] = 1.0
    Y_O2[0:Lslot_i, 0] = 1.0 / (1 + 3.76)
    Y_N2[0:Lslot_i, 0] = 3.76 / (1 + 3.76)
    Y_N2[Lslot_i:Lslot_i+Lcoflow_i, 0] = 1.0
    return Y_CH4, Y_O2, Y_N2, Y_H2O, Y_CO2

### ====================== NUMERICAL METHODS ======================

@njit(fastmath=True, parallel=True)
def advection(u, v, phi, dx, dy):
    adv = np.zeros_like(phi)
    ux_d = (phi[2:, 1:-1] - phi[1:-1, 1:-1]) / dx
    ux_g = (phi[1:-1, 1:-1] - phi[:-2, 1:-1]) / dx
    term_x = ((u[1:-1, 1:-1] - np.abs(u[1:-1, 1:-1])) * ux_d +
              (u[1:-1, 1:-1] + np.abs(u[1:-1, 1:-1])) * ux_g) / 2

    uy_d = (phi[1:-1, 2:] - phi[1:-1, 1:-1]) / dy
    uy_g = (phi[1:-1, 1:-1] - phi[1:-1, :-2]) / dy
    term_y = ((v[1:-1, 1:-1] - np.abs(v[1:-1, 1:-1])) * uy_d +
              (v[1:-1, 1:-1] + np.abs(v[1:-1, 1:-1])) * uy_g) / 2
    adv[1:-1, 1:-1] = term_x + term_y
    return adv

@njit(fastmath=True, parallel=True)
def diffusion(phi, dx, dy):
    lap = np.zeros_like(phi)
    lap[1:-1, 1:-1] = ((phi[2:, 1:-1] - 2*phi[1:-1, 1:-1] + phi[:-2, 1:-1]) / dx**2 +
                       (phi[1:-1, 2:] - 2*phi[1:-1, 1:-1] + phi[1:-1, :-2]) / dy**2)
    return lap

@njit(fastmath=True, parallel=True)
def pressure_poisson(p, u_star, v_star, dx, dy, dt, rho, omega=1.5, tol=1e-5, max_iter=1000):
    p_old = p.copy()
    s = np.zeros_like(p)
    s[1:-1, 1:-1] = (rho / dt) * (
        (u_star[2:, 1:-1] - u_star[:-2, 1:-1]) / (2*dx) +
        (v_star[1:-1, 2:] - v_star[1:-1, :-2]) / (2*dy)
    )
    for _ in range(max_iter):
        p_old = p.copy()
        for i in range(1, p.shape[0]-1):
            for j in range(1, p.shape[1]-1):
                p[i,j] = (1-omega)*p[i,j] + omega*((p[i+1,j] + p[i-1,j] + p[i,j+1] + p[i,j-1])/4 - s[i,j]*dx**2/4)
        bord_p(p)
        if np.max(np.abs(p - p_old)) < tol:
            break
    return p

@njit(fastmath=True, parallel=True)
def reaction_rate(Y_CH4, Y_O2, T, rho):
    conc_CH4 = rho * Y_CH4 / W_CH4
    conc_O2  = rho * Y_O2  / W_O2
    return A * conc_CH4 * conc_O2**2 * np.exp(-Ta / T)

### ====================== SIMULATION ENGINE ======================

def run_simulation(nt, nt_chim): 
    
    # Initialization
    u = np.zeros((nx, ny)) # Horizontal velocity field (at 0 m.s^-1)
    v = np.zeros((nx, ny)) # Vertical velocity field (at 0 m.s^-1)
    
    p = np.zeros((nx, ny)) # Pressure field (at 0 Pa)
    T = np.full((nx, ny), 300.0) # Temperature field (at 300 K)

    Y_CH4 = np.zeros((nx, ny)) # Mass fraction fields (at 0)
    Y_O2  = np.zeros((nx, ny))
    Y_N2  = np.zeros((nx, ny))
    Y_H2O = np.zeros((nx, ny))
    Y_CO2 = np.zeros((nx, ny))

    U_old = u.copy()
    V_old = v.copy()
    u_star = u.copy()
    v_star = v.copy()

    # Initial boundary conditions
    bord_T(T)
    bord_Y(Y_CH4, Y_O2, Y_N2, Y_H2O, Y_CO2)

    # Ignition strip parameters
    delta_ignite = 0.5e-3 # [m]
    half_band = int((delta_ignite / 2) / dy)
    center = ny // 2
    T[:, center - half_band:center + half_band] = 1000.0 # [K]

    ### ADVECTION + DIFFUSION
    for it in tqdm(range(nt)):
        
        max_vel = max(np.max(np.abs(u)), np.max(np.abs(v)), Vslot)
        dt = min(0.25 * min(dx, dy) / max_vel, 0.2 * min(dx, dy)**2 / nu)
        dt_chem = min(1e-7, dt)

        u_star[:] = u
        v_star[:] = v

        u_star[1:-1,1:-1] += dt * (-advection(u, v, u, dx, dy)[1:-1,1:-1] + nu * diffusion(u, dx, dy)[1:-1,1:-1])
        v_star[1:-1,1:-1] += dt * (-advection(u, v, v, dx, dy)[1:-1,1:-1] + nu * diffusion(v, dx, dy)[1:-1,1:-1])

        p = pressure_poisson(p, u_star, v_star, dx, dy, dt, rho)

        u[1:-1,1:-1] = u_star[1:-1,1:-1] - (dt / rho) * (p[2:,1:-1] - p[:-2,1:-1]) / (2*dx)
        v[1:-1,1:-1] = v_star[1:-1,1:-1] - (dt / rho) * (p[1:-1,2:] - p[1:-1,:-2]) / (2*dy)

        u = bord_u(u)
        v = bord_v(v, Lslot_i, Lcoflow_i)

        # Mass fraction - transport update
        for field in [Y_CH4, Y_O2, Y_N2, Y_H2O, Y_CO2, T]:
            field[1:-1,1:-1] += dt * (D * diffusion(field, dx, dy)[1:-1,1:-1] - advection(u, v, field, dx, dy)[1:-1,1:-1])

        # Mass sign correction
        Y_CH4 = np.maximum(Y_CH4, 0)
        Y_O2  = np.maximum(Y_O2, 0)
        Y_H2O = np.maximum(Y_H2O, 0)
        Y_CO2 = np.maximum(Y_CO2, 0)
        Y_N2  = np.maximum(Y_N2, 0)

        # Boundary conditions injection
        bord_T(T)
        bord_Y(Y_CH4, Y_O2, Y_N2, Y_H2O, Y_CO2)

        ### CHIMIE
        for i in range(nt_chim):

            # Reaction rate
            Q = reaction_rate(Y_CH4[1:-1,1:-1], Y_O2[1:-1,1:-1], T[1:-1,1:-1], rho)

            # Mass fraction - chemical update
            Y_CH4[1:-1,1:-1] += dt_chem * (W_CH4 * nu_CH4 * Q / rho)
            Y_O2[1:-1,1:-1]  += dt_chem * (W_O2  * nu_O2  * Q / rho)
            Y_H2O[1:-1,1:-1] += dt_chem * (W_H2O * nu_H2O * Q / rho)
            Y_CO2[1:-1,1:-1] += dt_chem * (W_CO2 * nu_CO2 * Q / rho)

            # Heat release
            omega_T = -(h_CH4 * nu_CH4 * Q + h_O2 * nu_O2 * Q + h_H2O * nu_H2O * Q + h_CO2 * nu_CO2 * Q)

            # Temperature update
            T[1:-1,1:-1] += dt_chem * omega_T / (rho * cp)

            # Ignition strip update
            T[:, center - half_band:center + half_band] = np.maximum(T[:, center - half_band:center + half_band], 1000.0)

            # Mass rounding correction
            sumY = (Y_CH4[1:-1,1:-1] + Y_O2[1:-1,1:-1] + Y_H2O[1:-1,1:-1] + Y_CO2[1:-1,1:-1]) 
            Y_N2[1:-1,1:-1] = np.maximum(1.0 - sumY, 0.0)

        ### CONVERGENCE
        if np.max(np.abs(u - U_old)) < 1e-6 and np.max(np.abs(v - V_old)) < 1e-6:
            print(f"Convergence reached at iteration {it}")
            break
       
        U_old[:] = u
        V_old[:] = v

    return u, v, p, T, Y_CH4, Y_O2, Y_N2, Y_H2O, Y_CO2

### ====================== RUN ======================

if __name__ == "__main__":
    
    nt = int(input("Number of time steps : ")) # min 5000 recommanded
    nt_chim = int(input("Number of chemical time steps : "))  # min 100 recommanded

    u, v, p, T, Y_CH4, Y_O2, Y_N2, Y_H2O, Y_CO2 = run_simulation(nt, nt_chim)

 ### ====================== PLOTS ======================

    Display.display(T, 'Temperature')
    Display.display(Y_CH4, 'CH4 Concentration')
    Display.display(Y_O2, 'O2 Concentration')
    Display.display(Y_N2, 'N2 Concentration')
    Display.display(Y_H2O, 'H2O Concentration')
    Display.display(Y_CO2, 'CO2 Concentration')

    print(f"T_max = {np.max(T):.0f} K")
    print(f"Y_H2O_max = {np.max(Y_H2O):.4f}")


### ====================== SAVE ======================

with open("parameters.txt", "w") as parameter_file:
    parameters = [("nt = ", nt), ("nt_chim = ", nt_chim), ("Lx = ", Lx), ("Ly = ", Ly), ("Lslot = ", Lslot), ("Lcoflow = ", Lcoflow), ("Vslot = ", Vslot), ("Vcoflow = ", Vcoflow), 
                  ("Tslot = ", Tslot), ("rho = ", rho), ("nu = ", nu), ("cp = ", cp), ("D = ", D),("Ta = ", Ta), ("A = ", A), ("nx = ", nx), ("ny = ", ny), ("dx = ", dx), 
                  ("dy = ", dy)
                 ]

    for param in parameters:
        line = f"{param[0]}{param[1]}\n"
        parameter_file.write(line)

    print("Fichier 'parameters.txt' créé avec succès !")