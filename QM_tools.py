import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from matplotlib.animation import FuncAnimation, PillowWriter
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations
from matplotlib.animation import FuncAnimation

'''
Variables and symbols
'''

# Variables
Nx = 1000 # number of points for the x-axis
Nt = 5000 # number of points for time evolution
t = np.linspace(0, 10, Nt)
scal = 10
mu, sigma = 1/2, 1/20
Name_1 = np.array([])

# Token Test: This is used to ask the user what system they want to see
def Token_test():
    global state, Name_1, x_1, dx, Token, word

    print('The following commands will dictate what type of potential the system will be affected by\n')
    print('Input 1, for a Harmonic Oscillator\n')
    print('Input 2, for an Infinite Well\n')
    print('Input 3, for a Potential Step\n')
    print('Input 4, for a Half Harmonic Oscillator\n')
    print('Input 5, for a Dirac delta potential\n')
    print('Input 6, for 10*x - 0.5 * x**2 + x**3 + 0.5 * x**4 \nie fourth degree harmonic oscillator\n')
    print('Input 7, for a Line\n')
    print('Input 8, for a Gaussian\n')
    print('Input 9, for 1/x\n')

    Token = int(input())

    match Token:
        case 1:
            Name_1 = np.append(Name_1, 'harmonic oscillator')
            x_1 = np.linspace(-15, 15, Nx)
        
        case 2:
            Name_1 = np.append(Name_1, 'infinite well')
            x_1 = np.linspace(-1, 2, Nx)
        
        case 3:
            Name_1 = np.append(Name_1, 'potential step')
            x_1 = np.linspace(-5, 5, Nx)
        
        case 4:
            Name_1 = np.append(Name_1, 'half Harmonic oscillator (V(x<0) = ∞)')
            x_1 = np.linspace(-5, 5, Nx)

        case 5:
            Name_1 = np.append(Name_1, 'dirac delta potential')
            x_1 = np.linspace(-10, 10, Nx)
            print('State will be in the ground state')
            state = 0
        case 6:
            Name_1 = np.append(Name_1, 'fourth degree \n harmonic oscillator') 
            x_1 = np.linspace(-15, 15, Nx)
        
        case 7:
            Name_1 = np.append(Name_1, '\n a line') 
            x_1 = np.linspace(-15, 15, Nx)
        
        case 8:
            Name_1 = np.append(Name_1, 'gaussian')
            x_1 = np.linspace(-25, 25, Nx)
            
        case 9:
            Name_1 = np.append(Name_1, '1/x')
            x_1 = np.linspace(-15, 15, Nx)
        
        case 10:
            print('Not available as of this time, token will be set to 1')
            Token = 1
            Name_1 = np.append(Name_1, 'harmonic oscillator')
            x_1 = np.linspace(-15, 15, Nx)

        case _:
            print('Bad token')
            return  # early return if bad input
    
    Name_1 = Name_1[0]

    if Token == 5:
        state = 0
    else:
        print('Input the state that the system needs to be in:')
        state = int(input())

    match state:
        case 0:
            word = f" for the ground state of {Name_1}"

        case 1:
            word = f" for the first excited state of {Name_1}"
        
        case 2:
            word = f" for the second excited state of {Name_1}"
        
        case 3:
            word = f" for the third excited state of {Name_1}"

        case _:
            word = f' for the {state}th excited state of {Name_1}'

    dx = x_1[1] - x_1[0]


'''
Setting up H witht The steps of this is as follows:
1) Define the potentials of the system.
2) Construct the Hamiltonian as an N by N matrix.
3) Find the eigenvalues and eigenvectors, which are used for the time evolution and psi graphs, respectively.
4) Plot psi and |psi|^2.
'''

# Potentials
def V_1(x, token):

    match Token:
        case 1:
            return 0.5 * x**2  # Harmonic oscillator
        
        case 2:
            return np.where((x < 1) & (x > 0), 0, 10**5)  # Inf well
        
        case 3:
            return np.where(x > 0, 10, 0)  # Potential step
        
        case 4:
            return np.where(x > 0, 0.5 * x**2, 10**5)  # Half Harmonic oscillator
        
        case 5:
            return np.where(x != 0, 0, 10**5)  # Dirac delta    

        case 6:
            return 10*x - 0.5 * x**2 + x**3 + 0.5 * x**4 # x^4 Potential

        case 7: 
            return np.where((x >= 0) & (x <= 7), x - 7, 0) # Line  
          
        case 8:
            return np.exp(-np.pow(x,2)/25) # Gaussian
        
        case 9:
            with np.errstate(divide='ignore', invalid='ignore'):
                return 1/x

def V(x):
    return V_1(x, Token)

def pot():
    global H, psi_x, E_1, e_vals, e_vecs

    # Hamiltonian
    kinetic = diags([1, -2, 1], offsets=[-1, 0, 1], shape=(Nx, Nx)).toarray() / dx**2
    potential = np.diag(V(x_1))
    H = -kinetic/2 + potential  # Assuming ħ = m = 1
    
    # eig vals and vectors
    e_vals, e_vecs = np.linalg.eigh(H)

    psi_x = e_vecs[:, state]
    
    match Token:
        case 1 | 10:
            psi_x = psi_x[::-1]
        case 2:
            psi_x = -1*psi_x
        case 6:
            psi_x = psi_x[::-1]

    E_1 = e_vals[state]
    '''
    Plots for both V and psi plus V and |psi|**2
    '''
    # Plot V(x) and Psi
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title(r'$V(x)$', fontsize=16)
    plt.plot(x_1, V(x_1), label=r'$V(x)$', color='blue')
    plt.xlabel('x', fontsize=24)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.ylabel(r'V(x)', fontsize=24)
    plt.ylim(np.min(V(x_1)), np.max(V(x_1)))
    plt.xlim(np.min(x_1), np.max(x_1))

    plt.subplot(1, 2, 2)
    plt.title(r'$\psi(x,0)$' + word, fontsize=16)
    plt.plot(x_1, psi_x.real, label=r'$\psi(x)$', color='blue')
    plt.xlabel('x', fontsize=24)
    plt.ylabel(r'$\psi(x)$', fontsize=24)
    plt.grid(True)
    plt.xlim(np.min(x_1), np.max(x_1))
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)

    # Plot V(x) |psi|^2
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title(r'$V(x)$', fontsize=16)
    plt.plot(x_1, V(x_1), label=r'$V(x)$', color='blue')
    plt.xlabel('x', fontsize=24)
    plt.ylabel(r'$V(x)$', fontsize=24)
    plt.grid(True)
    plt.ylim(np.min(V(x_1)), np.max(V(x_1)))
    plt.xlim(np.min(x_1), np.max(x_1))
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title(r'$|\psi(x,t)|^2$' + word, fontsize=32)
    plt.plot(x_1, np.abs(psi_x)**2, label='Probability Density', color='blue')
    plt.xlabel('x', fontsize=24)
    plt.ylabel(r'$|\psi(x,t)|^2$', fontsize=24)
    plt.grid(True)
    plt.xlim(np.min(x_1), np.max(x_1))
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)

    '''
    No subplots
    '''
    # Plot the real part of the wavefunction
    plt.figure(figsize=(12, 5))
    plt.title(r'$\psi(x,0)$' + word, fontsize=32)
    plt.plot(x_1, psi_x.real, label=r'$\psi(x)$', color='blue')
    plt.xlabel('x', fontsize=24)
    plt.ylabel(r'$\psi(x)$', fontsize=24)
    plt.grid(True)
    plt.xlim(np.min(x_1), np.max(x_1))
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)

    # Plot the probability density
    plt.figure(figsize=(12, 5))
    plt.title(r'$|\psi(x,t)|^2$' + word, fontsize=32)
    plt.plot(x_1, np.abs(psi_x)**2, label='Probability Density', color='blue')
    plt.xlabel('x', fontsize=24)
    plt.ylabel(r'$|\psi(x,t)|^2$', fontsize=24)
    plt.grid(True)
    plt.xlim(np.min(x_1), np.max(x_1))
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)

    # Print normalization (should be close to 1 for a normalized wavefunction)
    print(f"Normalization: {np.sum(np.abs(psi_x)**2)}")

def energy_plot():
    e_vals_10 = np.array([e_vals[i] for i in range(10)])

    plt.figure(figsize=(12, 5))
    plt.title(r'First 10 energy states of a' + Name_1, fontsize=32)
    plt.xlabel(f'States', fontsize=24)
    plt.ylabel(r'E', fontsize=24)
    plt.grid(True)
    plt.tight_layout()
    for i in range(10):
        plt.scatter(i,e_vals_10[i],color = 'red')
    plt.show(block=False)

'''
Time evolution figure
How this is done is by setting up psi_n, and then adding e^-iE_nt, with E_n being the respective eigenvalue for n
'''

# Set up animation figure

def animate_wavefunction():
    global ani

    fig, ax = plt.subplots(figsize=(8, 5))
    line_re, = ax.plot([], [], lw=2, label=r'Re[$\psi(x,t)$]', color='blue')
    line_im, = ax.plot([], [], lw=2, linestyle='--', label=r'Im[$\psi(x,t)$]', color='red')

    ax.set_xlim(x_1[0], x_1[-1])
    ax.set_ylim(-1.2 * np.max(np.abs(psi_x)), 1.2 * np.max(np.abs(psi_x)))
    ax.set_xlabel("x", fontsize=24)
    ax.set_ylabel(r"$\psi(x,t)$", fontsize=24)
    ax.set_title(r"$\psi(x,t)$" + word, fontsize=32)
    ax.legend()
    ax.grid()

    def init():
        line_re.set_data([], [])
        line_im.set_data([], [])
        return line_re, line_im

    def update(frame):
        psi_t = psi_x * np.exp(-1j * E_1 * t[frame])
        line_re.set_data(x_1, psi_t.imag)
        line_im.set_data(x_1, psi_t.real)
        return line_re, line_im

    ani = FuncAnimation(fig, update, frames=np.arange(0, len(t), 10),init_func=init, blit=True, interval = 7.5)

    plt.tight_layout()
    plt.show(block=True)

"""
WKB methods is done as follows:
1) Set up a dic
2) Macthes the potential with the token that was given
3) Asks the user if psi is sin-like or cos-like, and if there exist any hard walls
4) Then it askes the user to input the WKB inergal, wiht it being that WKB as been dumb-down for the computer as much as possible for the algrithem used can not solve complexed intergals 
5) User must make WKB turn into an equation for energy 
6) Then E as a function of n is ploted
"""

"""
Dictionary
"""
# Symbol definitions (do NOT remove)
h, n, π, w, m, x, E, u, x0, ℏ, L, A, V_0, x, p = sp.symbols('h n π w m x E u x0 ℏ L A V_0 x p', real=True, positive=True)

# Dictionary used for expression parsing
DICT = {
    'h': h, 'n': n, 'π': π, 'w': w, 'm': m, 'x': x, 'E': E, 'u': u, 'x0': x0, 'ℏ': ℏ, 'L': L, 'A': A, 'V_0': V_0, 'p':p
}
transformations = standard_transformations

def get_expr(prompt): #This is used to make inputs for the WKB part
    try:
        user_input = input(prompt)
        if user_input.strip() == '':
            return None
        return parse_expr(user_input, local_dict=DICT, transformations=transformations)
    except Exception as e:
        print(f"Error parsing input: {e}")
        return None

def WKB():
    match Token:
        case 1:
            V_0 = 1/2 * m * w**2 * x**2

        case 2:
            V_0 = 0

        case 7:
            V_0 = x - 7

        case _:
            print("Invalid token or as yet to be added to the WKB part of this program")
            return

    # Choose wave type
    print(f'Input either "cos" or "sin" to best describe the wave equation of {Name_1}:')
    wave_type = input().strip().lower()
    if wave_type not in ('cos', 'sin'):
        print('Error: Input must be "cos" or "sin"')
        return
    
    print(f'Your input was: {wave_type}')

    # Soft walls
    print('Are there soft walls? If so, input the number of them, else put zero:')
    try:
        soft = int(input())
        if soft < 0:
            print('Error: Number of soft walls cannot be negative.')
            return

    except ValueError:
        print("Error: Must input an integer.")
        return
    print(f'Your input was: {soft}')

    # Quantization condition RHS
    Con = π * ℏ / 2 if wave_type == 'cos' else π * ℏ
    RHS = Con * (n - soft / 4)

    print('\nRHS (Quantization condition):')
    sp.pprint(RHS)
    
    main_expr = sp.sqrt(2 * m * (E - V_0))
    comp_expr = sp.sqrt(2 * m * E) * sp.sqrt(1 - V_0 / E)
    print("\nThe main integral is:")
    sp.pprint(main_expr)

    print("\nHow the computer might interpret the integral is as:")
    sp.pprint(comp_expr)

    Intergrad = sp.sqrt(1 - V_0/E)

    a = get_expr('Lower bound test (just pick a set of points that mimic the upper and lower bounds, or plug in the upper and lower bounds): ')
    if a is None: 
        return

    b = get_expr('Upper bound (same rule as above): ')
    if b is None: 
        return

    Intergal = sp.integrate(Intergrad, (x,a,b))
    
    if isinstance(Intergal, sp.Integral):
        inter_test = 1  # Good
    else:
        inter_test = 0  # Bad

    match inter_test:
        case 0:
            return print('For that the intergal being worked on does not have an elementary solution. This code will latter use appromation methods to find the value of the intergal')
        case 1:
            const_expr = get_expr("{sp.sqrt(2*m*E) * sp.sqrt(1 - V_0/E)} as an elementary solution \n use any methods to simplify the intergal so the computer can solve it. IE, u-sub,trig-sub and so on \n after doing that was told, input all constant that may of come about (e.g., 1, ℏ, etc.): ")
            if const_expr is None: 
                return
            
            a = get_expr('Lower bound test: ')
            if a is None: 
                return

            b = get_expr('Upper bound test: ')
            if b is None: 
                return
            
            Intergrad = get_expr('Simply the Intergal if need be. Keep everythign as in terms of x if u-sub was done: ')
            Intergal = sp.integrate(Intergrad, (x, a, b))

            LHS = const_expr * Intergal

    sp.pprint(sp.Eq(LHS, RHS))

    print(f'\nEquation for {Name_1}. n needs to starts at 1.')
    print('You may now solve the equation for E.')

    # Optional: Ask user to input a closed-form for E
    E_input = get_expr("After solving for E, input your expression for E in terms of h (not ℏ), n, etc: ")
    E_val_0 = np.array([e_vals[i] for i in range(0,6)])
    n_vals = np.arange(0, 6)

    if E_input:
        print("You entered:")
        sp.pprint(E_input)

        def E_func(n_val, use_units=False):
            subs_dict = {n: n_val}
            if use_units:
                subs_dict.update({h: 1, m: 1, w: 1, ℏ: 1, π: 1, L: 1})
            return E_input.subs(subs_dict)
        
        print('The fowlloing are the first six energy states ')
        for i in range(0,7):
            print(f'\nE(n={i}): {E_func(i)}')

    def E_func(n_val):
        subs_dict = {n: n_val, h: 1, m: 1, w: 1, ℏ: 1, π: 1, L: 1}
        return E_input.subs(subs_dict)

    try:
        E_vals = [float(E_func(i)) for i in n_vals]
    except Exception as e:
        print(f"Error evaluating E(n): {e}")
        return

    plt.figure(figsize=(8, 5))
    plt.title(r'Energy as a function of its state of a unit-less ' + Name_1, fontsize=24)
    plt.scatter(n_vals, E_vals, label='Energy', color='purple')
    
    plt.xlabel('n', fontsize=18)
    plt.ylabel(r'$E(n)$', fontsize=18)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

Token_test() #Everything runs based on the Token test
pot()
energy_plot()
#animate_wavefunction()
#WKB()
