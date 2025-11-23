import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from matplotlib.animation import FuncAnimation, PillowWriter
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations
from matplotlib.animation import FuncAnimation
import math as math
import matplotlib as mpl
import seaborn as sb

x_0, x_N, N = -10,10, 1500
h = (x_N - x_0)/N

def Token_test():
    global Token, state, Name_1, word

    Name_1 = np.array([])

    '''
    Avalible inputs
    '''

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
    print('Input 10: for no Potentials')
    
    Token = int(input())
    
    match Token:
        case 1:
            Name_1 = np.append(Name_1, 'a harmonic oscillator')
        
        case 2:
            Name_1 = np.append(Name_1, 'an infinite well')
        
        case 3:
            Name_1 = np.append(Name_1, 'a potential step')
        
        case 4:
            Name_1 = np.append(Name_1, 'a half Harmonic oscillator (V(x<0) = ∞)')

        case 5:
            Name_1 = np.append(Name_1, 'a dirac delta potential')

        case 6:
            Name_1 = np.append(Name_1, 'a fourth degree \n harmonic oscillator') 
        
        case 7:
            Name_1 = np.append(Name_1, '\n a line potential') 
        
        case 8:
            Name_1 = np.append(Name_1, 'a gaussian')
            
        case 9:
            Name_1 = np.append(Name_1, '1/x')
        
        case 10:
            Name_1 = np.append(Name_1, 'V(x) = 0')

        case _:
            print('Bad token')
            return  # early return if bad input
    
    if Token == 5:
        print('State will be in the ground state')
        state = 0
    else:    
        print('Input the state you wish the system to be in')
        state = int(input())
    
    '''
    Grammer
    '''

    match state:
        case 0:
            word = f"The ground state of {Name_1[0]}"

        case 1:
            word = f"The first excited state of {Name_1[0]}"
        
        case 2:
            word = f"The second excited state of {Name_1[0]}"
        
        case 3:
            word = f"The third excited state of {Name_1[0]}"

        case _:
            word = f'The {state}th excited state of {Name_1[0]}'

def V(x):
    global x_0, x_N, h

    '''
    The set of Potentials
    '''

    match Token:
        case 1:
            return 0.5 * (x) ** 2
        
        case 2:
            x_0, x_N = -1, 2
            h = (x_N - x_0)/N

            if x < 1 and x > 0:
                return 0
            else:
                return 10**5
            
        case 3:
            if x < 0:
                return 0
            else:
                return 10
            
        case 4:
            if x > 0:
                return 0.5 * (x) ** 2
            else:
                return 10**5
        
        case 5: # sill needs work
            if x != 0:
                return 0
            else:
                return 10**5

        case 6:
            return 10*(x) - 0.5 * (x)**2 + (x)**3 + 0.5 * (x)**4
        
        case 7:
            if x >= 0 and x <= 7:
                return x - 7
            else:
                return 0

        case 8:
            return math.exp(-math.pow(x,2)/25)
        
        case 9:
            if x <= 0 and abs(x) < h:
                return -1/h 
            elif x > 0 and abs(x) < h:
                return 1/h 
            else:
                return 1/x
            
        case 10:
            return 0
        
def H():
    global x_points, psi_points, E_1, e_vals, e_vecs

    i = 0 # set up

    '''
    Setting up the kinetic energy part of H
    '''

    H_kin = diags([1, -2, 1], offsets=[-1, 0, 1], shape=(N + 1, N + 1)).toarray()
    H_kin /= h**2


    '''
    Setting up the potential
    '''

    H_pot = np.zeros((N + 1, N + 1))
    
    x = x_0

    while i < N: # The following while loop is to digitized H_pot
        H_pot[i][i] = V(x)
        i += 1
        x = x_0 + i * h
    if i == N:
        H_pot[i][i] = V(x_N)

    '''
    Setting up the total energy and then finding the allowed enegies and states 
    '''

    H = -H_kin/2 + H_pot
    e_vals, e_vecs = np.linalg.eigh(H)

    psi_x = e_vecs[:, state]
    E_1 = e_vals[state]

    j = 0
    x = x_0

    '''
    Setting up the data points for the x and psi values
    '''

    x_points = np.array([])
    psi_points = np.array([])

    while j < N:
        x_points = np.append(x_points, x)
        psi_points = np.append(psi_points, psi_x[j])

        j += 1
        x = x_0 + j * h

    if j == N:
        x_points = np.append(x_points, x)
        psi_points = np.append(psi_points, psi_x[j])

    x_points = x_points[::-1]   # Need to flip the terms to maintain order
    psi_points = psi_points[::-1]

'''
The following functions are the momentum and momentum squared oprators
'''

def p_hat(psi):
    d_dx = diags([-1/2, 0, 1/2], offsets=[-1, 0, 1], shape=(N + 1, N + 1)).toarray()

    d_dx[0, :3]  = np.array([-3/2, 2, -1/2])
    d_dx[-1, -3:] = np.array([1/2, -2, 3/2])

    d_dx /= h

    return -1j*(np.matmul(d_dx, psi))

def p_hat_2(psi):
    d2_dx2 = diags([1, -2, 1], offsets=[-1, 0, 1], shape=(N + 1, N + 1)).toarray()

    d2_dx2[0, 0:4] = np.array([2, -5, 4, -1])
    d2_dx2[-1, -4:] = np.array([-1, 4, -5, 2])

    d2_dx2 /= h**2

    return -1 * np.matmul(d2_dx2, psi)

def plot():
    
    '''
    Prints <x>,<x**2>, <p>, <p**2>, and the Heisenberg's Uncertainty Principle
    '''
    
    print('\n')
    print('-'*30)
    print(f'\n<\u03A8|\u03A8> = {np.sum(psi_points.T * psi_points):.2f}\n<x> = {np.sum(psi_points.T * x_points * psi_points):.2f}\n<x**2> = {np.sum(psi_points.T * (x_points)**2 * psi_points):.2f}\n<p> = {np.sum(psi_points.T * p_hat(psi_points))}\n<p**2> = {np.sum(psi_points.T * p_hat_2(psi_points)):.2f}')
    
    #the next two lines of code are used to define sigma values of x and p
    sigma_x = np.sqrt(np.sum(psi_points.T * (x_points)**2 * psi_points) - np.sum(psi_points.T * x_points * psi_points)**2)
    sigma_p = np.sqrt(np.sum(psi_points.T * p_hat_2(psi_points)) - np.sum(np.abs(psi_points.T * p_hat(psi_points)))**2)
    
    print(f'Heisenberg\'s Uncertainty Principle: {(sigma_x * sigma_p):.2} ≥ 1/2')

    '''
    The following lines in this fuction are uwed to plot, psi, psi,^2
    '''

    plt.style.use('dark_background')
    plt.figure(figsize=(12, 8))
    plt.tight_layout()
    plt.plot(x_points, psi_points)
    plt.title(f'{word}', fontsize = 26)
    plt.ylabel('\u03A8   ', rotation = 360, fontsize=20)
    plt.xlabel('x', fontsize = 20)

    if Token == 2:
        plt.xlim(-1,2)

    plt.axhline(y=0, color='white', linewidth=1)
    plt.axvline(x=0, color='white', linewidth=1)      
    plt.xticks([])
    plt.yticks([])

    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.style.use('dark_background')
    plt.tight_layout()
    plt.plot(x_points, psi_points.T * psi_points)
    plt.title(f'{word}', fontsize = 26)
    plt.ylabel('|\u03A8|^2         ', rotation = 360, fontsize=20)
    plt.xlabel('x', fontsize = 20)

    if Token == 2:
        plt.xlim(-1,2)

    plt.axhline(y=0, color='white', linewidth=1)
    plt.axvline(x=0, color='white', linewidth=1)      
    plt.xticks([])
    plt.yticks([])

    plt.grid()
    plt.show()

    '''
    The next set of lines are used to plot the first 10 enegry states
    '''

    if Token != 5: # There is no need to show ay other states 
        Ene = np.array([e_vals[i] for i in range(10)])
        data_set = np.arange(0,10)
    
        plt.style.use('dark_background')
        plt.figure(figsize=(12, 8))
        plt.tight_layout()
        plt.scatter(data_set, Ene)
        plt.title(f'Energy states of {word}', fontsize = 26)
        plt.ylabel('E', rotation = 360, fontsize=20)
        plt.xlabel('states', fontsize = 20)
        plt.axhline(y=0, color='white', linewidth=1)
        plt.axvline(x=0, color='white', linewidth=1)      
        plt.show()



def animate_wavefunction():

    '''
    The aim of this function is to show the Heisenberg picture of physics by adding a time evolution operator
    '''

    global ani

    #line spacing for time 
    t = np.linspace(0, 2 * np.pi / abs(E_1), 1000)

    '''
    Developing the plots
    '''
    fig, ax = plt.subplots(figsize=(12, 8))
    line_re, = ax.plot([], [], lw=2, label='real', color='blue')
    line_im, = ax.plot([], [], lw=2, linestyle='--', label='imaginary', color='red')

    ax.set_xlim(x_0, x_N)
    ax.set_ylim(-1.2 * np.max(np.abs(psi_points)), 1.2 * np.max(np.abs(psi_points)))
    ax.set_xlabel("x", fontsize=24)
    ax.set_ylabel('\u03A8   ', rotation = 360, fontsize=24)
    ax.set_title(f"{word}", fontsize=32)
    ax.legend()

    '''
    Setting up the real and imaginary data set 
    '''

    def init():
        line_re.set_data([], [])
        line_im.set_data([], [])
        return line_re, line_im

    def update(frame): # This function addes to the aforementioned data sets
        psi_t = psi_points * np.exp(-1j * E_1 * t[frame])
        line_re.set_data(x_points, psi_t.imag)
        line_im.set_data(x_points, psi_t.real)
        return line_re, line_im

    # The code below is an importated function that will animate the real and imaginary lines
    ani = FuncAnimation(fig, update, frames=np.arange(0, len(t), 10),init_func=init, blit=True, interval = 7.5)
    
    '''
    Fine tooning the plot
    
    '''
    plt.axhline(y=0, color='white', linewidth=1)
    plt.axvline(x=0, color='white', linewidth=1)   
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.style.use('dark_background')
    plt.show()

Token_test()
H()
plot()
animate_wavefunction()
