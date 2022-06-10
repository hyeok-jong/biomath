# Import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def plot_1_a(epsilon_list = [2, 5, 10, 15, 2, 5, 10, 15]):
    
    plt.figure(figsize = (8,6))
    cnt = 0
    
    # Set color list for plots with HTML forms
    color_list  = ["#2F4EA1", "#EE2123", "#9F3795", "#3F3F3F", "#127C3D", "#1DB6B5", "#B8B933", "#2F41A1"]
    
    for epsilon in epsilon_list:
  
        # ODE Equation in Research Article
        def my_ode(U, t, epsilon = epsilon, eta = 0.002, lambda1  = 0.001, lambda2 = 0.001, lambda3 = 0.001):
            u = U[0]
            v = U[1]
            w = U[2]
            
            return [-epsilon*u + u*v + lambda1*v,
                    epsilon*u - eta*v + lambda3*epsilon*w - u*v - lambda3*v,
                    lambda2*v - lambda3*epsilon*w + lambda3*v*w]
        
        def my_function(t, epsilon = epsilon, eta = 0.002, lambda1  = 0.001, lambda2 = 0.001, lambda3 = 0.001):
            u = np.exp(-epsilon*t) + (-np.exp(-2*epsilon*t)+np.exp(-epsilon*t))/(eta-epsilon) + (epsilon*(np.exp(-(eta+epsilon)*t)-np.exp(-epsilon*t)))/(eta*(eta-epsilon))\
              +(lambda1*epsilon*t*np.exp(-epsilon*t))/(eta-epsilon) + lambda1*epsilon*(np.exp(-eta*t)-np.exp(-epsilon*t))/(eta-epsilon)
           
            v = (epsilon/(eta-epsilon))*(np.exp(-epsilon*t)-np.exp(-eta*t))+epsilon*(np.exp(-eta*t)-np.exp(-2*epsilon*t))/((eta-epsilon)*(2*epsilon-eta))\
               +(np.exp(-(epsilon+eta)*t)-np.exp(-eta*t))/(eta-epsilon)

            w = ((lambda2*epsilon)/(eta-epsilon))*((np.exp(-epsilon*t)-np.exp(-lambda3*epsilon*t))/((lambda3-1)*epsilon)-(np.exp(-eta*t)-np.exp(-lambda3*epsilon*t))/(lambda2*epsilon-eta))\
               +(lambda2*(np.exp(-2*epsilon*t)-np.exp(-lambda3*epsilon*t)))/((epsilon-eta)*(eta-2*epsilon)*(lambda3-2))-(lambda2*(np.exp(-(epsilon+eta)*t)-np.exp(-lambda3*epsilon*t)))/((epsilon-eta)*(lambda3-epsilon-eta))\
               -(lambda2*epsilon*(np.exp(-eta*t)-np.exp(-lambda3*epsilon*t)))/((epsilon-eta)*(eta-2*epsilon)*(lambda3*epsilon-eta))+(lambda2*(np.exp(-eta*t)-np.exp(-lambda3*epsilon*t)))/((epsilon-eta)*(lambda3*epsilon-eta-2))

            return u, v, w
            
        # Initial values
        initial = [1, 0, 0]
        
        # Time range ; x axis
        time_range = np.linspace(0, 1, 500)
        
        # Get y value from ODE Equation
        ys = odeint(my_ode, initial, time_range)
        
        # Get y values from Equation
        y = my_function(time_range)
        
        
        # Plot
        # With different colors
        if cnt <4:
            plt.plot(time_range, y[0], color = color_list[cnt] , linewidth = 1, linestyle = '-')
        else:
            plt.plot(time_range, ys[:,0], color = color_list[cnt] , linewidth = 1.5, linestyle = ':')
        # For getting various colors
        cnt += 1
        
        
    # Using LaTeX in Python.
    plt.xlabel(xlabel = r"Dimensionless time ($\tau$)", size = 16, fontdict = {'color':'black', 'fontname':'Mongolian Baiti'})
    plt.ylabel(ylabel = r"Dimensionless concentration ($u$)", size = 16, fontdict = {'color':'black', 'fontname':'Mongolian Baiti'})
    
    # Index
    plt.text( 0.6, 0.9, r"$\lambda_1 = \lambda_2 = \lambda_3 = 0.001$", size = 16, fontdict = {'color':'black', 'family':'Mongolian Baiti'})
    
    # Labels for each lines
    plt.text( 0.4, 0.52, r"$\epsilon = 2$", size = 16, fontdict = {'color':'black', 'family':'Mongolian Baiti'})
    plt.text( 0.3, 0.3, r"$\epsilon = 5$", size = 16, fontdict = {'color':'black', 'family':'Mongolian Baiti'})
    plt.text( 0.35, 0.05, r"$\epsilon = 10$", size = 16, fontdict = {'color':'black', 'family':'Mongolian Baiti'})
    plt.text( 0.05, 0.01, r"$\epsilon = 15$", size = 16, fontdict = {'color':'black', 'family':'Mongolian Baiti'})
    
    plt.text( 0.45, -0.3, "(a)", size = 16, fontdict = {'color':'black', 'family':'Mongolian Baiti'})
    
    # Set axis grid
    plt.xticks([i/10 for i in range(0,11,2)])
    plt.yticks([i/10 for i in range(0,11,2)])
    
    # Set range to display
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig("1_a.jpeg", dpi = 800)
    
    

plot_1_a()
