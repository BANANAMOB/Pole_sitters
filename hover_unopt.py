import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, nsolve

from matplotlib.colors import LogNorm


MISSING = object()

# CR3BP parameters for Earth-Moon system
mu = 0.012150585609624  #  (Moon)/ (Earth + Moon)

def hover_eq(x, y):

    R31 = ((x+mu)**2+y**2)**(1/2)
    R32 = ((x-1+mu)**2+y**2)**(1/2)

    u = (1-mu)*y / R31**3 + mu*y / R32**3

    return u

if __name__ == "__main__":
   
    # Create hover line
    x, y = np.meshgrid(np.linspace(-1.25, 1.25, 500), np.linspace(0.001, 1.25, 800))

    f = (-x +
        (1-mu)*(x+mu) / ((x+mu)**2 + y**2)**1.5 +
        mu*(x-1+mu) / ((x-1+mu)**2 + y**2)**1.5)

    fig, ax = plt.subplots()

    cs = plt.contour(x, y, f, levels=[0])
    plt.close(fig)
    path = cs.get_paths()[0].vertices
    x_line, y_line = path[:, 0], path[:, 1]

    # Solve for hover solutions
    us = np.zeros_like(x_line)
    for i in range(len(x_line)):
        us[i] = hover_eq(x_line[i], y_line[i])


    # solve for lagrange points l1^ l2< l3>
    xL = symbols('xL')
    fL = (-xL + (1-mu)*(xL+mu) / (abs(xL+mu)**3) + mu*(xL-1+mu) / abs(xL-1+mu)**3)

    xL3 = nsolve(fL, xL, -1.2)
    xL1 = nsolve(fL, xL, 0.8)
    xL2 = nsolve(fL, xL, 1.2)
    print(xL3)
    print(xL1)
    print(xL2)

    # plot
    style_path = "/home/lha43/Desktop/Lucas/Generated_figures/paper.mplstyle"
    plt.style.use(style_path)

    plt.scatter(x_line, y_line, c=us, cmap='plasma',
                norm=LogNorm(vmin=us.min(), vmax=us.max()), s=20, zorder=3)
    plt.colorbar(label='$u$ [DU/TU$^2$]')

    plt.scatter(-mu, 0, color='dodgerblue', label='Earth', s=50, zorder=4)
    plt.scatter(1 - mu, 0, color='gray', label='Moon', s=40,zorder=4)
    plt.scatter(xL3, 0, color='k', label="L3",marker='>',edgecolor='w',s=50,zorder=4)
    plt.scatter(xL1, 0, color='k', label="L1",marker='^',edgecolor='w',s=50,zorder=4)
    plt.scatter(xL2, 0, color='k', label="L2",marker='<',edgecolor='w',s=50,zorder=4)
    plt.xlabel('$x$ [DU]')
    plt.ylabel('$z$ [DU]')
    #plt.xlim(0.75,1.25)
    #plt.ylim(0.0,0.2)
    plt.grid(True, zorder=0)
    plt.legend()
    plt.show()
       

