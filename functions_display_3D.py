"""3D vizualization."""

__author__ = 'Ernesto Costa'
__date__ = 'February 2014'

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

def rastrigin():
    fig = plt.figure()
    fig.suptitle('Rastrigin', fontsize=18, fontweight='bold')
    ax = Axes3D(fig)
    
    
    X = np.arange(-5.12, 5.12, 0.1)
    Y = np.arange(-5.12, 5.12, 0.1)
    
    X, Y = np.meshgrid(X, Y)
    Z = 10*(X**2 - 10* np.cos(2*3.14159*X)) + 20*(Y**2 - 10* np.cos(2*3.14159*Y))
    
    ax.plot_surface(X, Y, Z, rstride=1,cstride=1,cmap=cm.jet)
    ax.legend()
    plt.savefig('rastrigin.png', bbox_inches='tight')
 
 
def de_jong_f1():
    fig = plt.figure()
    fig.suptitle('Sphere', fontsize=18, fontweight='bold')
    ax = Axes3D(fig)
    

    X = np.arange(-5.12, 5.12, 0.25)
    Y = np.arange(-5.12, 5.12, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = X**2 + Y**2
    
    ax.plot_surface(X, Y, Z, rstride=1,cstride=1,cmap=cm.jet)
    ax.legend()
    plt.savefig('sphere.png', bbox_inches='tight')

    
def de_jong_f2():
    fig = plt.figure()
    fig.suptitle('Rosenbrock', fontsize=18, fontweight='bold')
    ax = Axes3D(fig)
    
    
    X = np.arange(-5.12, 5.12, 0.1)
    Y = np.arange(-5.12, 5.12, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = 100*(X**2 - Y)**2 + (1 - X)**2
    
    ax.plot_surface(X, Y, Z, rstride=1,cstride=1,cmap=cm.jet)
    ax.legend()
    plt.savefig('rosenbrock.png', bbox_inches='tight')
 

def de_jong_f3():
    fig = plt.figure()
    fig.suptitle('Quartic', fontsize=18, fontweight='bold', color='red')
    ax = Axes3D(fig)
    
    
    X = np.arange(-1.28, 1.28, 0.05)
    Y = np.arange(-1.28, 1.28, 0.05)
    R = np.random.randn(len(X)) 
    
    X, Y = np.meshgrid(X, Y)
    Z = (X**4 + 2*Y**4) + R
    
    ax.plot_surface(X, Y, Z, rstride=1,cstride=1,cmap=cm.jet)
    ax.legend()
    plt.savefig('quartic.png', bbox_inches='tight')


def de_jong_f4():
    fig = plt.figure()
    fig.suptitle('Step', fontsize=18, fontweight='bold', color='red')
    ax = Axes3D(fig)

    X = np.arange(-5.12, 5.12, 0.1)
    Y = np.arange(-5.12, 5.12, 0.1)

    X, Y = np.meshgrid(X, Y)
    Z = 6*2 + (np.floor(X)+np.floor(Y))
    print(Z)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet)
    ax.legend()
    plt.savefig('step.png', bbox_inches='tight')


def schewefel():
    fig = plt.figure()
    fig.suptitle('Schewefel', fontsize=18, fontweight='bold')
    ax = Axes3D(fig)
    
    X = np.arange(-500, 500, 5)
    Y = np.arange(-500, 500, 5) 
    
    X, Y = np.meshgrid(X, Y)
    
    Z = -X * np.sin(np.sqrt(np.absolute(X))) -Y*np.sin(np.sqrt(np.absolute(Y)))
    
    ax.plot_surface(X, Y, Z, rstride=1,cstride=1,cmap=cm.jet)
    ax.legend()
    plt.savefig('schwefel.png', bbox_inches='tight')

    
def griegwang():
    fig = plt.figure()
    fig.suptitle('Griegwangk')
    ax = Axes3D(fig)
    
    
    X = np.arange(-600, 600, 5)
    Y = np.arange(-600, 600, 5) 
    
    X, Y = np.meshgrid(X, Y)
    
    Z = 1 + (1.0/4000)* (X**2 + Y**2) + np.cos(X) * np.cos(Y/np.sqrt(2.0))
    ax.plot_surface(X, Y, Z, rstride=1,cstride=1,cmap=cm.jet)
    ax.legend()
    plt.savefig('griegwangk.png', bbox_inches='tight')
   

def schubert():
    fig = plt.figure()
    fig.suptitle('Shubert', fontsize=18, fontweight='bold', color='red')
    ax = Axes3D(fig)


    X = np.arange(-10, 10, 0.15)
    Y = np.arange(-10, 10, 0.15)

    X, Y = np.meshgrid(X, Y)
    Z = shubert_2D(X,Y)

    ax.plot_surface(X, Y, Z, rstride=1,cstride=1,cmap=cm.jet)
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()



def shubert_2D(X,Y):
    Z = (np.cos(2*X+1) + 2*np.cos(3*X+2) + 3*np.cos(4*X+3) + 4* np.cos(5*X+4) + 5* np.cos(6*X+5))*\
        (np.cos(2*Y+1) + 2*np.cos(3*Y+2) + 3*np.cos(4*Y+3) + 4* np.cos(5*Y+4) + 5* np.cos(6*Y+5))
    return Z



if __name__ == '__main__':
    # de_jong_f1()
    # de_jong_f2()
    # de_jong_f3()
    # rastrigin()
    # schewefel()
    # de_jong_f4()
    griegwang()
    #schubert()

    
    
