# 3D Animation of 2D Diffusion Equation using Python
```py
import scipy as sp
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation



dx=0.01        
dy=0.01     
a=0.5          
timesteps=500  
t=0.

nx = int(1/dx)
ny = int(1/dy)


dx2=dx**2 
dy2=dy**2 

dt = dx2*dy2/( 2*a*(dx2+dy2) )

ui = sp.zeros([nx,ny])
u = sp.zeros([nx,ny])

for i in range(nx):
 for j in range(ny):
  if ( ( (i*dx-0.5)**2+(j*dy-0.5)**2 <= 0.1)
   & ((i*dx-0.5)**2+(j*dy-0.5)**2>=.05) ):
    ui[i,j] = 1
def evolve_ts(u, ui):
 u[1:-1, 1:-1] = ui[1:-1, 1:-1] + a*dt*( 
                (ui[2:, 1:-1] - 2*ui[1:-1, 1:-1] + ui[:-2, 1:-1])/dx2 + 
                (ui[1:-1, 2:] - 2*ui[1:-1, 1:-1] + ui[1:-1, :-2])/dy2 )
        

def data_gen(framenumber, Z ,surf):
    global u
    global ui
    evolve_ts(u,ui)
    ui[:] = u[:]
    Z = ui
    
    ax.clear()
    plotset()
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.7)
    return surf,


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = sp.arange(0,1,dx)
Y = sp.arange(0,1,dy)
X,Y= sp.meshgrid(X,Y)

Z = ui 

def plotset():
    ax.set_xlim3d(0., 1.)
    ax.set_ylim3d(0., 1.)
    ax.set_zlim3d(-1.,1.)
    ax.set_autoscalez_on(False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    cset = ax.contour(X, Y, Z, zdir='x', offset=0. , cmap=cm.coolwarm)
    cset = ax.contour(X, Y, Z, zdir='y', offset=1. , cmap=cm.coolwarm)
    cset = ax.contour(X, Y, Z, zdir='z', offset=-1., cmap=cm.coolwarm)

plotset()
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.7)


fig.colorbar(surf, shrink=0.5, aspect=5)

ani = animation.FuncAnimation(fig, data_gen, fargs=(Z, surf),frames=1000, interval=30, blit=False)
ani.save("2dDiffusion.mp4", bitrate=1024)

#plt.show()    

```
