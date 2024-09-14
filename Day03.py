pip install basemap

import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(0,10,100)

fig=plt.figure()
plt.plot(x,np.sin(x),'-')
plt.plot(x,np.cos(x),'--')

from sklearn.datasets import load_iris
iris=load_iris()
features=iris.data.T
plt.scatter(features[0],features[1],alpha=0.2,s=100*features[3],c=iris.target,cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

def f(x,y):
    return np.sin(x) ** 10 + np.cos(10+y*x) *np.cos(x)
x=np.linspace(0, 5, 50)
y=np.linspace(0, 5, 40)

x,y=np.meshgrid(x,y)
z=f(x,y)
plt.contour(x,y,z,colors='black');

plt.contour(x,y,z,20,cmap='RdGy')

plt.contourf(x,y,z,20,cmap='RdGy')
plt.colorbar();

plt.imshow(z, extent=[0, 5, 0, 5], origin="lower",cmap='RdGy')
plt.colorbar();

contours =plt.contour(x,y,z,colors='black');
plt.clabel(contours,inline=True,fontsize=8)
plt.imshow(z,extent=[0,5,0,5],origin='lower',cmap='RdGy',alpha=0.5)
plt.colorbar();

plt.style.use('seaborn-white')
data=np.random.randn(1000)
plt.hist(data);

mean=[0,0]
cov=[[1,1],[1,2]]
x,y=np.random.multivariate_normal(mean,cov,10000).T
plt.hist2d(x,y,bins=30,cmap="Blues")
cb=plt.colorbar()
cb.set_label('counts in bin')

counts,xedges,yedges=np.histogram2d(x,y,bins=30)
plt.hexbin(x,y,gridsize=30,cmap='Blues')
cb=plt.colorbar(label='count in bin')

# HANDWRITTEN DIGITS
from sklearn.datasets import load_digits
digits = load_digits(n_class=6)

fig,ax=plt.subplots(8,8,figsize=(6,6))
for i,axi in enumerate(ax.flat):
    axi.imshow(digits.images[i],cmap='binary')
    axi.set(xticks=[],yticks=[])

fig,ax=plt.subplots(5,5,figsize=(5,5))
fig.subplots_adjust(hspace=0,wspace=0)

from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces().images
for i in range(5):
    for j in range(5):
        ax[i,j].xaxis.set_major_locator(plt.NullLocator())
        ax[i,j].yaxis.set_major_locator(plt.NullLocator())
        ax[i,j].imshow(faces[10*i+j],cmap="bone")

from mpl_toolkits import mplot3d
fig = plt.figure()
ax=plt.axes(projection='3d')

ax=plt.axes(projection='3d')
zline = np.linspace(0,15,1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline,yline,zline,'gray')

zdata=15*np.random.random(100)
xdata=np.sin(zdata)+0.1*np.random.rand(100)
ydata=np.cos(zdata)+0.1*np.random.rand(100)
ax.scatter3D(xdata,ydata,zdata,c=zdata,cmap='Greens');

def f(x,y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

x=np.linspace(-6,6,30)
y=np.linspace(-6,6,30)

x,y=np.meshgrid(x,y)
z=f(x,y)

fig=plt.figure()
ax=plt.axes(projection="3d")
ax.contour3D(x,y,z,50,cmap='binary')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.view_init(60,35)
fig

fig=plt.figure()
ax=plt.axes(projection="3d")
ax.plot_wireframe(x,y,z,color='black')
ax.set_title('wireframe');

ax=plt.axes(projection="3d")
ax.plot_surface(x,y,z,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
ax.set_title('wireframe');

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(8,8))
m=Basemap(projection='ortho',resolution=None,lat_0=50,lon_0=-100)
m.bluemarble(scale=0.5);

def draw_map(m,scale=0.2):
    m.shadedrelief(scale=scale)
    
fig = plt.figure(figsize=(8,6),edgecolor='w')
m=Basemap(projection='cyl',resolution=None,llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180)
draw_map(m)

fig = plt.figure(figsize=(8,6),edgecolor='w')
m=Basemap(projection='moll',resolution=None,lat_0=50,lon_0=-100)
draw_map(m)

plt.figure(figsize=(8,8))
m=Basemap(projection='ortho',resolution=None,lat_0=50,lon_0=0)
draw_map(m)

plt.figure(figsize=(8,8))
m=Basemap(projection='lcc',resolution=None,lat_0=50,lon_0=0,lat_1=45,lat_2=55,width=1.6E7,height=1.2E7)
draw_map(m)