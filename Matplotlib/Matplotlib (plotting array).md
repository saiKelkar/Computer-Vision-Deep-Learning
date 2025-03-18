```
import matplotlib.pyplot as plt

# To load the graph inside the notebook
%matplotlib inline

a = np.array([2, 1, 5, 7, 4, 6, 8, 14, 10, 9, 18, 20, 22])

plt.plot(a)
plt.show()
```

```
# For multi-dimensional visualizations

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

X = np.arange(-5, 5, 0.15)
Y = np.arange(-5, 5, 0.15)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = 'viridis')
```

```
plt.figure() <-- creates a blank figure where we can draw plots

add_subplot(projection = '3d') <-- adds a 3D subplot to the figure, allowing you to plot in 3D space

np.arange(-5, 5, 0.15) <-- np.arange(start, stop, step)

np.meshgrid() <-- generates a 2D grid from the 1D arrays X and Y
for example: if X = [0, 1, 2] and Y = [0, 1, 2], 
np.meshgrid() gives,

X = [[0, 1, 2],
     [0, 1, 2],
     [0, 1, 2]]

Y = [[0, 0, 0],
     [1, 1, 1],
     [2, 2, 2]]

R = np.sqrt(X**2, Y**2) <-- computes the Euclidean distance - radius - from the origin to each point on the grid
Z = np.sin(R) <-- calculates the sine of each radius values to get the height - this generates the wavy surface like ripples

plot_surface() <-- plots a 3D surface using the X, Y, Z data
rstride = 1 <-- row stride (step size along the Y-axis)
smaller values make a smoother surface
cstride = 1 <-- column stride (step size along the X-axis)
cmap = 'viridis' <-- colormap to color the surface
'viridis' is a popular gradient from yellow to blue

```

![[Pasted image 20250313221102.png]]
