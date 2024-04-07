import numpy as np
import matplotlib.pyplot as plt

xmax = 10
n = 2

Resolution = 500

Height = np.zeros((Resolution, Resolution))

class Particle:
    def __init__(self, dim):
        self.position = np.random.uniform(0, xmax, dim)
        self.velocity = np.random.uniform(0, xmax/10, dim)
        self.best_position = self.position
        self.best_value = float('-inf')

def AlpineHeight(position,dim):

    SinValues = 1
    RootValues = 1

    for i in range(0,dim):
        SinValues = SinValues * np.sin(position[i])
        RootValues = (RootValues * position[i])

    if RootValues < 0:
        RootValues = 0

    result = (SinValues * np.sqrt(RootValues))
    return result

def MapValues(x, x1, x2, y1, y2):
    """
       Map values from range (x1, x2) to range (y1, y2).

       Args:
       x: Value to be mapped
       x1: Start of the original range
       x2: End of the original range
       y1: Start of the target range
       y2: End of the target range

       Returns:
       Mapped value in the target range
       """
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


def optimize( dim, num_particles, max_iter):
    # Initialize particles
    particles = [Particle(dim) for _ in range(num_particles)]

    # Initialize global best
    global_best_position = None
    global_best_value = float('-inf')

    for Iteration in range(max_iter):

        print(global_best_value)

        for particle in particles:
            # Evaluate particle
            particle_value = AlpineHeight(particle.position,dim)

            # Update personal best
            if particle_value > particle.best_value:
                particle.best_position = particle.position
                particle.best_value = particle_value

            # Update global best
            if particle_value > global_best_value:
                global_best_position = particle.position
                global_best_value = particle_value

        # Update velocities and positions
        for particle in particles:
            inertia_weight = 1
            cognitive_weight = 1.5
            social_weight = 1.5

            # Update velocity
            particle.velocity = (inertia_weight/(Iteration+1) * particle.velocity +
                                 cognitive_weight * np.random.rand(dim) * (particle.best_position - particle.position)
                                 + social_weight * np.random.rand(dim) * (global_best_position - particle.position))

            # Update position
            particle.position += particle.velocity

            # Clamp position to the search space
            particle.position = np.clip(particle.position, 0, 10)

    return global_best_position, global_best_value



## this fragment is just used to create an approximate plot of the function

for i in range (0, Resolution):
    for j in range(0, Resolution):
        EndRange = np.pow(xmax,n)/xmax
        ValueX = MapValues(i,0,Resolution,0,EndRange)
        ValueY = MapValues(j,0,Resolution,0,EndRange)

        Height[i,j] = AlpineHeight([ValueX, ValueY],2)


# Generate some sample data
x = np.linspace(0, xmax, Resolution)
y = np.linspace(0, xmax, Resolution)
x, y = np.meshgrid(x, y)


# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(x, y, Height)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Plot')

optimal_position, optimal_value = optimize( n, 50, 500)

ax.scatter(optimal_position[0], optimal_position[1], optimal_value, color='red', s=100)

print(optimal_value, optimal_position)
# Show the plot
plt.show()

