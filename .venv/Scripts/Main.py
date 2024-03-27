import numpy as np

# Function to calculate distance between 2 points
def DistanceBetweenCities(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Function to calculate distance for the entire Path
def TourDistance(Path, CitiesX, CitiesY):
    TotalDistance = 0
    for i in range(len(Path)):
        if i == len(Path) - 1:
            TotalDistance += DistanceBetweenCities(CitiesX[Path[i]], CitiesY[Path[i]], CitiesX[Path[0]], CitiesY[Path[0]])
        else:
            TotalDistance += DistanceBetweenCities(CitiesX[Path[i]], CitiesY[Path[i]], CitiesX[Path[i+1]], CitiesY[Path[i+1]])
    print(TotalDistance)
    return TotalDistance

# Cities coordinates
c1_x = np.array([0, 3, 6, 7, 15, 12, 14, 9, 7, 0])
c1_y = np.array([1, 4, 5, 3, 0, 4, 10, 6, 9, 10])

#cities 2
c2_x = np.array([0, 2, 6, 7,  15,   12, 14, 9.5, 7.5, 0.5]);
c2_y = np.array([1, 3, 5, 2.5, -0.5, 3.5, 10, 7.5, 9, 10]);

#cities 3
c3_x = np.array([0, 3, 6, 7,  15,   10, 16, 5, 8, 1.5]);
c3_y = np.array([1, 2, 1, 4.5, -1, 2.5, 11, 6, 9, 12]);

#cities 4
c4_x = np.array([3, 2, 12, 7,  9,  3, 16, 11, 9, 2]);
c4_y = np.array([1, 4, 2, 4.5, 9, 1.5, 11, 8, 10, 7]);


# Population size
size = 250

# Initialize population
Population = np.zeros((size, 10), dtype=int)

# Generate random population
for i in range(size):
    random_arr = np.random.permutation(10)
    Population[i, :] = random_arr

# Calculate distances for the first individual in the population
distances = TourDistance(Population[0], c3_x, c3_y)
