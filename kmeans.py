import pennylane as qml
from pennylane import numpy as np
from sklearn.cluster import KMeans
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path to your image
image_path = r'C:\Users\colle\OneDrive\Quantum\Side Project\manatee_outline.png'

# Load the image
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Unable to load image at {image_path}")
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply edge detection
    edges = cv2.Canny(blurred, threshold1=450, threshold2=550)

    # Find contours with higher accuracy
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Plot all contours with reduced coordinates
    plt.figure()
    for contour in contours:
        # Simplify the contour to reduce the number of points
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        coordinates = approx[:, 0, :]

        plt.plot(coordinates[:, 0], coordinates[:, 1], marker='o')

        # Add coordinates to the plot
        #for (x, y) in coordinates:
            #plt.text(x, y, f'({x},{y})', fontsize=8, ha='right')

    plt.title('Simplified Image Outline with Coordinates')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.gca().invert_yaxis()  # Invert y-axis to match the image coordinates
    plt.show()
# Create a device with a simulator backend
dev = qml.device('default.qubit', wires=2)

# Define a quantum node
@qml.qnode(dev)
def circuit(params):
    qml.RY(params[0], wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(params[1], wires=1)
    return qml.expval(qml.PauliZ(0))

# Perform k-means clustering
coordinates = np.array(coordinates)
kmeans = KMeans(n_clusters=5, random_state=0).fit(coordinates)
cluster_centers = kmeans.cluster_centers_

# Define the TSP problem using the cluster centers
n = len(cluster_centers)
distances = np.linalg.norm(cluster_centers[:, np.newaxis] - cluster_centers, axis=2)

# Define the cost function for TSP
def cost_function(params):
    return sum(distances[i, j] * circuit(params) for i in range(n) for j in range(n) if i != j)

# Optimize the cost function using a classical optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.1)
params = np.random.randn(2)
steps = 100

for _ in range(steps):
    params = opt.step(cost_function, params)

# Print the optimized parameters
print("Optimized parameters:", params)

# Define the shortest path based on the optimized parameters
path = []
for i in range(n):
    for j in range(n):
        if i != j and circuit(params) > 0.5:
            path.append((i, j))

# Print the path
print("Shortest path:", path)