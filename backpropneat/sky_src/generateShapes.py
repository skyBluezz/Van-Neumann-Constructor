import matplotlib.pyplot as plt
import math
import numpy as np

def generate_circle_dataset(Npoints): 

    # Radii of the two sets
    radii = [1,3]
    inputs = []
    targets = []

    for n in range(Npoints):

        # Give equal split to each targets
        target = 0 if n < (Npoints // 2) else 1

        # First half of elements will create points for the inner circle
        r = radii[target]

        # Add gaussian noise to the radius of the circle
        r += 1 * np.random.normal(0, 0.5)
        theta = 2 * np.pi * np.random.rand()
        x, y = r*np.cos(theta), r*np.sin(theta)

        inputs.append([x,y])
        targets.append(target)
    
    inputs, targets = np.stack(inputs), np.stack(targets)
    p = np.random.permutation(len(inputs))
    return inputs[p]/np.max(inputs), targets[p]

def generate_xor_dataset(Npoints): 

    targets = []
    samples = []
    xs = -1 + 2*np.random.rand(Npoints)
    ys = -1 + 2*np.random.rand(Npoints)
    targets = 1*np.logical_xor(xs > 0, ys > 0)
    return list(zip(xs,ys)), targets

def plot_dataset(inputs, targets):
    """Assumes a dataset with binary labels
    
        Args:
            inputs:   ListLike, length N list or array carrying an x,y coordinate
            targets:  ListLike, length N list or array cayying 0 or 1 for binary class
    """

    inputs  = np.stack(inputs)
    targets = np.array(targets)
    assert len(np.unique(targets)) == 2, "Ensure targets is a binary set"

    plt.scatter(inputs[:,0], inputs[:,1], c=targets)
    plt.show()
    return

def generate_spiral_dataset(Npoints):
    cwise  = [spiral_xy(i, 1) for i in range(Npoints)]
    ccwise = [spiral_xy(i, -1) for i in range(Npoints)]
    
    # Combine the two lists
    inputs  = cwise + ccwise 
    targets = [0]*Npoints + [1]*Npoints
    return np.stack(inputs), np.stack(targets)

def spiral_xy(i, spiral_num):
    """
    Create the data for a spiral.

    Arguments:
        i runs from 0 to Npoiints
        spiral_num is 1 or -1 for rotation direction
    """
    φ = i/16 * math.pi # Rotation increments
    r = 6.5 * ((104 - i)/104)
    x = (r * math.cos(φ) * spiral_num)/13 + 0.5
    y = (r * math.sin(φ) * spiral_num)/13 + 0.5
    return (x, y)
    
# i, t = generate_circle_dataset(2000)
# plot_dataset(i, t)