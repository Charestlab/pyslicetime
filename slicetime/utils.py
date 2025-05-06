import numpy as np

def linspacefixeddiff(x, d, n):
    """
    Return a vector of equally spaced values starting at `x`, 
    with fixed difference `d` between successive elements, for `n` elements.
    
    Parameters:
    x (float): starting value
    d (float): difference between successive values
    n (int): number of values

    Returns:
    np.ndarray: linearly spaced values
    """

    x2 = x + d * (n - 1)
    return np.linspace(x, x2, n)

# Example usage
print(np.allclose(linspacefixeddiff(0, 2, 5), [0, 2, 4, 6, 8]))  # Should print: True