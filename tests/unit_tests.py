# %%
import os
import sys

# Get the current script's directory
current_dir = os.path.dirname(__file__)

# Calculate the path to the 'src' directory
src_dir = os.path.join(current_dir, '../src')

# Add the 'src' directory to the sys.path
sys.path.append(src_dir)

# Now you can import 'gaussian' as if it's a regular module
import unittest
import numpy as np
import guassian 

class TestGaussianFilter(unittest.TestCase):
    def test_default_parameters(self):
        # Test the function with default parameters
        kernel_size = 3
        gaussian = guassian.gaussian_2d(kernel_size=kernel_size)
        expected_gaussian = np.array([[0.05854983, 0.09653235, 0.05854983],
                                      [0.09653235, 0.15915494, 0.09653235],
                                      [0.05854983, 0.09653235, 0.05854983]])
        np.testing.assert_array_almost_equal(gaussian, expected_gaussian)
        
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
