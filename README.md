# Gaussian Filter Function

The `gaussian_filter` function is a Python implementation of a 2D Gaussian filter, commonly used in image processing for tasks such as blurring or smoothing an image. This README provides an overview of the function, its parameters, and how to use it.

## Table of Contents

- [Gaussian Filter Function](#gaussian-filter-function)
  - [Table of Contents](#table-of-contents)
  - [Function Description](#function-description)
  - [Usage](#usage)
  - [Parameters](#parameters)
  - [Examples](#examples)
    - [Example 1: Using Default Parameters](#example-1-using-default-parameters)
    - [Example 2: Using Custom Parameters](#example-2-using-custom-parameters)
  - [License](#license)

## Function Description

The `gaussian_filter` function generates a 2D Gaussian filter with the following parameters:

- `kernel_size`: Size of the filter kernel.
- `sigma_x`: Standard deviation in the x-direction (default is 1).
- `sigma_y`: Standard deviation in the y-direction (default is 1).
- `mu_x`: Mean (center) of the Gaussian in the x-direction (default is 0).
- `mu_y`: Mean (center) of the Gaussian in the y-direction (default is 0).

The function creates a grid of points, computes the Gaussian filter based on the provided parameters, and returns the filter as a NumPy array.

## Usage

To use the `gaussian_filter` function in your Python code, you need to import it as follows:

```python
from your_module import gaussian_filter
```

Where `your_module` is the name of the module where you've defined the `gaussian_filter` function.

## Parameters

- `kernel_size` (int): Size of the filter kernel.
- `sigma_x` (float, optional): Standard deviation in the x-direction (default is 1).
- `sigma_y` (float, optional): Standard deviation in the y-direction (default is 1).
- `mu_x` (float, optional): Mean (center) of the Gaussian in the x-direction (default is 0).
- `mu_y` (float, optional): Mean (center) of the Gaussian in the y-direction (default is 0).

## Examples

### Example 1: Using Default Parameters

```python
kernel_size = 3
gaussian = gaussian_filter(kernel_size=3)
print("Gaussian filter of {} X {}:".format(kernel_size, kernel_size))
print(gaussian)
```

### Example 2: Using Custom Parameters

```python
kernel_size = 3
sigma_x = 2
sigma_y = 1
mu_x = 0.5
mu_y = -0.5
gaussian = gaussian_filter(kernel_size=kernel_size, sigma_x=sigma_x, sigma_y=sigma_y, mu_x=mu_x, mu_y=mu_y)
print("Gaussian filter with custom parameters:")
print(gaussian)
```

## License

This code is available under the [MIT License](LICENSE). You are free to use, modify, and distribute it in accordance with the terms of the license.

