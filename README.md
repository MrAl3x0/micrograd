![Random Dog](https://placedog.net/500)

# Micrograd

A minimalistic autograd engine implemented by following [Andrej Karpathy's Micrograd video](https://www.youtube.com/watch?v=VMj-3S1tku0&t=1s). This project builds an automatic differentiation system using computational graphs, supporting basic operations and backpropagation.

## Features
- Scalar-based autograd engine
- Supports operations like addition, multiplication, exponentiation, and tanh activation
- Computational graph visualization with Graphviz
- Simple multi-layer perceptron (MLP) implementation

## Installation
Ensure you have Python installed along with the required dependencies:

```bash
pip install -r requirements.txt
```

If using Conda, create an environment and install dependencies:

```bash
conda create --name micrograd-env python=3.10
conda activate micrograd-env
pip install numpy matplotlib graphviz
```

## Usage
### Basic Example
```python
from micrograd import Value

x = Value(2.0)
y = x * 3 + 1
y.backward()
print(x.grad)  # Should compute the gradient
```

### Visualizing Computational Graphs
```python
from micrograd import Value, draw_dot

x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
b = Value(6.88, label='b')

n = x1 * w1 + x2 * w2 + b
o = n.tanh()
o.backward()

draw_dot(o).render('graph', format='png')
```

## Testing
Run unit tests using:
```bash
python -m unittest discover
```

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m "Add new feature"`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a Pull Request

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments
This project was implemented by following [Andrej Karpathy's Micrograd video](https://www.youtube.com/watch?v=VMj-3S1tku0&t=1s).
