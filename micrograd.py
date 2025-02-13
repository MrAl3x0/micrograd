import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
import unittest

class Value:
    """
    Represents a scalar value with automatic differentiation.
    Supports basic arithmetic operations and backpropagation.
    """
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data  # Scalar value
        self.grad = 0.0  # Gradient for backpropagation
        self._prev = set(_children)  # Track dependencies
        self._backward = lambda: None  # Placeholder for gradient computation
        self._op = _op  # Operation that produced this value
        self.label = label  # Optional label for visualization

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        """Addition operation."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), _op='+')
        
        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        
        out._backward = _backward
        return out

    def __neg__(self):
        """Negation operation."""
        return self * -1

    def __sub__(self, other):
        """Subtraction operation."""
        return self + (-other)

    def __mul__(self, other):
        """Multiplication operation."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), _op='*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out

    def __rmul__(self, other):
        """Reverse multiplication to support scalar * Value."""
        return self * other

    def tanh(self):
        """Hyperbolic tangent activation function."""
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        
        out._backward = _backward
        return out

    def __truediv__(self, other):
        """Division operation."""
        return self * other**-1

    def exp(self):
        """Exponential function."""
        out = Value(math.exp(self.data), (self,), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        
        out._backward = _backward
        return out

    def __pow__(self, other):
        """Power operation."""
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f'**{other}')
        
        def _backward():
            self.grad += other * self.data**(other-1) * out.grad
        
        out._backward = _backward
        return out

    def backward(self):
        """Backpropagation through the computational graph."""
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

class TestValue(unittest.TestCase):
    def test_addition(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        self.assertEqual(c.data, 5.0)
    
    def test_multiplication(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        self.assertEqual(c.data, 6.0)
    
    def test_subtraction(self):
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        self.assertEqual(c.data, 2.0)
    
    def test_tanh(self):
        a = Value(0.0)
        c = a.tanh()
        self.assertAlmostEqual(c.data, 0.0, places=5)
    
    def test_exponentiation(self):
        a = Value(2.0)
        c = a ** 3
        self.assertEqual(c.data, 8.0)

if __name__ == '__main__':
    unittest.main()
