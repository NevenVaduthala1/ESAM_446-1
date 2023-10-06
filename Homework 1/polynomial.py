import math
import numpy as np
import scipy as sp
from sympy import gcd
from sympy import gcd, Poly, symbols, simplify
import re

x = symbols('x')  # Define the variable for the polynomial

class Polynomial:
    def __init__(self, coefficients):
        self.coefficients = np.array(coefficients, dtype=int)
        self.order = len(coefficients) - 1

    @staticmethod
    def from_string(s):
        # Adding * between coefficient and variable if it's missing
        s = re.sub(r'(\d)([a-z])', r'\1*\2', s)

        # Creating a SymPy polynomial from the string
        polynomial = Poly(s, x)

        # Extracting the coefficients as a list and reversing it to have them in the correct order
        coefficients = polynomial.all_coeffs()[::-1]

        # If the coefficients list is empty, it means the polynomial is a zero polynomial
        if len(coefficients) == 0:
            return Polynomial([0])  # Returning a zero polynomial

        # Converting coefficients to integers and creating a Polynomial object
        coefficients = [int(coeff) for coeff in coefficients]
        return Polynomial(coefficients)




    def __repr__(self):
        terms = []
        for i, coef in reversed(list(enumerate(self.coefficients))):
            if coef:
                if i == 0:
                    terms.append(str(coef))
                elif i == 1:
                    term = f"{coef}*x" if coef != 1 else "x"
                    term = term if coef != -1 else "-x"
                    terms.append(term)
                else:
                    term = f"{coef}*x^{i}" if coef != 1 else f"x^{i}"
                    term = term if coef != -1 else f"-x^{i}"
                    terms.append(term)

        return ' + '.join(terms).replace('+ -', '- ')




    def __add__(self, other):
        max_order = max(self.order, other.order)
        result = np.zeros(max_order + 1, dtype=int)
        for i in range(max_order + 1):
            result[i] = self.coefficients[i] if i <= self.order else 0
            result[i] += other.coefficients[i] if i <= other.order else 0
        return Polynomial(result)

    def __sub__(self, other):
        max_order = max(self.order, other.order)
        result = np.zeros(max_order + 1, dtype=int)
        for i in range(max_order + 1):
            result[i] = self.coefficients[i] if i <= self.order else 0
            result[i] -= other.coefficients[i] if i <= other.order else 0
        return Polynomial(result)

    def __mul__(self, other):
        order = self.order + other.order
        result = np.zeros(order + 1, dtype=int)
        for i in range(self.order + 1):
            for j in range(other.order + 1):
                result[i+j] += self.coefficients[i] * other.coefficients[j]
        return Polynomial(result)

    def __eq__(self, other):
        return np.array_equal(self.coefficients, other.coefficients)

    def to_sympy_poly(self):
        return Poly(self.coefficients[::-1], x)

    def __truediv__(self, other):
        if other.is_zero():
            raise ZeroDivisionError("Cannot divide by zero polynomial")
        return RationalPolynomial(self, other)

    def is_zero(self):
        return np.all(self.coefficients == 0)



x = symbols('x')

class RationalPolynomial:
    def __init__(self, numerator, denominator):
        self.numerator = numerator  # These are now SymPy expressions
        self.denominator = denominator
        self.simplify()

    def simplify(self):
        gcd_poly = gcd(Poly(self.numerator, x), Poly(self.denominator, x))
        self.numerator = simplify(self.numerator / gcd_poly.as_expr())
        self.denominator = simplify(self.denominator / gcd_poly.as_expr())

    @staticmethod
    def from_string(s):
        numerator_str, denominator_str = map(str.strip, s.split("/"))
        numerator = simplify(numerator_str.strip("()"))
        denominator = simplify(denominator_str.strip("()"))
        return RationalPolynomial(numerator, denominator)

    def __repr__(self):
        num_str = str(self.numerator).replace('**', '^')
        den_str = str(self.denominator).replace('**', '^')
        return f"({num_str})/({den_str})"

    def __add__(self, other):
        # Implementation of the addition operation
        numerator = self.numerator * other.denominator + other.numerator * self.denominator
        denominator = self.denominator * other.denominator
        return RationalPolynomial(numerator, denominator)

    def __sub__(self, other):
        # Implementation of the subtraction operation
        numerator = self.numerator * other.denominator - other.numerator * self.denominator
        denominator = self.denominator * other.denominator
        return RationalPolynomial(numerator, denominator)

    def __mul__(self, other):
        # Implementation of the multiplication operation
        return RationalPolynomial(self.numerator * other.numerator, self.denominator * other.denominator)

    def __truediv__(self, other):
        # Implementation of the division operation
        return RationalPolynomial(self.numerator * other.denominator, self.denominator * other.numerator)

    def __eq__(self, other):
        # Implementation of the equality check operation
        return self.numerator * other.denominator == self.denominator * other.numerator






a = Polynomial.from_string("-4 + x^2")
b = Polynomial.from_string("x^2 - 4")
c = Polynomial.from_string("-2 - x + 9*x^2")
r1 = RationalPolynomial.from_string("(2 + x)/(-1 + x + 2*x^3)")
r2 = RationalPolynomial.from_string("(x + 1)/(x^2 - 1)")

print(r1)
print(r2)
