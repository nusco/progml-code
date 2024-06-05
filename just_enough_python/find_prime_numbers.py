# Import the square root function
from math import sqrt


# Return True if the argument is a prime number, False otherwise
def is_prime(n):
    # n is prime if it cannot be divided evenly by any number in
    # the range from 2 to the square root of n.
    max_check_value = sqrt(n)
    # range(a, b) goes from a included to b excluded. Both a and b
    # must be integers, so we convert max_check_value to an integer
    # with the in-built int() function.
    for x in range(2, int(max_check_value) + 1):
        # Check the remainder of the integer division of n by x
        if n % x == 0:
            return False  # n can be divided by x, so it's not prime
    return True  # n is prime


MAX_RANGE = 100
primes = []
print("Computing the prime numbers from 2 to %d:" % MAX_RANGE)
for n in range(2, MAX_RANGE):
    if is_prime(n):
        primes.append(n)
print(primes)
