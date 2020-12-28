'''
Created on Dec 30, 2019

@author: Sestren
'''
import argparse
import collections
import heapq
import itertools
import math

class MultiplesOf3And5:
    def main(self, max_num: int=1_000) -> int:
        result = sum(
            n for n in
            range(max_num) if
            n % 3 == 0 or n % 5 == 0
            )
        return result

class EvenFibonacciNumbers:
    def gen_fibonacci(self, max_num: int) -> int:
        a, b = 0, 1
        for _ in range(max_num):
            if a > max_num:
                break
            yield a
            a, b = b, a + b
    
    def main(self, max_num: int=4_000_000) -> int:
        result = sum(
            n for n in
            self.gen_fibonacci(max_num) if
            n <= max_num and n % 2 == 0
            )
        return result

class LargestPrimeFactor:
    def get_largest_prime_factor(self, target: int) -> int:
        i = 2
        largest_prime_factor = 1
        while i <= target:
            while target % i == 0:
                largest_prime_factor = i
                target //= i
            i += 1
        result = largest_prime_factor
        return result
        
    def main(self, target: int=600_851_475_143) -> int:
        result = self.get_largest_prime_factor(target)
        return result

if __name__ == '__main__':
    '''
    Usage
    python ProjectEuler.py 2
    '''
    solvers = {
        1: (
            MultiplesOf3And5,
            'Multiples of 3 and 5',
            'What is the sum of all the multiples of 3 or 5 below 1000?',
            ),
        2: (
            EvenFibonacciNumbers,
            'Even Fibonacci Numbers',
            'What is the sum of all even Fibonacci numbers whose values do '\
            'not exceed four million?',
            ),
        3: (
            LargestPrimeFactor,
            'Largest Prime Factor',
            'What is the largest prime factor of the number 600851475143?',
            ),
        }
    parser = argparse.ArgumentParser()
    parser.add_argument('problem', help='Solve for a given problem', type=int)
    args = parser.parse_args()
    problem = args.problem
    solver = solvers[problem][0]()
    solution = solver.main()
    print(f'Solution for Problem {problem}:', solvers[problem][1])
    print(solvers[problem][2])
    print(f'  Answer:', solution)
