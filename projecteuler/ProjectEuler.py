'''
Created on Dec 30, 2019

@author: Sestren
'''
import argparse
import collections
import heapq
import itertools
import math

class Problem_1: # Multiples of 3 and 5
    '''
    Multiples of 3 and 5
    https://projecteuler.net/problem=1
    '''
    def main(self, max_num: int=1_000) -> int:
        result = sum(
            n for n in
            range(max_num) if
            n % 3 == 0 or n % 5 == 0
            )
        return result

class Problem_2: # Even Fibonacci numbers
    '''
    Even Fibonacci numbers
    https://projecteuler.net/problem=2
    '''
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

class Problem_3: # Largest prime factor
    '''
    Largest prime factor
    https://projecteuler.net/problem=3
    '''
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

class Problem_4: # Largest palindrome product
    '''
    Largest palindrome product
    https://projecteuler.net/problem=4
    '''
    def gen_palindrome_products(self, digits: int) -> int:
        min_num = 10 ** (digits - 1)
        max_num = 10 ** digits - 1
        for a in range(min_num, max_num + 1):
            for b in range(min_num, max_num + 1):
                num = a * b
                str_num = str(num)
                N = len(str_num)
                palindrome_ind = True
                for i in range(N // 2):
                    if str_num[i] != str_num[-i - 1]:
                        palindrome_ind = False
                        break
                if palindrome_ind:
                    yield num
    
    def main(self) -> int:
        result = max(self.gen_palindrome_products(3))
        return result

class Problem_5: # Smallest multiple
    '''
    Smallest multiple
    https://projecteuler.net/problem=5
    '''
    def solve(self, max_num: int=20) -> int:
        lcm = 1
        for factor in range(2, max_num + 1):
            lcm *= factor
        for candidate in range(max_num, lcm + 1, max_num):
            # if candidate % 10_000 == 0:
            #     print(candidate)
            for factor in range(1, max_num):
                if candidate % factor != 0:
                    break
            else:
                lcm = candidate
                break
        result = lcm
        return result
    
    def main(self) -> int:
        result = self.solve(20)
        return result

class Template: # Template
    '''
    Template
    https://projecteuler.net/problem=?
    '''
    def solve(self) -> int:
        return -1
    
    def main(self) -> int:
        result = self.solve()
        return result

if __name__ == '__main__':
    '''
    Usage
    python ProjectEuler.py 4
    '''
    solvers = {
        1: (
            Problem_1, 'Multiples of 3 and 5',
            'What is the sum of all the multiples of 3 or 5 below 1000?',
            ),
        2: (
            Problem_2, 'Even Fibonacci numbers',
            'What is the sum of all even Fibonacci numbers whose values do '\
            'not exceed four million?',
            ),
        3: (
            Problem_3, 'Largest prime factor',
            'What is the largest prime factor of the number 600851475143?',
            ),
        4: (
            Problem_4, 'Largest palindrome product',
            'What is the largest palindrome made from the product of two '\
            '3-digit numbers?',
            ),
        5: (
            Problem_5, 'Smallest multiple',
            'What is the smallest positive number that is evenly divisible '\
            'by all of the numbers from 1 to 20?',
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
