'''
Created on March 19, 2022

@author: Sestren
'''
import argparse
import collections
import copy
from functools import lru_cache
import heapq
import itertools
import math
import random
import sys
from typing import Dict, List, Set, Tuple

class StandingOvation: # 2015.Q.1
    '''
    2015.Q.1
    https://codingcompetitions.withgoogle.com/codejam/round/0000000000433515/0000000000433738
    Minimum people to invite to get everyone to clap
    '''
    def is_standing_ovation(self, shyness_counts: tuple) -> bool:
        standing_ovation = False
        applause_count = 0
        for shyness, count in enumerate(shyness_counts):
            if applause_count >= shyness:
                applause_count += count
        if applause_count >= sum(shyness_counts):
            standing_ovation = True
        result = standing_ovation
        return result

    def solve_slowly(self, shyness_counts: tuple):
        max_shyness = len(shyness_counts) + 1
        result = max_shyness
        left = 0
        right = max_shyness
        while left < right:
            mid = left + (right - left) // 2
            modified_shyness_counts = list(shyness_counts)
            modified_shyness_counts[0] += mid
            if self.is_standing_ovation(modified_shyness_counts):
                right = mid
            else:
                left = mid + 1
        result = left
        return result

    def solve(self, shyness_counts: tuple):
        curr_ovation = 0
        for shyness, count in enumerate(shyness_counts):
            curr_ovation = max(curr_ovation, shyness)
            curr_ovation += count
        result = curr_ovation - sum(shyness_counts)
        return result
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            parts = input().split(' ')
            max_shyness = int(parts[0])
            shyness_counts = list(map(int, list(parts[1])))
            solution = self.solve(shyness_counts)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

class Template:
    def solve(self, raw_input):
        result = len(raw_input)
        return result
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            raw_input = input()
            solution = self.solve(raw_input)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

if __name__ == '__main__':
    '''
    Usage
    python GoogleCodeJam2015.py 2015.Q.1 < inputs/SolverA.in
    '''
    solvers = {
        '2015.Q.1': (StandingOvation, 'StandingOvation'),
        # 'Solver': (Solver, 'Solver'),
        }
    parser = argparse.ArgumentParser()
    parser.add_argument('problem', help='Solve for a given problem', type=str)
    args = parser.parse_args()
    problem = args.problem
    solver = solvers[problem][0]()
    print(f'Solution for "{problem}" ({solvers[problem][1]})')
    solution = solver.main()
    #print(f'  Answer:', solution)

# Template for Submission page
'''
import collections

class Solver:
    def solve(self, raw_input):
        result = len(raw_input)
        return result
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            raw_input = input()
            solution = self.solve(raw_input)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

if __name__ == '__main__':
    solver = Solver()
    solver.main()
'''

# Usage for Interactive Problems
'''
-- python judges/DatBae.py 0 python solvers/DatBae.py

import os
os.system('python filename.py')
'''