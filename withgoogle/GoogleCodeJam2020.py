'''
Created on Jan 4, 2021

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

class Vestigium:
    '''
    https://codingcompetitions.withgoogle.com/codejam/round/000000000019fd27/000000000020993c
    '''
    def solve(self, matrix):
        N = len(matrix)
        triangle_sum = (N * (N + 1)) // 2
        trace = sum(matrix[i][i] for i in range(N))
        row_repeat_count = sum(
            1 for
            row_data in matrix if
            sum(row_data) != triangle_sum or len(set(row_data)) != N
            )
        col_repeat_count = 0
        for col in range(N):
            col_sum = sum(matrix[row][col] for row in range(N))
            if any((
                col_sum != triangle_sum,
                len(set(matrix[i][col] for i in range(N))) != N,
                )):
                col_repeat_count += 1
        result = (trace, row_repeat_count, col_repeat_count)
        return result
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            matrix_size = int(input())
            matrix = [
                list(map(int, input().split(' '))) for
                _ in range(matrix_size)
                ]
            (trace, row_repeat_count, col_repeat_count) = self.solve(matrix)
            output_row = 'Case #{}: {} {} {}'.format(
                test_id,
                trace,
                row_repeat_count,
                col_repeat_count,
                )
            output.append(output_row)
            print(output_row)
        return output

class Solver:
    '''
import collections

if __name__ == '__main__':
    solver = Solver()
    solver.main()
    '''
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

'''
Usage for Interactive Problems
-- python judges/DatBae.py 0 python solvers/DatBae.py

import os
os.system('python filename.py')

import collections

class Solver:

if __name__ == '__main__':
    solver = Solver()
    solver.main()
'''

if __name__ == '__main__':
    '''
    Usage
    python GoogleCodeJam2020.py Template < inputs/Template.in
    '''
    solvers = {
        '2020.Q.1': (Vestigium2, 'Vestigium'),
        # '2020.Q.2': (NestingDepth, 'Nesting Depth'),
        # '2020.Q.3': (ParentingPartneringReturns, 'Parenting Partnering Returns'),
        # '2020.Q.4': (ESAbATAd, 'ESAbATAd'),
        # '2020.Q.5': (Indicium, 'Indicium'),
        # '2020.1A.1': (PatternMatching, 'Pattern Matching'),
        # '2020.1A.2': (PascalWalk, 'PascalWalk'),
        # '2020.1A.3': (Problem2020_1A_3, 'Problem2020_1A_3'),
        # '2020.1A.4': (Problem2020_1A_4, 'Problem2020_1A_4'),
        # '2020.1B.1': (Expogo, 'Expogo'),
        # '2020.1B.2': (BlindfoldedBullseye, 'Blindfolded Bullseye'),
        # '2020.1B.3': (JoinTheRanks, 'Join the Ranks'),
        # '2020.1C.1': (OverexcitedFan, 'Overexcited Fan'),
        # '2020.1C.2': (Overrandomized, 'Overrandomized'),
        # '2020.1C.3': (OversizedPancakeChoppers, 'Oversized Pancake Choppers'),
        # '2020.2.1': (IncrementalHouseOfPancakes, 'Incremental House of Pancakes'),
        # '2020.2.2': (SecurityUpdate, 'Security Update'),
        # '2020.2.3': (WormholeInOne, 'Wormhole in One'),
        # '2020.2.4': (EmacsPlusPlus, 'Emacs++'),
        'Solver': (Solver, 'Solver'),
        }
    parser = argparse.ArgumentParser()
    parser.add_argument('problem', help='Solve for a given problem', type=str)
    args = parser.parse_args()
    problem = args.problem
    solver = solvers[problem][0]()
    print(f'Solution for "{problem}" ({solvers[problem][1]})')
    solution = solver.main()
    #print(f'  Answer:', solution)
