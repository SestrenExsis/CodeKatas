'''
Created on March 19, 2022

@author: Sestren
'''
import argparse
from bdb import Breakpoint
import collections
import copy
import functools
import heapq
import itertools
import math
import random
import sys
from typing import Dict, List, Set, Tuple

class SavingTheUniverseAgain: # 2018.Q.1
    '''
    2018.Q.1
    https://codingcompetitions.withgoogle.com/codejam/round/00000000000000cb/0000000000007966
    - You can never reduce damage by changing a 'SC' to a 'CS'
    - The greedy approach would be to always hack at the location where it will
      save the most damage at that moment
    - This location is always going to be the rightmost position where a 'CS'
      is found in a given moment
    '''
    def solve(self, max_damage: int, initial_program: str):
        result = 'IMPOSSIBLE'
        charges = 0
        beams = []
        for char in initial_program:
            if char == 'C':
                charges += 1
                beams.append(0)
            elif char == 'S':
                power = 2 ** charges
                beams.append(power)
        hacks = 0
        while sum(beams) > max_damage:
            for i in reversed(range(1, len(beams))):
                if beams[i - 1] == 0 and beams[i] > 0:
                    beams[i - 1] = beams[i] // 2
                    beams[i] = 0
                    break
            else:
                break
            hacks += 1
        if sum(beams) <= max_damage:
            result = hacks
        return result
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            raw_input = input()
            A, program = raw_input.split(' ')
            max_damage = int(A)
            solution = self.solve(max_damage, program)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

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
    python GoogleCodeJam2018.py 2018.Q.1 < inputs/SolverA.in
    '''
    solvers = {
        'Q.1': (SavingTheUniverseAgain, 'Saving The Universe Again'),
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