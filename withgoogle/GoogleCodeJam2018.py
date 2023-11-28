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

class SavingTheUniverseAgain: # Q.1
    '''
    Q.1
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

class TroubleSort: # Q.2
    '''
    Q.2
    https://codingcompetitions.withgoogle.com/codejam/round/00000000000000cb/00000000000079cb
    '''
    def solve(self, nums):
        evens = sorted(nums[i] for i in range(len(nums)) if i % 2 == 0)
        odds = sorted(nums[i] for i in range(len(nums)) if i % 2 == 1)
        prev_num = evens[0]
        for i in range(1, len(nums)):
            index = i // 2
            if i % 2 == 0:
                if evens[index] < prev_num:
                    return i - 1
                prev_num = evens[index]
            else:
                if odds[index] < prev_num:
                    return i - 1
                prev_num = odds[index]
        return 'OK'
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            N = int(input())
            nums = list(map(int, input().split(' ')))
            solution = self.solve(nums)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

class WaffleChoppers: # 1A.1
    def solve(self, waffle: list, h_cuts: int, v_cuts: int):
        rows = len(waffle)
        cols = len(waffle[0])
        total_chip_count = 0
        row_chips = [0] * rows
        column_chips = [0] * cols
        for row in range(rows):
            chip_count = 0
            for col in range(cols):
                if waffle[row][col] == '@':
                    column_chips[col] += 1
                    chip_count += 1
            row_chips[row] = chip_count
            total_chip_count += chip_count
        result = True
        if total_chip_count > 0:
            diner_count = (h_cuts + 1) * (v_cuts + 1)
            chips_per_piece = int(chip_count // diner_count)
            if total_chip_count != chips_per_piece * diner_count:
                result = False
            else:
                chips_per_row = chips_per_piece * (v_cuts + 1)
                chips_per_col = chips_per_piece * (h_cuts + 1)
                h_partitions = [0]
                # Check rows
                row_chip_count = 0
                for row in range(rows):
                    row_chip_count += row_chips[row]
                    if row_chip_count == chips_per_row:
                        row_chip_count = 0
                        h_partitions.append(row + 1)
                if len(h_partitions) - 2 != h_cuts:
                    result = False
                # Check columns
        return result
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            rows, cols, h_cuts, v_cuts = tuple(map(int, input().split(' ')))
            waffle = []
            for _ in range(rows):
                waffle.append(input())
            solution = 'IMPOSSIBLE'
            if self.solve(waffle, h_cuts, v_cuts):
                solution = 'POSSIBLE'
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
    python GoogleCodeJam2018.py 1A.1 < inputs/WaffleChoppers.in
    '''
    solvers = {
        'Q.1': (SavingTheUniverseAgain, 'Saving The Universe Again'),
        'Q.2': (TroubleSort, 'Trouble Sort'),
        '1A.1': (WaffleChoppers, 'Waffle Choppers'),
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