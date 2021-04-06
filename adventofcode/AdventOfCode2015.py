'''
Created 2021-04-04

@author: Sestren
'''
import argparse
import collections
import copy
import datetime
import functools
import heapq
import operator
import re
import time
from typing import Dict, List, Set, Tuple
    
def get_raw_input_lines() -> list:
    raw_input_lines = []
    while True:
        try:
            raw_input_line = input()
            raw_input_lines.append(raw_input_line)
        except EOFError:
            break
        except StopIteration:
            break
        except KeyboardInterrupt:
            break
    return raw_input_lines

class Template: # Template
    '''
    Template
    https://adventofcode.com/2015/day/?
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, parsed_input):
        result = len(parsed_input)
        return result
    
    def solve2(self, parsed_input):
        result = len(parsed_input)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        parsed_input = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(parsed_input),
            self.solve2(parsed_input),
            )
        result = solutions
        return result

class Day02: # I Was Told There Would Be No Math
    '''
    I Was Told There Would Be No Math
    https://adventofcode.com/2015/day/2
    '''
    def get_dimensions(self, raw_input_lines: List[str]):
        dimensions = []
        for raw_input_line in raw_input_lines:
            # Dimension = (Length, Width, Height)
            dimension = tuple(map(int, raw_input_line.split('x')))
            dimensions.append(dimension)
        result = dimensions
        return result
    
    def solve(self, dimensions: List[Tuple[int]]):
        wrapping_paper = []
        for dimension in dimensions:
            l, w, h = dimension
            smallest_side = min((l * w, w * h, h * l))
            needed = 2 * l * w + 2 * w * h + 2 * h * l + smallest_side
            wrapping_paper.append(needed)
        result = sum(wrapping_paper)
        return result
    
    def solve2(self, dimensions: List[Tuple[int]]):
        ribbon = []
        for dimension in dimensions:
            l, w, h = dimension
            perimeters = [
                2 * (l + w),
                2 * (w + h),
                2 * (h + l),
            ]
            volume = l * w * h
            needed = min(perimeters) + volume
            ribbon.append(needed)
        result = sum(ribbon)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        dimensions = self.get_dimensions(raw_input_lines)
        solutions = (
            self.solve(dimensions),
            self.solve2(dimensions),
            )
        result = solutions
        return result

class Day01: # Not Quite Lisp
    '''
    Not Quite Lisp
    https://adventofcode.com/2015/day/1
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = raw_input_lines[0]
        return result
    
    def solve(self, parsed_input):
        floor = 0
        for char in parsed_input:
            if char == '(':
                floor += 1
            elif char == ')':
                floor -= 1
        result = floor
        return result
    
    def solve2(self, parsed_input):
        floor = 0
        position = 0
        for position, char in enumerate(parsed_input, start=1):
            if char == '(':
                floor += 1
            elif char == ')':
                floor -= 1
            if floor == -1:
                result = position
                break
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        parsed_input = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(parsed_input),
            self.solve2(parsed_input),
            )
        result = solutions
        return result

if __name__ == '__main__':
    '''
    Usage
    python AdventOfCode2015.py 2 < inputs/2015day02.in
    '''
    solvers = {
        1: (Day01, 'Not Quite Lisp'),
        2: (Day02, 'I Was Told There Would Be No Math'),
    #     3: (Day03, '???'),
    #     4: (Day04, '???'),
    #     5: (Day05, '???'),
    #     6: (Day06, '???'),
    #     7: (Day07, '???'),
    #     8: (Day08, '???'),
    #     9: (Day09, '???'),
    #    10: (Day10, '???'),
    #    11: (Day11, '???'),
    #    12: (Day12, '???'),
    #    13: (Day13, '???'),
    #    14: (Day14, '???'),
    #    15: (Day15, '???'),
    #    16: (Day16, '???'),
    #    17: (Day17, '???'),
    #    18: (Day18, '???'),
    #    19: (Day19, '???'),
    #    20: (Day20, '???'),
    #    21: (Day21, '???'),
    #    22: (Day22, '???'),
    #    23: (Day23, '???'),
    #    24: (Day24, '???'),
    #    25: (Day25, '???'),
        }
    parser = argparse.ArgumentParser()
    parser.add_argument('day', help='Solve for a given day', type=int)
    args = parser.parse_args()
    day = args.day
    solver = solvers[day][0]()
    solutions = solver.main()
    print(f'Solutions for Day {day}:', solvers[day][1])
    print(f'  Part 1:', solutions[0])
    print(f'  Part 2:', solutions[1])
