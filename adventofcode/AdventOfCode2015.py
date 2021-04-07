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
import hashlib
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

class Day04: # The Ideal Stocking Stuffer
    '''
    The Ideal Stocking Stuffer
    https://adventofcode.com/2015/day/4
    '''
    def get_secret_key(self, raw_input_lines: List[str]):
        result = raw_input_lines[0]
        return result
    
    def solve(self, secret_key):
        num = 0
        while True:
            algorithm = hashlib.md5()
            key = secret_key + str(num)
            algorithm.update(key.encode('utf-8'))
            if algorithm.hexdigest()[:5] == '00000':
                break
            num += 1
        result = num
        return result
    
    def solve2(self, secret_key):
        num = 0
        while True:
            algorithm = hashlib.md5()
            key = secret_key + str(num)
            algorithm.update(key.encode('utf-8'))
            if algorithm.hexdigest()[:6] == '000000':
                break
            num += 1
        result = num
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        secret_key = self.get_secret_key(raw_input_lines)
        solutions = (
            self.solve(secret_key),
            self.solve2(secret_key),
            )
        result = solutions
        return result

class Day03: # Perfectly Spherical Houses in a Vacuum
    '''
    Perfectly Spherical Houses in a Vacuum
    https://adventofcode.com/2015/day/3
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = raw_input_lines[0]
        return result
    
    def solve(self, parsed_input):
        row = 0
        col = 0
        houses = set()
        for char in parsed_input:
            if char == '^':
                row -= 1
            elif char == '>':
                col += 1
            elif char == 'v':
                row += 1
            elif char == '<':
                col -= 1
            houses.add((row, col))
        result = len(houses)
        return result
    
    def solve2(self, parsed_input):
        santa_id = 0
        santas = [
            (0, 0),
            (0, 0),
            ]
        houses = set()
        for char in parsed_input:
            pos = santas[santa_id]
            if char == '^':
                santas[santa_id] = (pos[0] - 1, pos[1])
            elif char == 'v':
                santas[santa_id] = (pos[0] + 1, pos[1])
            elif char == '<':
                santas[santa_id] = (pos[0], pos[1] - 1)
            elif char == '>':
                santas[santa_id] = (pos[0], pos[1] + 1)
            houses.add(santas[santa_id])
            santa_id = 1 - santa_id
        result = len(houses)
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
    python AdventOfCode2015.py 4 < inputs/2015day04.in
    '''
    solvers = {
        1: (Day01, 'Not Quite Lisp'),
        2: (Day02, 'I Was Told There Would Be No Math'),
        3: (Day03, 'Perfectly Spherical Houses in a Vacuum'),
        4: (Day04, 'The Ideal Stocking Stuffer'),
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
