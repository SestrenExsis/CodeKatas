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

class Day01: # Not Quite Lisp
    '''
    Not Quite Lisp
    https://adventofcode.com/2015/day/1
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

if __name__ == '__main__':
    '''
    Usage
    python AdventOfCode2015.py 1 < inputs/2015day01.in
    '''
    solvers = {
        1: (Day01, 'Not Quite Lisp'),
    #     2: (Day02, 'Inventory Management System'),
    #     3: (Day03, 'No Matter How You Slice It'),
    #     4: (Day04, 'Repose Record'),
    #     5: (Day05, 'Alchemical Reduction'),
    #     6: (Day06, 'Chronal Coordinates'),
    #     7: (Day07, 'The Sum of Its Parts'),
    #     8: (Day08, 'Memory Maneuver'),
    #     9: (Day09, 'Marble Mania'),
    #    10: (Day10, 'The Stars Align'),
    #    11: (Day11, 'Chronal Charge'),
    #    12: (Day12, 'Subterranean Sustainability'),
    #    13: (Day13, 'Mine Cart Madness'),
    #    14: (Day14, 'Chocolate Charts'),
    #    15: (Day15, 'Beverage Bandits'),
    #    16: (Day16, 'Chronal Classification'),
    #    17: (Day17, 'Reservoir Research'),
    #    18: (Day18, 'Settlers of The North Pole'),
    #    19: (Day19, 'Go With The Flow'),
    #    20: (Day20, 'A Regular Map'),
    #    21: (Day21, 'Chronal Conversion'),
    #    22: (Day22, 'Mode Maze'),
    #    23: (Day23, 'Experimental Emergency Teleportation'),
    #    24: (Day24, 'Immune System Simulator 20XX'),
    #    25: (Day25, 'Four-Dimensional Adventure'),
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
