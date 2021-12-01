'''
Created on 2021-11-29

@author: Sestren
'''
import argparse
import collections
import copy
import functools
import operator
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
    https://adventofcode.com/2021/day/?
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

class Day01: # Sonar Sweep
    '''
    https://adventofcode.com/2021/day/1
    '''
    def get_sonar_sweeps(self, raw_input_lines: List[str]):
        sonar_sweeps = []
        for raw_input_line in raw_input_lines:
            sonar_sweeps.append(int(raw_input_line))
        result = sonar_sweeps
        return result
    
    def solve(self, sonar_sweeps):
        count = 0
        for i in range(1, len(sonar_sweeps)):
            if sonar_sweeps[i] > sonar_sweeps[i - 1]:
                count += 1
        result = count
        return result
    
    def solve2(self, sonar_sweeps):
        count = 0
        a, b, c = sonar_sweeps[0], sonar_sweeps[1], sonar_sweeps[2]
        for i in range(3, len(sonar_sweeps)):
            d = sonar_sweeps[i]
            if d > a:
                count += 1
            a, b, c = b, c, d
        result = count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        sonar_sweeps = self.get_sonar_sweeps(raw_input_lines)
        solutions = (
            self.solve(sonar_sweeps),
            self.solve2(sonar_sweeps),
            )
        result = solutions
        return result

if __name__ == '__main__':
    '''
    Usage
    python AdventOfCode2021.py 1 < inputs/2021day01.in
    '''
    solvers = {
        1: (Day01, 'Sonar Sweep'),
    #     2: (Day02, 'XXX'),
    #     3: (Day03, 'XXX'),
    #     4: (Day04, 'XXX'),
    #     5: (Day05, 'XXX'),
    #     6: (Day06, 'XXX'),
    #     7: (Day07, 'XXX'),
    #     8: (Day08, 'XXX'),
    #     9: (Day09, 'XXX'),
    #    10: (Day10, 'XXX'),
    #    11: (Day11, 'XXX'),
    #    12: (Day12, 'XXX'),
    #    13: (Day13, 'XXX'),
    #    14: (Day14, 'XXX'),
    #    15: (Day15, 'XXX'),
    #    16: (Day16, 'XXX'),
    #    17: (Day17, 'XXX'),
    #    18: (Day18, 'XXX'),
    #    19: (Day19, 'XXX'),
    #    20: (Day20, 'XXX'),
    #    21: (Day21, 'XXX'),
    #    22: (Day22, 'XXX'),
    #    23: (Day23, 'XXX'),
    #    24: (Day24, 'XXX'),
    #    25: (Day25, 'XXX'),
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
