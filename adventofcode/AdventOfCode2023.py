'''
Created on 2023-11-28

@author: Sestren
'''
import argparse
import collections
import copy
import functools
import heapq
import itertools
import random
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
    https://adventofcode.com/2022/day/?
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

class Day01: # Trebuchet?!
    '''
    https://adventofcode.com/2023/day/1
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, parsed_input):
        calibration_values = []
        for row_data in parsed_input:
            calibration_value = 0
            for char in row_data:
                try:
                    value = int(char)
                    calibration_value = 10 * value
                    break
                except ValueError:
                    pass
            for char in reversed(row_data):
                try:
                    value = int(char)
                    calibration_value += value
                    break
                except ValueError:
                    pass
            calibration_values.append(calibration_value)
        result = sum(calibration_values)
        return result
    
    def solve2(self, parsed_input):
        numbers = {
            'one': 1,
            'two': 2,
            'three': 3,
            'four': 4,
            'five': 5,
            'six': 6,
            'seven': 7,
            'eight': 8,
            'nine': 9,
        }
        calibration_values = []
        for row_data in parsed_input:
            parts = []
            for i in range(len(row_data)):
                try:
                    value = int(row_data[i])
                    parts.append(value)
                    break
                except ValueError:
                    for number in sorted(numbers, key=len):
                        if row_data[i:].startswith(number):
                            value = numbers[number]
                            parts.append(value)
                            break
                    if len(parts) > 0:
                        break
            for i in reversed(range(len(row_data))):
                try:
                    value = int(row_data[i])
                    parts.append(value)
                    break
                except ValueError:
                    for number in sorted(numbers, key=len):
                        if row_data[i:].startswith(number):
                            value = numbers[number]
                            parts.append(value)
                            break
                    if len(parts) > 1:
                        break
            calibration_values.append(10 * parts[0] + parts[1])
        result = sum(calibration_values)
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
    python AdventOfCode2023.py 1 < inputs/2023day01.in
    '''
    solvers = {
        1: (Day01, 'Trebuchet?!'),
    #     2: (Day02, 'Unknown'),
    #     3: (Day03, 'Unknown'),
    #     4: (Day04, 'Unknown'),
    #     5: (Day05, 'Unknown'),
    #     6: (Day06, 'Unknown'),
    #     7: (Day07, 'Unknown'),
    #     8: (Day08, 'Unknown'),
    #     9: (Day09, 'Unknown'),
    #    10: (Day10, 'Unknown'),
    #    11: (Day11, 'Unknown'),
    #    12: (Day12, 'Unknown'),
    #    13: (Day13, 'Unknown'),
    #    14: (Day14, 'Unknown'),
    #    15: (Day15, 'Unknown'),
    #    16: (Day16, 'Unknown'),
    #    17: (Day17, 'Unknown'),
    #    18: (Day18, 'Unknown'),
    #    19: (Day19, 'Unknown'),
    #    20: (Day20, 'Unknown'),
    #    21: (Day21, 'Unknown'),
    #    22: (Day22, 'Unknown'),
    #    23: (Day23, 'Unknown'),
    #    24: (Day24, 'Unknown'),
    #    25: (Day25, 'Unknown'),
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
