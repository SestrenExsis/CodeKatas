'''
Created on Nov 24, 2020

@author: Sestren
'''
import argparse
from typing import List
    
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
    https://adventofcode.com/2020/day/?
    '''
    def get_parsed_input(self, raw_input_lines: List[str]) -> List[str]:
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, parsed_input: List[str]) -> int:
        result = 0
        return result
    
    def solve2(self, parsed_input: List[str]) -> str:
        result = 0
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

class Day02:
    '''
    ???
    https://adventofcode.com/2020/day/2
    '''
    def get_parsed_input(self, raw_input_lines: 'List'):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, parsed_input: 'List'):
        result = len(parsed_input)
        return result
    
    def solve2(self, parsed_input: 'List'):
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

class Day01:
    '''
    ???
    https://adventofcode.com/2020/day/1
    '''
    def get_parsed_input(self, raw_input_lines: 'List'):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, parsed_input: 'List'):
        result = len(parsed_input)
        return result
    
    def solve2(self, parsed_input: 'List'):
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
    python Solver.py 1 < day01.in
    '''
    solvers = {
        1: (Day01, 'Day 1'),
    #     2: (Day02, '???'),
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
