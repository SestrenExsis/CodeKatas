'''
Created on 2022-11-30

@author: Sestren
'''
import argparse
import collections
import copy
import heapq
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

class Day01: # Calorie Counting
    '''
    https://adventofcode.com/2022/day/1
    '''
    def get_elves(self, raw_input_lines: List[str]):
        elves = [[]]
        for raw_input_line in raw_input_lines:
            if len(raw_input_line) < 1:
                elves.append([])
            else:
                elves[-1].append(int(raw_input_line))
        result = elves
        return result
    
    def solve(self, elves):
        result = max(sum(calories) for calories in elves)
        return result
    
    def solve2(self, elves):
        heap = []
        for calories in elves:
            heapq.heappush(heap, -1 * sum(calories))
        top_calories = []
        top_calories.append(-1 * heapq.heappop(heap))
        top_calories.append(-1 * heapq.heappop(heap))
        top_calories.append(-1 * heapq.heappop(heap))
        result = sum(top_calories)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        elves = self.get_elves(raw_input_lines)
        solutions = (
            self.solve(elves),
            self.solve2(elves),
            )
        result = solutions
        return result

if __name__ == '__main__':
    '''
    Usage
    python AdventOfCode2022.py 1 < inputs/2022day01.in
    '''
    solvers = {
        1: (Day01, 'Calorie Counting'),
    #     2: (Day02, 'Day02'),
    #     3: (Day03, 'Day03'),
    #     4: (Day04, 'Day04'),
    #     5: (Day05, 'Day05'),
    #     6: (Day06, 'Day06'),
    #     7: (Day07, 'Day07'),
    #     8: (Day08, 'Day08'),
    #     9: (Day09, 'Day09'),
    #    10: (Day10, 'Day10'),
    #    11: (Day11, 'Day11'),
    #    12: (Day12, 'Day12'),
    #    13: (Day13, 'Day13'),
    #    14: (Day14, 'Day14'),
    #    15: (Day15, 'Day15'),
    #    16: (Day16, 'Day16'),
    #    17: (Day17, 'Day17'),
    #    18: (Day18, 'Day18'),
    #    19: (Day19, 'Day19'),
    #    20: (Day20, 'Day20'),
    #    21: (Day21, 'Day21'),
    #    22: (Day22, 'Day22'),
    #    23: (Day23, 'Day23'),
    #    24: (Day24, 'Day24'),
    #    25: (Day25, 'Day25'),
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
