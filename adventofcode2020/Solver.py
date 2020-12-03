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

class Day03:
    '''
    Toboggan Trajectory
    https://adventofcode.com/2020/day/3
    '''
    def get_parsed_input(self, raw_input_lines: List[str]) -> List[str]:
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def get_hit_count(self, trees: List[str], right: int, down: int) -> int:
        row, col = 0, 0
        hit_count = 0
        rows = len(trees)
        cols = len(trees[0])
        while row < rows:
            if trees[row][col % cols] == '#':
                hit_count += 1
            col += right
            row += down
        result = hit_count
        return result
    
    def solve(self, parsed_input: List[str]) -> int:
        result = self.get_hit_count(parsed_input, 3, 1)
        return result
    
    def solve2(self, parsed_input: List[str]) -> str:
        result = self.get_hit_count(parsed_input, 1, 1)
        result *= self.get_hit_count(parsed_input, 3, 1)
        result *= self.get_hit_count(parsed_input, 5, 1)
        result *= self.get_hit_count(parsed_input, 7, 1)
        result *= self.get_hit_count(parsed_input, 1, 2)
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
    Password Philosophy
    https://adventofcode.com/2020/day/2
    '''
    def get_parsed_input(self, raw_input_lines: 'List'):
        result = []
        for raw_input_line in raw_input_lines:
            a, b, c = raw_input_line.split(' ')
            num_a, num_b = map(int, a.split('-'))
            char = b[0]
            password = c
            result.append((num_a, num_b, char, password))
        return result
    
    def solve(self, parsed_input: 'List'):
        valid_password_count = 0
        for min_count, max_count, char, password in parsed_input:
            char_count = password.count(char)
            if min_count <= char_count <= max_count:
                valid_password_count += 1
        result = valid_password_count
        return result
    
    def solve2(self, parsed_input: 'List'):
        valid_password_count = 0
        for i, j, char, password in parsed_input:
            check = 0
            if password[i - 1] == char:
                check += 1
            if password[j - 1] == char:
                check += 1
            if check == 1:
                valid_password_count += 1
        result = valid_password_count
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
    Report Repair
    https://adventofcode.com/2020/day/1
    '''
    def get_parsed_input(self, raw_input_lines: 'List'):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(int(raw_input_line))
        return result
    
    def solve(self, parsed_input: 'List'):
        result = -1
        seen = set()
        for num in parsed_input:
            target = 2020 - num
            if target in seen:
                result = num * target
            else:
                seen.add(num)
        return result
    
    def solve2(self, parsed_input: 'List'):
        nums = sorted(parsed_input)
        N = len(nums)
        for i in range(N):
            target = 2020 - nums[i]
            j = i + 1
            k = N - 1
            while j < k:
                total = nums[j] + nums[k]
                if total == target:
                    return nums[i] * nums[j] * nums[k]
                elif total < target:
                    j += 1
                elif total > target:
                    k -= 1
    
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
    python Solver.py 3 < day03.in
    '''
    solvers = {
        1: (Day01, 'Report Repair'),
        2: (Day02, 'Password Philosophy'),
        3: (Day03, 'Toboggan Trajectory'),
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
