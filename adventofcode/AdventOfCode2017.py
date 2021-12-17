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
    https://adventofcode.com/2017/day/?
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

class Day06: # Memory Reallocation
    '''
    https://adventofcode.com/2017/day/6
    '''
    def get_banks(self, raw_input_lines: List[str]):
        banks = []
        for index, element in enumerate(raw_input_lines[0].split()):
            banks.append(int(element))
        result = banks
        return result
    
    def solve(self, banks):
        seen = set()
        redistribution_count = 0
        configuration = tuple(banks)
        while configuration not in seen:
            seen.add(configuration)
            max_value = max(banks)
            index = 0
            while True:
                if banks[index] == max_value:
                    break
                index += 1
            blocks = banks[index]
            banks[index] = 0
            for i in range(1, blocks + 1):
                j = (index + i) % len(banks)
                banks[j] += 1
            configuration = tuple(banks)
            redistribution_count += 1
        result = redistribution_count
        return result
    
    def solve2(self, banks):
        cycle_start = None
        cycle_size = 0
        seen = set()
        redistribution_count = 0
        configuration = tuple(banks)
        while True:
            if configuration in seen and cycle_start is None:
                cycle_start = configuration
                cycle_size = 1
            elif cycle_start is not None and configuration == cycle_start:
                break
            else:
                cycle_size += 1
            seen.add(configuration)
            max_value = max(banks)
            index = 0
            while True:
                if banks[index] == max_value:
                    break
                index += 1
            blocks = banks[index]
            banks[index] = 0
            for i in range(1, blocks + 1):
                j = (index + i) % len(banks)
                banks[j] += 1
            configuration = tuple(banks)
            redistribution_count += 1
        result = cycle_size
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        banks = self.get_banks(raw_input_lines)
        solutions = (
            self.solve(banks),
            self.solve2(banks),
            )
        result = solutions
        return result

class Day05: # A Maze of Twisty Trampolines, All Alike
    '''
    https://adventofcode.com/2017/day/5
    '''
    def get_offsets(self, raw_input_lines: List[str]):
        offsets = {}
        for index, raw_input_line in enumerate(raw_input_lines):
            offset = int(raw_input_line)
            offsets[index] = offset
        result = offsets
        return result
    
    def solve(self, offsets):
        position = 0
        step_count = 0
        while 0 <= position < len(offsets):
            offset = offsets[position]
            offsets[position] += 1
            position += offset
            step_count += 1
        result = step_count
        return result
    
    def solve2(self, offsets):
        position = 0
        step_count = 0
        while 0 <= position < len(offsets):
            offset = offsets[position]
            if offset >= 3:
                offsets[position] -= 1
            else:
                offsets[position] += 1
            position += offset
            step_count += 1
        result = step_count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        offsets = self.get_offsets(raw_input_lines)
        solutions = (
            self.solve(copy.deepcopy(offsets)),
            self.solve2(copy.deepcopy(offsets)),
            )
        result = solutions
        return result

class Day04: # High-Entropy Passphrases
    '''
    https://adventofcode.com/2017/day/4
    '''
    def get_passphrases(self, raw_input_lines: List[str]):
        passphrases = []
        for raw_input_line in raw_input_lines:
            passphrase = tuple(raw_input_line.split(' '))
            passphrases.append(passphrase)
        result = passphrases
        return result
    
    def solve(self, passphrases):
        valid_passphrase_count = 0
        for passphrase in passphrases:
            if len(passphrase) == len(set(passphrase)):
                valid_passphrase_count +=1
        result = valid_passphrase_count
        return result
    
    def solve2(self, passphrases):
        valid_passphrase_count = 0
        for passphrase in passphrases:
            anagrams = set()
            for word in passphrase:
                anagram = ''.join(sorted(word))
                if anagram in anagrams:
                    break
                anagrams.add(anagram)
            else:
                valid_passphrase_count += 1
        result = valid_passphrase_count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        passphrases = self.get_passphrases(raw_input_lines)
        solutions = (
            self.solve(passphrases),
            self.solve2(passphrases),
            )
        result = solutions
        return result

class Day03: # Spiral Memory
    '''
    https://adventofcode.com/2017/day/3
    '''
    def solve(self, target_square):
        # Each square-shaped ring adds a predictable amount of numbers
        size = 1
        squares_per_layer = [1]
        total_squares = 1
        while total_squares < target_square:
            size += 2
            next_squares = size ** 2 - total_squares
            squares_per_layer.append(next_squares)
            total_squares += next_squares
        # On each layer of the square-shaped ring, the maximum distance 
        # ping pongs between size - 1 at the corners and half that distance
        # at the midpoint of each side
        max_distance = size - 1
        min_distance = max_distance // 2
        distance = max_distance
        direction = -1
        square = sum(squares_per_layer)
        while square > target_square:
            distance += direction
            if distance == min_distance:
                direction = 1
            elif distance == max_distance:
                direction = -1
            square -= 1
        result = distance
        return result
    
    def solve2(self, target_square):
        squares = {
            (0, 0): 1,
        }
        def get_value(row, col):
            if (row, col) not in squares:
                value = 0
                for (nrow, ncol) in (
                    (row - 1, col - 1),
                    (row - 1, col),
                    (row - 1, col + 1),
                    (row, col - 1),
                    (row, col + 1),
                    (row + 1, col - 1),
                    (row + 1, col),
                    (row + 1, col + 1),
                ):
                    if (nrow, ncol) in squares:
                        value += squares[(nrow, ncol)]
                squares[(row, col)] = value
            return squares[(row, col)]
        row, col = (0, 1)
        direction = 'UP'
        while True:
            get_value(row, col)
            if squares[(row, col)] > target_square:
                return squares[row, col]
            if abs(row) == abs(col): # Change direction at each corner
                if direction == 'UP':
                    direction = 'LEFT'
                elif direction == 'LEFT':
                    direction = 'DOWN'
                elif direction == 'DOWN':
                    direction = 'RIGHT'
                elif direction == 'RIGHT':
                    col += 1
                    get_value(row, col)
                    direction = 'UP'
            if direction == 'UP':
                row -= 1
            elif direction == 'LEFT':
                col -= 1
            elif direction == 'DOWN':
                row += 1
            elif direction == 'RIGHT':
                col += 1
    
    def main(self):
        assert self.solve(1) == 0
        assert self.solve(12) == 3
        assert self.solve(23) == 2
        assert self.solve(1024) == 31
        raw_input_lines = get_raw_input_lines()
        target_square = int(raw_input_lines[0])
        solutions = (
            self.solve(target_square),
            self.solve2(target_square),
            )
        result = solutions
        return result

class Day02: # Corruption Checksum
    '''
    https://adventofcode.com/2017/day/2
    '''
    def get_spreadsheet(self, raw_input_lines: List[str]):
        spreadsheet = []
        for raw_input_line in raw_input_lines:
            row_data = []
            for cell in raw_input_line.split('\t'):
                num = int(cell)
                row_data.append(num)
            spreadsheet.append(row_data)
        result = spreadsheet
        return result
    
    def solve(self, spreadsheet):
        checksum = 0
        for row_data in spreadsheet:
            diff = abs(max(row_data) - min(row_data))
            checksum += diff
        result = checksum
        return result
    
    def solve2(self, spreadsheet):
        total = 0
        for row_data in spreadsheet:
            nums = set()
            found_ind = False
            for num in sorted(row_data):
                for divisor in nums:
                    if num % divisor == 0:
                        total += num // divisor
                        found_ind = True
                        break
                if found_ind:
                    break
                nums.add(num)
        result = total
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        spreadsheet = self.get_spreadsheet(raw_input_lines)
        solutions = (
            self.solve(spreadsheet),
            self.solve2(spreadsheet),
            )
        result = solutions
        return result

class Day01: # Inverse Captcha
    '''
    https://adventofcode.com/2017/day/1
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = raw_input_lines[0]
        return result
    
    def solve(self, parsed_input):
        captcha = 0
        prev_digit = int(parsed_input[-1])
        for char in parsed_input:
            digit = int(char)
            if digit == prev_digit:
                captcha += digit
            prev_digit = digit
        result = captcha
        return result
    
    def solve2(self, parsed_input):
        N = len(parsed_input)
        captcha = 0
        for i in range(len(parsed_input)):
            j = (i + N // 2) % N
            digit = int(parsed_input[i])
            other_digit = int(parsed_input[j])
            if digit == other_digit:
                captcha += digit
        result = captcha
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
    python AdventOfCode2017.py 1 < inputs/2017day01.in
    '''
    solvers = {
        1: (Day01, 'Inverse Captcha'),
        2: (Day02, 'Corruption Checksum'),
        3: (Day03, 'Spiral Memory'),
        4: (Day04, 'High-Entropy Passphrases'),
        5: (Day05, 'A Maze of Twisty Trampolines, All Alike'),
        6: (Day06, 'Memory Reallocation'),
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
