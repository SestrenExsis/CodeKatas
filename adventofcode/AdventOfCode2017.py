'''
Created on 2021-11-29

@author: Sestren
'''
import argparse
import collections
import copy
import functools
import operator
import re
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

class Day11: # Hex Ed
    '''
    https://adventofcode.com/2017/day/11
    '''
    steps = {
        'n' : ( 0, -1),
        'ne': ( 1, -1),
        'se': ( 1,  0),
        's' : ( 0,  1),
        'sw': (-1,  1),
        'nw': (-1,  0),
    }

    def get_path(self, raw_input_lines: List[str]):
        path = raw_input_lines[0].split(',')
        result = path
        return result
    
    def get_distance(self, q, r):
        distance = 0
        if q < 0 and r > 0:
            distance += min(abs(q), r)
            q += distance
            r -= distance
        elif q > 0 and r < 0:
            distance += min(q, abs(r))
            q -= distance
            r += distance
        distance += abs(q) + abs(r)
        result = distance
        return result
    
    def solve(self, path):
        q = 0
        r = 0
        for step in path:
            q += self.steps[step][0]
            r += self.steps[step][1]
        distance = self.get_distance(q, r)
        result = distance
        return result
    
    def solve2(self, path):
        q = 0
        r = 0
        max_distance = 0
        for step in path:
            q += self.steps[step][0]
            r += self.steps[step][1]
            distance = self.get_distance(q, r)
            max_distance = max(max_distance, distance)
        result = max_distance
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        path = self.get_path(raw_input_lines)
        solutions = (
            self.solve(path),
            self.solve2(path),
            )
        result = solutions
        return result

class Day10: # Knot Hash
    '''
    https://adventofcode.com/2017/day/10
    '''
    def get_lengths(self, raw_input_lines: List[str]):
        lengths = list(map(int, raw_input_lines[0].split(',')))
        result = lengths
        return result
    
    def solve(self, lengths, num_count: int=256):
        nums = list(range(num_count))
        cursor = 0
        skip_size = 0
        for length in lengths:
            left = cursor
            right = cursor + length - 1
            while left < right:
                L = left % len(nums)
                R = right % len(nums)
                nums[L], nums[R] = nums[R], nums[L]
                left += 1
                right -= 1
            cursor = (cursor + length + skip_size) % len(nums)
            skip_size += 1
        result = nums[0] * nums[1]
        return result
    
    def solve2(self, chars):
        lengths = []
        for char in chars:
            lengths.append(ord(char))
        lengths += [17, 31, 73, 47, 23]
        nums = list(range(256))
        cursor = 0
        skip_size = 0
        for round_id in range(64):
            for length in lengths:
                left = cursor
                right = cursor + length - 1
                while left < right:
                    L = left % len(nums)
                    R = right % len(nums)
                    nums[L], nums[R] = nums[R], nums[L]
                    left += 1
                    right -= 1
                cursor = (cursor + length + skip_size) % len(nums)
                skip_size += 1
        xors = [0] * 16
        for i in range(16):
            for j in range(16):
                xors[i] ^= nums[16 * i + j]
        chars = []
        for i in range(16):
            chars.append(hex(xors[i])[2:])
        result = ''.join(chars)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        lengths = self.get_lengths(raw_input_lines)
        assert self.solve([3, 4, 1, 5], 5) == 12
        solutions = (
            self.solve(lengths),
            self.solve2(raw_input_lines[0]),
            )
        result = solutions
        return result

class Day09: # Stream Processing
    '''
    https://adventofcode.com/2017/day/9
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = raw_input_lines[0]
        return result
    
    def solve(self, parsed_input):
        score = 0
        stack = [parsed_input[0]]
        for char in parsed_input[1:]:
            if stack[-1] == '!':
                stack.pop()
            elif char == '}' and stack[-1] == '{':
                score += len(stack)
                stack.pop()
            elif char == '{' and stack[-1] == '{':
                stack.append('{')
            elif char == '!' and stack[-1] == '<':
                stack.append('!')
            elif char == '<' and stack[-1] != '<':
                stack.append('<')
            elif char == '>' and stack[-1] == '<':
                stack.pop()
        result = score
        return result
    
    def solve2(self, parsed_input):
        garbage_removed = 0
        stack = [parsed_input[0]]
        for char in parsed_input[1:]:
            if stack[-1] == '!':
                stack.pop()
            elif char == '}' and stack[-1] == '{':
                stack.pop()
            elif char == '{' and stack[-1] == '{':
                stack.append('{')
            elif char == '!' and stack[-1] == '<':
                stack.append('!')
            elif char == '<' and stack[-1] != '<':
                stack.append('<')
            elif char == '>' and stack[-1] == '<':
                stack.pop()
            elif stack[-1] == '<':
                garbage_removed += 1
        result = garbage_removed
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

class Day08: # I Heard You Like Registers
    '''
    https://adventofcode.com/2017/day/8
    '''
    def get_instructions(self, raw_input_lines: List[str]):
        instructions = []
        for raw_input_line in raw_input_lines:
            a, b = raw_input_line.split(' if ')
            register, operation, amount = a.split(' ')
            amount = int(amount)
            register2, inequality, amount2 = b.split(' ')
            amount2 = int(amount2)
            instructions.append((
                (register, operation, amount),
                (register2, inequality, amount2),
            ))
        result = instructions
        return result
    
    def execute_instruction(self, registers, instruction):
        operation, condition = instruction
        condition_satisfied_ind = False
        register2, inequality, amount2 = condition
        if inequality == '<':
            condition_satisfied_ind = registers[register2] < amount2
        elif inequality == '<=':
            condition_satisfied_ind = registers[register2] <= amount2
        elif inequality == '==':
            condition_satisfied_ind = registers[register2] == amount2
        elif inequality == '!=':
            condition_satisfied_ind = registers[register2] != amount2
        elif inequality == '>=':
            condition_satisfied_ind = registers[register2] >= amount2
        elif inequality == '>':
            condition_satisfied_ind = registers[register2] > amount2
        else:
            raise Exception('Conditional operator not found!')
        if condition_satisfied_ind:
            register, operation, amount = operation
            if operation == 'inc':
                registers[register] += amount
            elif operation == 'dec':
                registers[register] -= amount
    
    def solve(self, instructions):
        registers = collections.defaultdict(int)
        for instruction in instructions:
            self.execute_instruction(registers, instruction)
        result = max(registers.values())
        return result
    
    def solve2(self, instructions):
        result = float('-inf')
        registers = collections.defaultdict(int)
        for instruction in instructions:
            self.execute_instruction(registers, instruction)
            max_register_value = max(registers.values())
            result = max(result, max_register_value)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        instructions = self.get_instructions(raw_input_lines)
        solutions = (
            self.solve(instructions),
            self.solve2(instructions),
            )
        result = solutions
        return result

class Day07: # Recursive Circus
    '''
    https://adventofcode.com/2017/day/7
    '''
    class Program:
        def __init__(self, program_name: str, weight: int, children: set):
            self.name = program_name
            self.weight = weight
            self.children = set(children)

    def get_programs(self, raw_input_lines: List[str]):
        programs = {}
        for raw_input_line in raw_input_lines:
            line = raw_input_line.replace(',', '')
            parts = line.split(' ')
            program_name = parts[0]
            weight = int(parts[1][1:-1])
            children = []
            for item in parts[3:]:
                children.append(item)
            program = self.Program(program_name, weight, children)
            programs[program_name] = program
            # print(program.name, program.weight, program.children)
        result = programs
        return result
    
    def find_bottom_program(self, programs):
        bases = set()
        subs = set()
        for program in programs.values():
            if len(program.children) > 1:
                bases.add(program.name)
            for child in program.children:
                subs.add(child)
        bottom_program = next(iter(bases - subs))
        result = bottom_program
        return result
    
    def solve(self, programs):
        result = self.find_bottom_program(programs)
        return result
    
    def solve2(self, programs):
        programs_to_weigh = set(programs.keys())
        weights = {}
        while len(programs_to_weigh) > 0:
            programs_weighed = set()
            for program_name in programs_to_weigh:
                program = programs[program_name]
                if len(program.children & programs_to_weigh) > 0:
                    continue
                weights[program_name] = program.weight
                for child in program.children:
                    weights[program_name] += weights[child]
                programs_weighed.add(program_name)
            programs_to_weigh -= programs_weighed
        result = -1
        work = set(programs.keys())
        while len(work) > 0:
            program_name = work.pop()
            program = programs[program_name]
            if len(program.children) < 3:
                continue
            weight_counts = collections.defaultdict(int)
            mode_weight = -1
            for child in program.children:
                weight_counts[weights[child]] += 1
            if len(weight_counts) > 1:
                program_to_change = None
                for child_name in program.children:
                    if weight_counts[weights[child_name]] == 1:
                        program_to_change = child_name
                    else:
                        mode_weight = weights[child_name]
                diff = mode_weight - weights[program_to_change]
                program = programs[program_to_change]
                result = program.weight + diff
                break
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        programs = self.get_programs(raw_input_lines)
        solutions = (
            self.solve(programs),
            self.solve2(programs),
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
        7: (Day07, 'Recursive Circus'),
        8: (Day08, 'I Heard You Like Registers'),
        9: (Day09, 'Stream Processing'),
       10: (Day10, 'Knot Hash'),
       11: (Day11, 'Hex Ed'),
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
