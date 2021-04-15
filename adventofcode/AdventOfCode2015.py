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

class Day07: # Some Assembly Required
    '''
    Some Assembly Required
    https://adventofcode.com/2015/day/7
    '''
    def get_connections(self, raw_input_lines: List[str]):
        connections = {}
        for raw_input_line in raw_input_lines:
            a, wire = raw_input_line.split(' -> ')
            connection = list(a.split(' '))
            for i in range(len(connection)):
                try:
                    value = int(connection[i])
                    connection[i] = value
                except ValueError:
                    pass
            connections[wire] = tuple(connection)
        result = connections
        return result
    
    def solve(self, connections):
        # NOTE: This is not in a workable state yet
        OPS = {'NOT', 'AND', 'OR', 'LSHIFT', 'RSHIFT'}
        MODULO = 2 ** 16 # 0 to 65535
        inputs = {}
        while len(connections) > 0:
            for wire, connection in connections.items():
                ready = True
                for i, part in enumerate(connection):
                    if (
                        type(part) is str and
                        part not in OPS and
                        part not in inputs
                    ):
                        ready = False
                        break
                if ready:
                    print(wire, connection)
                    if len(connection) == 1:
                        if type(connection[0]) is int:
                            inputs[wire] = connection[0]
                        else:
                            inputs[wire] = inputs[connection[0]]
                    elif len(connection) == 2:
                        OP, a = connection
                        assert OP == 'NOT'
                        if type(a) is str:
                            a = inputs[a]
                        value = ~a
                    elif len(connection) == 3:
                        a, OP, b = connection
                        value = 0
                        if type(a) is str:
                            a = inputs[a]
                        if type(b) is str:
                            b = inputs[b]
                        if OP == 'AND':
                            value = a & b
                        elif OP == 'OR':
                            value = a | b
                        elif OP == 'LSHIFT':
                            value = a << b
                        elif OP == 'RSHIFT':
                            value = a >> b
                        value = value % MODULO
                    else:
                        raise AssertionError
            for wire in inputs:
                if wire in connections:
                    connections.pop(wire)
        result = inputs['i']
        return result
    
    def solve2(self, connections):
        result = len(connections)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        connections = self.get_connections(raw_input_lines)
        solutions = (
            self.solve(connections),
            self.solve2(connections),
            )
        result = solutions
        return result

class Day06: # Probably a Fire Hazard
    '''
    Probably a Fire Hazard
    https://adventofcode.com/2015/day/6
    '''
    def get_instructions(self, raw_input_lines: List[str]):
        instructions = []
        for raw_input_line in raw_input_lines:
            a, b = raw_input_line.split(' through ')
            mode = 'ERROR'
            if 'toggle' in a:
                a1, a2 = a.split(' ')
                left, top = tuple(map(int, a2.split(',')))
                mode = 'toggle'
            else:
                a1, a2, a3 = a.split(' ')
                left, top = tuple(map(int, a3.split(',')))
                if a2 == 'on':
                    mode = 'turn_on'
                elif a2 == 'off':
                    mode = 'turn_off'
            right, bottom = tuple(map(int, b.split(',')))
            instruction = (mode, left, top, right, bottom)
            instructions.append(instruction)
        result = instructions
        return result
    
    def solve(self, instructions):
        lights = [0] * (1000 * 1000)
        for mode, left, top, right, bottom in instructions:
            for row in range(top, bottom + 1):
                for col in range(left, right + 1):
                    index = 1000 * row + col
                    if mode == 'turn_on':
                        lights[index] = 1
                    elif mode == 'turn_off':
                        lights[index] = 0
                    elif mode == 'toggle':
                        lights[index] =  1 - lights[index]
        result = sum(lights)
        return result
    
    def solve2(self, instructions):
        lights = [0] * (1000 * 1000)
        for mode, left, top, right, bottom in instructions:
            for row in range(top, bottom + 1):
                for col in range(left, right + 1):
                    index = 1000 * row + col
                    if mode == 'turn_on':
                        lights[index] += 1
                    elif mode == 'turn_off':
                        lights[index] = max(0, lights[index] - 1)
                    elif mode == 'toggle':
                        lights[index] += 2
        result = sum(lights)
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

class Day05: # Doesn't He Have Intern-Elves For This?
    '''
    Doesn't He Have Intern-Elves For This?
    https://adventofcode.com/2015/day/5
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, parsed_input):
        '''
        Nice strings:
            Contain at least three vowels (aeiou)
            Contain at least one letter that appears twice in a row
            Do not contain the strings ab, cd, pq, or xy
        '''
        nice_string_count = 0
        for string in parsed_input:
            nice_ind = all([
                sum([
                    string.count('a'),
                    string.count('e'),
                    string.count('i'),
                    string.count('o'),
                    string.count('u'),
                ]) >= 3,
                any(
                    string[i] == string[i - 1] for
                    i in range(1, len(string))
                ),
                all([
                    'ab' not in string,
                    'cd' not in string,
                    'pq' not in string,
                    'xy' not in string,
                ]),
            ])
            if nice_ind:
                nice_string_count += 1
        result = nice_string_count
        return result
    
    def solve2(self, parsed_input):
        '''
        Nice strings:
            Contain a pair of any two letters that appear twice without overlapping
            Contain at least one letter which repeats with ONE letter between them
        '''
        nice_string_count = 0
        for string in parsed_input:
            pairs = {}
            duplicate_pair_found = False
            triplet_found = False
            for i in range(1, len(string)):
                pair = string[i - 1:i + 1]
                if pair in pairs and pairs[pair] < i - 1:
                    duplicate_pair_found = True
                if pair not in pairs:
                    pairs[pair] = i
                if i < len(string) - 1:
                    triplet = string[i - 1: i + 2]
                    if triplet[0] == triplet[2]:
                        triplet_found = True
            if duplicate_pair_found and triplet_found:
                nice_string_count += 1
        result = nice_string_count
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
    python AdventOfCode2015.py 6 < inputs/2015day06.in
    '''
    solvers = {
        1: (Day01, 'Not Quite Lisp'),
        2: (Day02, 'I Was Told There Would Be No Math'),
        3: (Day03, 'Perfectly Spherical Houses in a Vacuum'),
        4: (Day04, 'The Ideal Stocking Stuffer'),
        5: (Day05, 'Doesn\'t He Have Intern-Elves For This?'),
        6: (Day06, 'Probably a Fire Hazard'),
        7: (Day07, 'Some Assembly Required'),
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
