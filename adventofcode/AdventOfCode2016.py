'''
Created 2021-06-28

@author: Sestren
'''
import argparse
import collections
import copy
import datetime
import functools
import heapq
import hashlib
import itertools
import json
import operator
import random
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
    https://adventofcode.com/2016/day/?
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

class Day04: # Security Through Obscurity
    '''
    Security Through Obscurity
    https://adventofcode.com/2016/day/4
    '''
    def get_rooms(self, raw_input_lines: List[str]):
        rooms = {}
        for raw_input_line in raw_input_lines:
            prefix, suffix = raw_input_line.split('[')
            parts = prefix.split('-')
            encrypted_name = '-'.join(parts[:-1])
            sector_id = int(parts[-1])
            checksum = suffix[:-1]
            rooms[encrypted_name] = (sector_id, checksum)
        result = rooms
        return result
    
    def decrypt(self, encrypted_name, shift_amount):
        def shift(char):
            return chr(ord('a') + (ord(char) - ord('a') + shift_amount) % 26)
        shift_amount %= 26
        decrypted_chars = []
        for char in encrypted_name:
            if char == '-':
                decrypted_chars.append(' ')
            else:
                decrypted_char = shift(char)
                decrypted_chars.append(decrypted_char)
        result = ''.join(decrypted_chars)
        return result
    
    def solve(self, rooms):
        sector_ids_of_real_rooms = []
        for encrypted_name, (sector_id, checksum) in rooms.items():
            char_counts = collections.Counter(encrypted_name)
            checksum_chars = []
            for char, count in char_counts.items():
                if char in 'abcdefghijklmnopqrstuvwxyz':
                    heapq.heappush(checksum_chars, (-count, char))
            room_checksum = ''
            for _ in range(5):
                _, char = heapq.heappop(checksum_chars)
                room_checksum += char
            if room_checksum == checksum:
                sector_ids_of_real_rooms.append(sector_id)
        result = sum(sector_ids_of_real_rooms)
        return result
    
    def solve2(self, rooms):
        decrypted_names = {}
        for encrypted_name, (sector_id, checksum) in rooms.items():
            decrypted_name = self.decrypt(encrypted_name, sector_id % 26)
            decrypted_names[decrypted_name] = sector_id
        target_sector_id = decrypted_names['northpole object storage']
        result = target_sector_id
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        rooms = self.get_rooms(raw_input_lines)
        solutions = (
            self.solve(rooms),
            self.solve2(rooms),
            )
        result = solutions
        return result

class Day03: # Squares With Three Sides
    '''
    Squares With Three Sides
    https://adventofcode.com/2016/day/3
    '''
    def get_triangles(self, raw_input_lines: List[str]):
        triangles = []
        for raw_input_line in raw_input_lines:
            triangle = tuple(map(int, raw_input_line.split()))
            triangles.append(triangle)
        result = triangles
        return result
    
    def get_points(self, raw_input_lines: List[str]):
        points = {}
        for row, raw_input_line in enumerate(raw_input_lines):
            a, b, c = tuple(map(int, raw_input_line.split()))
            points[(row, 0)] = a
            points[(row, 1)] = b
            points[(row, 2)] = c
        result = points
        return result
    
    def solve(self, triangles):
        possible_count = 0
        for a, b, c in triangles:
            if all([
                (a + b) > c,
                (b + c) > a,
                (a + c) > b,
            ]):
                possible_count += 1
        result = possible_count
        return result
    
    def solve2(self, points):
        possible_count = 0
        for col in range(3):
            for row_group in range(len(points) // 9):
                a = points[(3 * row_group + 0, col)]
                b = points[(3 * row_group + 1, col)]
                c = points[(3 * row_group + 2, col)]
                if all([
                    (a + b) > c,
                    (b + c) > a,
                    (a + c) > b,
                ]):
                    possible_count += 1
        result = possible_count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        triangles = self.get_triangles(raw_input_lines)
        points = self.get_points(raw_input_lines)
        solutions = (
            self.solve(triangles),
            self.solve2(points),
            )
        result = solutions
        return result

class Day02: # Bathroom Security
    '''
    Bathroom Security
    https://adventofcode.com/2016/day/2
    '''
    moves = {
        'U': (-1,  0),
        'D': ( 1,  0),
        'L': ( 0, -1),
        'R': ( 0,  1),
    }

    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, parsed_input):
        keypad = {
            (0, 0): '1',
            (0, 1): '2',
            (0, 2): '3',
            (1, 0): '4',
            (1, 1): '5',
            (1, 2): '6',
            (2, 0): '7',
            (2, 1): '8',
            (2, 2): '9',
        }
        row = 1
        col = 1
        bathroom_code = []
        for instructions in parsed_input:
            for move in instructions:
                next_row = row + self.moves[move][0]
                next_col = col + self.moves[move][1]
                if (next_row, next_col) in keypad:
                    row = next_row
                    col = next_col
            bathroom_code.append(keypad[(row, col)])
        result = ''.join(bathroom_code)
        return result
    
    def solve2(self, parsed_input):
        keypad = {
            (0, 2): '1',
            (1, 1): '2',
            (1, 2): '3',
            (1, 3): '4',
            (2, 0): '5',
            (2, 1): '6',
            (2, 2): '7',
            (2, 3): '8',
            (2, 4): '9',
            (3, 1): 'A',
            (3, 2): 'B',
            (3, 3): 'C',
            (4, 2): 'D',
        }
        row = 1
        col = 1
        bathroom_code = []
        for instructions in parsed_input:
            for move in instructions:
                next_row = row + self.moves[move][0]
                next_col = col + self.moves[move][1]
                if (next_row, next_col) in keypad:
                    row = next_row
                    col = next_col
            bathroom_code.append(keypad[(row, col)])
        result = ''.join(bathroom_code)
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

class Day01: # No Time for a Taxicab
    '''
    No Time for a Taxicab
    https://adventofcode.com/2016/day/1
    '''
    rotations = {
        'L': -1,
        'R':  1,
    }

    directions = {
        0: (-1,  0), # North
        1: ( 0,  1), # East
        2: ( 1,  0), # South
        3: ( 0, -1), # West
    }

    def get_instructions(self, raw_input_lines: List[str]):
        instructions = []
        for raw_input_line in raw_input_lines[0].split(', '):
            rotation = raw_input_line[:1]
            blocks = int(raw_input_line[1:])
            instructions.append((rotation, blocks))
        result = instructions
        return result
    
    def solve(self, instructions):
        row = 0
        col = 0
        facing = 0
        for rotation, blocks in instructions:
            facing = (facing + self.rotations[rotation]) % len(self.directions)
            row += blocks * self.directions[facing][0]
            col += blocks * self.directions[facing][1]
        result = abs(row) + abs(col)
        return result
    
    def solve2(self, instructions):
        row = 0
        col = 0
        facing = 0
        visits = set()
        visits.add((row, col))
        repeat_visit = None
        for rotation, blocks in instructions:
            facing = (facing + self.rotations[rotation]) % len(self.directions)
            target_row = row + blocks * self.directions[facing][0]
            target_col = col + blocks * self.directions[facing][1]
            while row != target_row or col != target_col:
                row += self.directions[facing][0]
                col += self.directions[facing][1]
                if (row, col) in visits:
                    repeat_visit = (row, col)
                    break
                visits.add((row, col))
            if repeat_visit is not None:
                break
        result = abs(repeat_visit[0]) + abs(repeat_visit[1])
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

if __name__ == '__main__':
    '''
    Usage
    python AdventOfCode2016.py 1 < inputs/2016day01.in
    '''
    solvers = {
        1: (Day01, 'No Time for a Taxicab'),
        2: (Day02, 'Bathroom Security'),
        3: (Day03, 'Squares With Three Sides'),
        4: (Day04, 'Security Through Obscurity'),
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
