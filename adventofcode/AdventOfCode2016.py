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

class AssembunnyVM: # Virtual Machine for running Assembunny code
    def __init__(self):
        self.cycle_count = 0
        self.instructions = []
        self.registers = {'a': 0, 'b': 0, 'c': 0, 'd': 0}
        self.pc = 0
    
    def load_raw_input(self, raw_input_lines: List[str]):
        self.instructions = []
        for raw_input_line in raw_input_lines:
            parts = raw_input_line.split(' ')
            for i in range(len(parts)):
                try:
                    parts[i] = int(parts[i])
                except ValueError:
                    pass
            instruction = tuple(parts)
            self.instructions.append(instruction)
    
    def __cpy(self, x, y):
        x_val = x if type(x) is int else self.registers[x]
        if type(y) is str and y in self.registers:
            self.registers[y] = x_val
    
    def __add(self, x, y):
        y_val = y if type(y) is int else self.registers[y]
        if type(x) is str and x in self.registers:
            self.registers[x] += y_val
    
    def __jnz(self, x, y):
        x_val = x if type(x) is int else self.registers[x]
        y_val = y if type(y) is int else self.registers[y]
        if x_val != 0:
            self.pc += y_val - 1
    
    def __tgl(self, x):
        x_val = x if type(x) is int else self.registers[x]
        target_pc = self.pc + x_val
        try:
            instruction = list(self.instructions[target_pc])
            if instruction[0] == 'inc':
                instruction[0] = 'dec'
            elif len(instruction) == 2:
                instruction[0] = 'inc'
            elif instruction[0] == 'jnz':
                instruction[0] = 'cpy'
            elif len(instruction) == 3:
                instruction[0] = 'jnz'
            self.instructions[target_pc] = tuple(instruction)
        except IndexError:
            pass

    def step(self):
        instruction = self.instructions[self.pc]
        op = instruction[0]
        if op == 'cpy':
            self.__cpy(instruction[1], instruction[2])
        elif op == 'inc':
            self.__add(instruction[1], 1)
        elif op == 'dec':
            self.__add(instruction[1], -1)
        elif op == 'jnz':
            self.__jnz(instruction[1], instruction[2])
        elif op == 'tgl':
            self.__tgl(instruction[1])
        self.pc += 1
        self.cycle_count += 1

    def run(self, time_limit:float=float('inf')) -> bool:
        halt_ind = False
        start_time = time.time()
        self.pc = 0
        while self.pc < len(self.instructions):
            self.step()
            elapsed_time = time.time() - start_time
            if elapsed_time >= time_limit:
                break
        else:
            halt_ind = True
        result = halt_ind
        return result
    
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

class Day23: # Safe Cracking
    '''
    Safe Cracking
    https://adventofcode.com/2016/day/23
    '''
    def solve(self, raw_input_lines):
        vm = AssembunnyVM()
        vm.load_raw_input(raw_input_lines)
        vm.registers['a'] = 7
        vm.run()
        result = vm.registers['a']
        print('cycles:', vm.cycle_count)
        return result
    
    def solve2(self, raw_input_lines):
        vm = AssembunnyVM()
        vm.load_raw_input(raw_input_lines)
        vm.registers['a'] = 12
        result = None
        if vm.run(5.0):
            result = vm.registers['a']
        print('cycles:', vm.cycle_count)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        solutions = (
            self.solve(raw_input_lines),
            self.solve2(raw_input_lines),
            )
        result = solutions
        return result

class Day22Incomplete: # Grid Computing
    '''
    Grid Computing
    https://adventofcode.com/2016/day/22
    '''
    def get_grid(self, raw_input_lines: List[str]):
        grid = {}
        for raw_input_line in raw_input_lines[2:]:
            parts = raw_input_line.split()
            parts0 = parts[0].split('-')
            col = int(parts0[1][1:])
            row = int(parts0[2][1:])
            size = int(parts[1][:-1])
            used = int(parts[2][:-1])
            avail = int(parts[3][:-1])
            use_pct = int(parts[4][:-1])
            grid[(row, col)] = {
                'size': size,
                'used': used,
                'avail': avail,
                'use_pct': use_pct
            }
        result = grid
        return result

    def show_grid(self, grid, rows, cols):
        for row in range(rows):
            row_data = []
            for col in range(cols):
                cell = grid[(row, col)]
                row_data.append(str(cell['used']) + '/' + str(cell['size']))
            print(' '.join(row_data))
    
    def solve(self, grid):
        viable_pair_count = 0
        for node_a, node_a_data in grid.items():
            if node_a_data['used'] < 1:
                continue
            for node_b, node_b_data in grid.items():
                if node_b == node_a:
                    continue
                if node_a_data['used'] <= node_b_data['avail']:
                    viable_pair_count += 1
        result = viable_pair_count
        return result
    
    def solve2(self, grid):
        result = len(grid)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        grid = self.get_grid(raw_input_lines)
        rows = max(row for row, _ in grid.keys())
        cols = max(col for _, col in grid.keys())
        self.show_grid(grid, rows, cols)
        solutions = (
            self.solve(grid),
            self.solve2(grid),
            )
        result = solutions
        return result

class Day21: # Scrambled Letters and Hash
    '''
    Scrambled Letters and Hash
    https://adventofcode.com/2016/day/21
    '''
    def get_instructions(self, raw_input_lines: List[str]):
        instructions = []
        for raw_input_line in raw_input_lines:
            '''
            swap position X with position Y
            swap letter A with letter B
            rotate left X steps
            rotate right X steps
            rotate based on position of letter A
            reverse positions X through Y
            move position X to position Y
            '''
            instruction = None
            parts = raw_input_line.split(' ')
            if parts[0] == 'swap' and parts[1] == 'position':
                x = int(parts[2])
                y = int(parts[5])
                instruction = ('swap position', x, y)
            elif parts[0] == 'swap' and parts[1] == 'letter':
                a = parts[2]
                b = parts[5]
                instruction = ('swap letter', a, b)
            elif parts[0] == 'rotate' and parts[1] == 'left':
                x = int(parts[2])
                instruction = ('rotate left', x)
            elif parts[0] == 'rotate' and parts[1] == 'right':
                x = int(parts[2])
                instruction = ('rotate right', x)
            elif parts[0] == 'rotate' and parts[1] == 'based':
                a = parts[6]
                instruction = ('rotate base', a)
            elif parts[0] == 'reverse':
                x = int(parts[2])
                y = int(parts[4])
                instruction = ('reverse', x, y)
            elif parts[0] == 'move':
                x = int(parts[2])
                y = int(parts[5])
                instruction = ('move', x, y)
            instructions.append(instruction)
        result = instructions
        return result
    
    def solve(self, password, instructions):
        chars = list(password)
        for instruction in instructions:
            if instruction[0] == 'swap position':
                x, y = instruction[1], instruction[2]
                chars[x], chars[y] = chars[y], chars[x]
            elif instruction[0] == 'swap letter':
                a, b = instruction[1], instruction[2]
                x = chars.index(a)
                y = chars.index(b)
                chars[x], chars[y] = chars[y], chars[x]
            elif instruction[0] == 'rotate left':
                x = instruction[1] % len(chars)
                chars = chars[x:] + chars[:x]
            elif instruction[0] == 'rotate right':
                x = instruction[1] % len(chars)
                chars = chars[-x:] + chars[:-x]
            elif instruction[0] == 'rotate base':
                a = instruction[1]
                pos = chars.index(a)
                index = (pos + 1 + (0 if pos < 4 else 1)) % len(chars)
                chars = chars[-index:] + chars[:-index]
            elif instruction[0] == 'reverse':
                x, y = instruction[1], instruction[2]
                while x < y:
                    chars[x], chars[y] = chars[y], chars[x]
                    x += 1
                    y -= 1
            elif instruction[0] == 'move':
                x, y = instruction[1], instruction[2]
                char = chars[x]
                chars = chars[:x] + chars[x + 1:]
                chars = chars[:y] + [char] + chars[y:]
            elif instruction[0] == 'inverse rotate base':
                a = instruction[1]
                pos = chars.index(a)
                index = pos
                if pos == 1:
                    index = 7
                elif pos == 3:
                    index = 6
                elif pos == 5:
                    index = 5
                elif pos == 7:
                    index = 4
                elif pos == 2:
                    index = 2
                elif pos == 4:
                    index = 1
                elif pos == 6:
                    index = 0
                elif pos == 0:
                    index = 7
                chars = chars[-index:] + chars[:-index]
        result = ''.join(chars)
        return result
    
    def solve2(self, scrambled_password, instructions):
        '''
        What are the inverses of the following functions?
            swap position X with position Y:
                swap position X with position Y
            swap letter A with letter B:
                swap letter A with letter B ???
            rotate left X steps:
                rotate right X steps
            rotate right X steps:
                rotate left X steps
            rotate based on position of letter A:
                if you found the target letter at:
                    ... 1, it was originally at 0, so rotate left 1
                    ... 3, it was originally at 1, so rotate left 2
                    ... 5, it was originally at 2, so rotate left 3
                    ... 7, it was originally at 3, so rotate left 4
                    ... 2, it was originally at 4, so rotate right 2
                    ... 4, it was originally at 5, so rotate right 1
                    ... 6, it was originally at 6, so do nothing
                    ... 0, it was originally at 7, so rotate left 1
            reverse positions X through Y:
                reverse positions X through Y
            move position X to position Y:
                move position Y to position X
        '''
        inverse_instructions = []
        for instruction in reversed(instructions):
            parts = list(instruction)
            if instruction[0] == 'swap position':
                pass
            elif instruction[0] == 'swap letter':
                pass
            elif instruction[0] == 'rotate left':
                parts[0] = 'rotate right'
            elif instruction[0] == 'rotate right':
                parts[0] = 'rotate left'
            elif instruction[0] == 'rotate base':
                parts[0] = 'inverse rotate base'
            elif instruction[0] == 'reverse':
                pass
            elif instruction[0] == 'move':
                parts[1], parts[2] = parts[2], parts[1]
            inverse_instruction = tuple(parts)
            inverse_instructions.append(inverse_instruction)
        result = self.solve(scrambled_password, inverse_instructions)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        instructions = self.get_instructions(raw_input_lines)
        solutions = (
            self.solve('abcdefgh', instructions),
            self.solve2('fbgdceah', instructions),
            )
        result = solutions
        return result

class Day20: # Firewall Rules
    '''
    Firewall Rules
    https://adventofcode.com/2016/day/20
    '''
    MIN_IP = 0
    MAX_IP = 4294967295

    def get_banned_ranges(self, raw_input_lines: List[str]):
        banned_ranges = set()
        for raw_input_line in raw_input_lines:
            banned_range = tuple(map(int, raw_input_line.split('-')))
            banned_ranges.add(banned_range)
        result = banned_ranges
        return result
    
    def solve(self, banned_ranges):
        '''
        The lowest-valued IP that is not blocked must occur adjacent to
        the boundaries of one of the blocked ranges. In other words, if
        a blocked range is A-B, then the only possible candidates could be
        A - 1 or B + 1
        '''
        candidates = set()
        for a, b in banned_ranges:
            candidates.add(a - 1)
            candidates.add(b + 1)
        min_valid_ip = float('inf')
        for candidate in candidates:
            if candidate < self.MIN_IP or candidate > self.MAX_IP:
                continue
            valid_ind = True
            for lower, upper in banned_ranges:
                if lower <= candidate <= upper:
                    valid_ind = False
                    break
            if valid_ind:
                min_valid_ip = min(min_valid_ip, candidate)
        result = min_valid_ip
        return result
    
    def solve2(self, banned_ranges):
        valid_ip_count = 0
        current_ip = self.MIN_IP
        for a, b in sorted(banned_ranges):
            if a > current_ip:
                valid_ip_count += a - current_ip - 1
            current_ip = max(current_ip, b)
        if current_ip < self.MAX_IP:
            valid_ip_count += self.MAX_IP - current_ip
        result = valid_ip_count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        banned_ranges = self.get_banned_ranges(raw_input_lines)
        solutions = (
            self.solve(banned_ranges),
            self.solve2(banned_ranges),
            )
        result = solutions
        return result

class Day19: # An Elephant Named Joseph
    '''
    An Elephant Named Joseph
    https://adventofcode.com/2016/day/?
    '''
    class Node():
        def __init__(self, node_id):
            self.node_id = node_id
            self.value = 1
            self.next = None

    def get_elf_count(self, raw_input_lines: List[str]):
        result = int(raw_input_lines[0])
        return result
    
    def solve(self, elf_count):
        current_elf = self.Node(1)
        prev_elf = current_elf
        for elf_id in range(2, elf_count + 1):
            elf = self.Node(elf_id)
            prev_elf.next = elf
            prev_elf = elf
        prev_elf.next = current_elf
        while current_elf.next != current_elf:
            next_elf = current_elf.next
            current_elf.value += next_elf.value
            current_elf.next = next_elf.next
            current_elf = next_elf
        result = current_elf.node_id
        return result
    
    def solve2_slowly(self, elf_count):
        current_elf = self.Node(1)
        prev_elf = current_elf
        for elf_id in range(2, elf_count + 1):
            elf = self.Node(elf_id)
            prev_elf.next = elf
            prev_elf = elf
        prev_elf.next = current_elf
        while current_elf.next != current_elf:
            across_elf = current_elf
            for _ in range(elf_count // 2):
                across_elf = across_elf.next
                prev_elf = prev_elf.next
            current_elf.value += across_elf.value
            prev_elf.next = across_elf.next
            prev_elf = current_elf
            current_elf = current_elf.next
            elf_count -= 1
        result = current_elf.node_id
        return result
    
    def solve2(self, elf_count):
        '''
        The pattern appears to be:
        index starts at 1 and increments by 1
        solution starts at 1
        solution increments by 1 if it is less than half the index
        solution increments by 2 if it is >= half the index
        solution starts over at 1 if it is >= index
        '''
        elf_id = 1
        for i in range(1, elf_count + 1):
            if elf_id >= i // 2:
                elf_id += 2
            else:
                elf_id += 1
            if elf_id > i:
                elf_id = 1
        result = elf_id
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        elf_count = self.get_elf_count(raw_input_lines)
        for i in range(2, 100):
            assert self.solve2_slowly(i) == self.solve2(i)
        solutions = (
            self.solve(elf_count),
            self.solve2(elf_count),
            )
        result = solutions
        return result

class Day18: # Like a Rogue
    '''
    Like a Rogue
    https://adventofcode.com/2016/day/18
    '''
    def get_first_row(self, raw_input_lines: List[str]):
        first_row = raw_input_lines[0]
        result = first_row
        return result
    
    def solve(self, first_row, rows):
        traps = {'^^.', '.^^', '^..', '..^'}
        safe_tile_count = sum(1 for char in first_row if char == '.')
        prev_row = first_row
        for _ in range(rows - 1):
            row = []
            for i in range(len(prev_row)):
                left = '.' if i < 1 else prev_row[i - 1]
                center = prev_row[i]
                right = '.' if i >= len(prev_row) - 1 else prev_row[i + 1]
                key = left + center + right
                cell = '^' if key in traps else '.'
                row.append(cell)
            prev_row = ''.join(row)
            safe_tile_count += sum(1 for char in prev_row if char == '.')
        result = safe_tile_count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        first_row = self.get_first_row(raw_input_lines)
        solutions = (
            self.solve(first_row, 40),
            self.solve(first_row, 400_000),
            )
        result = solutions
        return result

class Day17: # Two Steps Forward
    '''
    Two Steps Forward
    https://adventofcode.com/2016/day/17
    '''
    def get_passcode(self, raw_input_lines: List[str]):
        passcode = raw_input_lines[0]
        result = passcode
        return result
    
    def solve(self, passcode):
        rows = 4
        cols = 4
        directions = {
            'U': (-1, 0, 0),
            'D': ( 1, 0, 1),
            'L': ( 0,-1, 2),
            'R': ( 0, 1, 3),
        }
        shortest_path = None
        work = collections.deque()
        work.append((0, 0, 0, '')) # (distance, row, col, path)
        while len(work) > 0:
            distance, row, col, path = work.pop()
            if row == 3 and col == 3:
                shortest_path = path
                break
            input_string = (passcode + path).encode('utf-8')
            hash = hashlib.md5(input_string).hexdigest()[:4]
            for direction in directions:
                row_offset, col_offset, index = directions[direction]
                next_row = row + row_offset
                next_col = col + col_offset
                if (
                    0 <= next_row < rows and
                    0 <= next_col < cols and
                    hash[index] in 'bcdef'
                ):
                    work.appendleft(
                        (distance + 1, next_row, next_col, path + direction),
                    )
        result = shortest_path
        return result
    
    def solve2(self, passcode):
        rows = 4
        cols = 4
        directions = {
            'U': (-1, 0, 0),
            'D': ( 1, 0, 1),
            'L': ( 0,-1, 2),
            'R': ( 0, 1, 3),
        }
        longest_path = None
        work = collections.deque()
        work.append((0, 0, 0, '')) # (distance, row, col, path)
        while len(work) > 0:
            distance, row, col, path = work.pop()
            if row == 3 and col == 3:
                if longest_path is None or longest_path[0] < distance:
                    longest_path = (distance, path)
                continue
            input_string = (passcode + path).encode('utf-8')
            hash = hashlib.md5(input_string).hexdigest()[:4]
            for direction in directions:
                row_offset, col_offset, index = directions[direction]
                next_row = row + row_offset
                next_col = col + col_offset
                if (
                    0 <= next_row < rows and
                    0 <= next_col < cols and
                    hash[index] in 'bcdef'
                ):
                    work.appendleft(
                        (distance + 1, next_row, next_col, path + direction),
                    )
        result = longest_path[1]
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        passcode = self.get_passcode(raw_input_lines)
        solutions = (
            self.solve(passcode),
            self.solve2(passcode),
            )
        result = solutions
        return result

class Day16: # Dragon Checksum
    '''
    Dragon Checksum
    https://adventofcode.com/2016/day/16
    '''
    def get_initial_state(self, raw_input_lines: List[str]):
        initial_state = list(map(int, raw_input_lines[0]))
        result = initial_state
        return result
    
    def solve(self, initial_state, disk_length):
        state = initial_state
        while len(state) < disk_length:
            a = state
            b = [1 - x for x in reversed(a)]
            state = a + [0] + b
        checksum = state[:disk_length]
        while True:
            half_len = len(checksum) // 2
            next_checksum = checksum[:half_len]
            for i in range(half_len):
                index = 2 * i
                value = 1 - (abs(checksum[index] - checksum[index + 1]))
                next_checksum[i] = value
            checksum = next_checksum[:]
            if len(checksum) % 2 == 1:
                break
        result = ''.join(map(str, checksum))
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        initial_state = self.get_initial_state(raw_input_lines)
        solutions = (
            self.solve(initial_state[:], 272),
            self.solve(initial_state[:], 35_651_584),
            )
        result = solutions
        return result

class Day15: # Timing is Everything
    '''
    Timing is Everything
    https://adventofcode.com/2016/day/15
    '''
    def get_discs(self, raw_input_lines: List[str]):
        discs = []
        for raw_input_line in raw_input_lines:
            parts = raw_input_line.split(' ')
            size = int(parts[3])
            offset = int(parts[11][:-1])
            discs.append((size, offset))
        result = discs
        return result
    
    def solve(self, discs):
        for i, (size, offset) in enumerate(discs):
            offset += i + 1
            discs[i] = (size, offset)
        t = 0
        while True:
            valid_ind = True
            for size, offset in discs:
                if (offset + t) % size != 0:
                    valid_ind = False
                    break
            if valid_ind:
                break
            t += 1
        result = t
        return result
    
    def solve2(self, discs):
        discs.append((11, 0))
        result = self.solve(discs)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        discs = self.get_discs(raw_input_lines)
        solutions = (
            self.solve(copy.deepcopy(discs)),
            self.solve2(copy.deepcopy(discs)),
            )
        result = solutions
        return result

class Day14: # One-Time Pad
    '''
    One-Time Pad
    https://adventofcode.com/2016/day/14
    '''
    def get_salt(self, raw_input_lines: List[str]):
        result = raw_input_lines[0]
        return result
    
    def solve(self, salt, repeat=1):
        keys = []
        index = 0
        hashes = collections.deque()
        quintuples_in_range = collections.defaultdict(int)
        while len(keys) < 64:
            message = salt + str(index)
            hash = message
            for _ in range(repeat + 1):
                hash = hashlib.md5(hash.encode('utf-8')).hexdigest()
            triples = set()
            for i in range(len(hash) - 2):
                candidate = hash[i: i + 3]
                if len(set(candidate)) == 1:
                    triple = candidate[0]
                    triples.add(triple)
                    break
            quintuples = set()
            for i in range(len(hash) - 4):
                candidate = hash[i: i + 5]
                if len(set(candidate)) == 1:
                    quintuple = candidate[0]
                    if quintuple not in quintuples:
                        quintuples_in_range[quintuple] += 1
                    quintuples.add(quintuple)
            hashes.append((index, hash, triples, quintuples))
            while len(hashes) >= 1000:
                key, hash, triples, quintuples = hashes.popleft()
                for quintuple in quintuples:
                    quintuples_in_range[quintuple] -= 1
                qs = set(
                    q for q, count in
                    quintuples_in_range.items() if
                    count > 0
                )
                if len(triples & qs) > 0:
                    keys.append(key)
            index += 1
        result = keys[64 - 1]
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        salt = self.get_salt(raw_input_lines)
        solutions = (
            self.solve(salt, 1),
            self.solve(salt, 2016),
            )
        result = solutions
        return result

class Day13: # A Maze of Twisty Little Cubicles
    '''
    A Maze of Twisty Little Cubicles
    https://adventofcode.com/2016/day/13
    '''
    offset = 0

    def get_parsed_input(self, raw_input_lines: List[str]):
        self.offset = int(raw_input_lines[0])
    
    def hamming(self, num):
        weight = 0
        while num > 0:
            weight += num & 1
            num >>= 1
        result = weight
        return result
    
    @functools.lru_cache(maxsize=1024)
    def wall(self, row, col):
        num = col ** 2 + 3 * col + 2 * col * row + row + row ** 2
        num += self.offset
        weight = self.hamming(num)
        result = weight % 2 == 1
        return result
    
    def solve(self, target_row, target_col):
        min_steps = float('inf')
        work = collections.deque()
        work.append((0, 1, 1))
        visited = set()
        while len(work) > 0:
            steps, row, col = work.pop()
            if (row, col) == (target_row, target_col):
                min_steps = steps
                break
            if (row, col) in visited:
                continue
            visited.add((row, col))
            for (next_row, next_col) in (
                (row + 1, col),
                (row - 1, col),
                (row, col + 1),
                (row, col - 1),
            ):
                if (
                    next_row < 0 or
                    next_col < 0 or
                    self.wall(next_row, next_col)
                ):
                    continue
                work.appendleft((steps + 1, next_row, next_col))
        result = min_steps
        return result
    
    def solve2(self):
        work = collections.deque()
        work.append((0, 1, 1))
        visited = set()
        while len(work) > 0:
            steps, row, col = work.pop()
            if steps > 50:
                continue
            if (row, col) in visited:
                continue
            visited.add((row, col))
            for (next_row, next_col) in (
                (row + 1, col),
                (row - 1, col),
                (row, col + 1),
                (row, col - 1),
            ):
                if (
                    next_row < 0 or
                    next_col < 0 or
                    self.wall(next_row, next_col)
                ):
                    continue
                work.appendleft((steps + 1, next_row, next_col))
        result = len(visited)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        parsed_input = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(39, 31),
            self.solve2(),
            )
        result = solutions
        return result

class Day12: # Leonardo's Monorail
    '''
    Leonardo's Monorail
    https://adventofcode.com/2016/day/12
    '''
    def solve(self, raw_input_lines):
        vm = AssembunnyVM()
        vm.load_raw_input(raw_input_lines)
        injections = {
            5: ('a += c, c = 0', 7),
        }
        vm.run()
        result = vm.registers['a']
        return result
    
    def solve2(self, raw_input_lines):
        vm = AssembunnyVM()
        vm.load_raw_input(raw_input_lines)
        vm.registers['c'] = 1
        vm.run()
        result = vm.registers['a']
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        solutions = (
            self.solve(raw_input_lines),
            self.solve2(raw_input_lines),
            )
        result = solutions
        return result

class Day11: # Radioisotope Thermoelectric Generators
    '''
    Radioisotope Thermoelectric Generators
    https://adventofcode.com/2016/day/11
    Rules:
    - You and the elevator start on the first floor
    - A microchip is powered only if it is on the same floor as its corresponding RTG
    - You cannot leave unpowered microchips in the same area as another RTG
    - The elevator can move between floors one at a time
    - The elevator can carry one or two devices at a time only
    Goal:
    - Find the minimum number of steps to get all objects to the fourth floor
    '''
    def get_state(self, raw_input_lines: List[str]):
        '''
        Each row contains the state of all types on that floor
        Floors:
            0 = What floor the elevator is on
            1 = 1st floor
            2 = 2nd floor
            3 = 3rd floor
            4 = 4th floor
        Types:
            0 = thulium
            1 = plutonium
            2 = promethium
            3 = ruthenium
            4 = strontium
        States:
            0 = No microchip or generator
            1 = Microchip only
            2 = Generator only
            3 = Both a microchip and a generator
        '''
        ignored_words = {
            'The',
            'floor',
            'contains',
            'a',
            'compatible',
            'and',
            'relevant',
        }
        floor_count = len(raw_input_lines)
        device_type_count = 0
        device_types = {}
        devices = set()
        for floor, raw_input_line in enumerate(raw_input_lines, start=1):
            line = ''.join(
                ' ' if char == '-' else char for char in
                raw_input_line if
                char not in '.,'
            )
            tokens = list(
                word for word in
                line.split(' ') if
                word not in ignored_words
            )
            for i, token in enumerate(tokens):
                if token in ('generator', 'microchip'):
                    device_type = tokens[i - 1]
                    if device_type not in device_types:
                        device_types[device_type] = device_type_count
                        device_type_count += 1
                    device_type_id = device_types[device_type]
                    device_value = 1
                    if token == 'generator':
                        device_value = 2
                    devices.add((floor, device_type_id, device_value))
        state = [[1]]
        for _ in range(floor_count):
            state.append([0] * device_type_count)
        for floor, device_type_id, device_value in devices:
            state[floor][device_type_id] += device_value
        result = state
        return result
    
    def solve_slow(self, initial_state, devices):
        '''
        States:
            0 = No microchip or generator (Protected, Harmless)
            1 = Microchip only (Unprotected, Harmless)
            2 = Generator only (Protected, Harmful)
            3 = Both a microchip and a generator (Protected, Harmful)
        No floor may contain a 1 if it contains either a 2 or a 3,
        which represents an unshielded microchip that is in the
        presence of another generator
        '''
        initial_state = tuple(tuple(data) for data in initial_state)
        seen = set()
        min_step_count = float('inf')
        work = collections.deque()
        work.append((0, initial_state))
        while len(work) > 0:
            step_count, state = work.pop()
            if state in seen:
                continue
            seen.add(state)
            # Do not allow unprotected devices to be
            # on the same floor as harmful ones
            valid_ind = True
            for floor in (1, 2, 3, 4):
                states = set(state[floor])
                if 1 in states and (2 in states or 3 in states):
                    valid_ind = False
                    break
            if not valid_ind:
                continue
            # The goal is to get all the devices to the fourth floor
            if set(state[4]) == {3}:
                min_step_count = step_count
                break
            floor = state[0][0]
            for device_count in (1, 2):
                # Exactly one or two devices from the current floor
                # can be taken each step
                choices = itertools.combinations(
                    devices,
                    device_count,
                )
                for choice in choices:
                    # The elevator moves up or down one floor per step
                    for next_floor in (floor - 1, floor + 1):
                        if next_floor < 1 or next_floor > 4:
                            continue
                        temp = []
                        for data in state:
                            temp.append(list(data))
                        temp[0][0] = next_floor
                        valid_ind = True
                        for dtype, dvalue in choice:
                            if temp[floor][dtype] & dvalue == 0:
                                valid_ind = False
                                break
                            temp[floor][dtype] -= dvalue
                            temp[next_floor][dtype] += dvalue
                        if valid_ind:
                            next_state = tuple(
                                tuple(data) for
                                data in temp
                            )
                            work.appendleft((step_count + 1, next_state))
        result = min_step_count
        return result
    
    def canonicalize(self, state) -> Tuple:
        canonical_state = []
        canonical_state.append([state[0][0]])
        transposed = list(map(list, zip(*state[1:])))
        for data in sorted(transposed):
            canonical_state.append(data)
        result = tuple(tuple(data) for data in canonical_state)
        return result
    
    def solve_fast(self, initial_state, devices):
        '''
        Use sorted transpose of the state when checking for states
        that have been seen before, since they are all equivalent
        '''
        initial_state = tuple(tuple(data) for data in initial_state)
        seen = set()
        min_step_count = float('inf')
        work = collections.deque()
        work.append((0, initial_state))
        max_step_count = 0
        while len(work) > 0:
            step_count, state = work.pop()
            canonical_state = self.canonicalize(state)
            if canonical_state in seen:
                continue
            seen.add(canonical_state)
            # Do not allow unprotected devices to be
            # on the same floor as harmful ones
            valid_ind = True
            for floor in (1, 2, 3, 4):
                states = set(state[floor])
                if 1 in states and (2 in states or 3 in states):
                    valid_ind = False
                    break
            if not valid_ind:
                continue
            # The goal is to get all the devices to the fourth floor
            if set(state[4]) == {3}:
                min_step_count = step_count
                break
            floor = state[0][0]
            for device_count in (1, 2):
                # Exactly one or two devices from the current floor
                # can be taken each step
                choices = itertools.combinations(
                    devices,
                    device_count,
                )
                for choice in choices:
                    # The elevator moves up or down one floor per step
                    for next_floor in (floor - 1, floor + 1):
                        if next_floor < 1 or next_floor > 4:
                            continue
                        temp = []
                        for data in state:
                            temp.append(list(data))
                        temp[0][0] = next_floor
                        valid_ind = True
                        for dtype, dvalue in choice:
                            if temp[floor][dtype] & dvalue == 0:
                                valid_ind = False
                                break
                            temp[floor][dtype] -= dvalue
                            temp[next_floor][dtype] += dvalue
                        if valid_ind:
                            next_state = tuple(
                                tuple(data) for
                                data in temp
                            )
                            work.appendleft((step_count + 1, next_state))
        result = min_step_count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        state = self.get_state(raw_input_lines)
        devices = [
            (0, 1), (0, 2),
            (1, 1), (1, 2),
            (2, 1), (2, 2),
            (3, 1), (3, 2),
            (4, 1), (4, 2),
        ]
        state2 = copy.deepcopy(state)
        devices2 = [
            (0, 1), (0, 2),
            (1, 1), (1, 2),
            (2, 1), (2, 2),
            (3, 1), (3, 2),
            (4, 1), (4, 2),
            (5, 1), (5, 2),
            (6, 1), (6, 2),
        ]
        device_type_count = len(state)
        for i in range(1, len(state2)):
            device_value = 0
            if i == 1:
                device_value = 3
            state2[i].append(device_value)
            state2[i].append(device_value)
        solutions = (
            self.solve_fast(state, devices),
            self.solve_fast(state2, devices2),
            )
        result = solutions
        return result

class Day10: # Balance Bots
    '''
    Balance Bots
    https://adventofcode.com/2016/day/10
    '''
    class Bot:
        def __init__(self, bot_id):
            self.bot_id = bot_id
            self.chips = []
            self.low_target = None
            self.high_target = None

    def get_bots(self, raw_input_lines: List[str]):
        bots = {}
        for raw_input_line in raw_input_lines:
            parts = raw_input_line.split(' ')
            if parts[0] == 'value':
                chip = int(parts[1])
                bot_id = int(parts[5])
                if bot_id not in bots:
                    bots[bot_id] = self.Bot(bot_id)
                heapq.heappush(bots[bot_id].chips, chip)
            elif parts[0] == 'bot':
                bot_id = int(parts[1])
                if bot_id not in bots:
                    bots[bot_id] = self.Bot(bot_id)
                low_type = parts[5]
                low_id = int(parts[6])
                bots[bot_id].low_target = (low_type, low_id)
                high_type = parts[10]
                high_id = int(parts[11])
                bots[bot_id].high_target = (high_type, high_id)
        result = bots
        return result
    
    def solve(self, bots):
        '''
        what is the number of the bot that is responsible for comparing
        value-61 microchips with value-17 microchips?
        '''
        target_bot_id = None
        while True:
            active_ind = False
            for bot in bots.values():
                if len(bot.chips) == 2:
                    if min(bot.chips) == 17 and max(bot.chips) == 61:
                        target_bot_id = bot.bot_id
                        break
                if (
                    len(bot.chips) == 2 and
                    bot.low_target is not None and
                    bot.high_target is not None
                ):
                    active_ind = True
                    low_chip = heapq.heappop(bot.chips)
                    high_chip = heapq.heappop(bot.chips)
                    if bot.low_target[0] == 'bot':
                        other_bot_id = bot.low_target[1]
                        heapq.heappush(bots[other_bot_id].chips, low_chip)
                    if bot.high_target[0] == 'bot':
                        other_bot_id = bot.high_target[1]
                        heapq.heappush(bots[other_bot_id].chips, high_chip)
            if not active_ind:
                break
        result = target_bot_id
        return result
    
    def solve2(self, bots):
        '''
        What is the value of multiplying outputs 0, 1, and 2?
        '''
        outputs = {}
        while True:
            active_ind = False
            for bot in bots.values():
                if (
                    len(bot.chips) == 2 and
                    bot.low_target is not None and
                    bot.high_target is not None
                ):
                    active_ind = True
                    low_chip = heapq.heappop(bot.chips)
                    high_chip = heapq.heappop(bot.chips)
                    if bot.low_target[0] == 'bot':
                        other_bot_id = bot.low_target[1]
                        heapq.heappush(bots[other_bot_id].chips, low_chip)
                    elif bot.low_target[0] == 'output':
                        output_id = bot.low_target[1]
                        outputs[output_id] = low_chip
                    if bot.high_target[0] == 'bot':
                        other_bot_id = bot.high_target[1]
                        heapq.heappush(bots[other_bot_id].chips, high_chip)
                    elif bot.high_target[0] == 'output':
                        output_id = bot.high_target[1]
                        outputs[output_id] = high_chip
            if not active_ind:
                break
        result = outputs[0] * outputs[1] * outputs[2]
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        bots = self.get_bots(raw_input_lines)
        solutions = (
            self.solve(copy.deepcopy(bots)),
            self.solve2(bots),
            )
        result = solutions
        return result

class Day09: # Explosives in Cyberspace
    '''
    Explosives in Cyberspace
    https://adventofcode.com/2016/day/9
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = raw_input_lines[0]
        return result
    
    def solve(self, parsed_input):
        cursor = 0
        length = 0
        while cursor < len(parsed_input):
            if parsed_input[cursor] == '(':
                cursor += 1
                size = 0
                while parsed_input[cursor] != 'x':
                    size = 10 * size + int(parsed_input[cursor])
                    cursor += 1
                cursor += 1
                repeat = 0
                while parsed_input[cursor] != ')':
                    repeat = 10 * repeat + int(parsed_input[cursor])
                    cursor += 1
                length += size * repeat
                cursor += size + 1
            else:
                length += 1
                cursor += 1
        result = length
        return result
    
    def solve2(self, parsed_input):
        '''
        (25x3)(3x3)ABC(2x3)XY(5x2)PQRSTX(18x9)(3x2)TWO(5x7)SEVEN
                                                   111     66666
                   999     99     666661           888     33333
        9*3 + 9*2 + 6*5 + 1 + 18*3 + 63*5 = 445
        '''
        counts = [0] * len(parsed_input)
        cursor = len(counts) - 1
        while cursor >= 0:
            if parsed_input[cursor] == ')':
                end = cursor
                while parsed_input[cursor] != '(':
                    cursor -= 1
                start = cursor + 1
                section = parsed_input[start:end]
                size, repeat = map(int, section.split('x'))
                for i in range(end + 1, end + size + 1):
                    counts[i] *= repeat
                cursor -= 1
            else:
                counts[cursor] = 1
                cursor -= 1
        result = sum(counts)
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

class Day08: # Two-Factor Authentication
    '''
    Two-Factor Authentication
    https://adventofcode.com/2016/day/8
    '''
    def get_instructions(self, raw_input_lines: List[str]):
        instructions = []
        for raw_input_line in raw_input_lines:
            tokens = raw_input_line.split(' ')
            instruction = None
            if tokens[0] == 'rect':
                pair = tokens[1].split('x')
                col_count = int(pair[0])
                row_count = int(pair[1])
                instruction = ('rect', col_count, row_count)
            elif tokens[1] == 'row':
                row = int(tokens[2].split('=')[1])
                shift_amount = int(tokens[4])
                instruction = ('rotate_row', row, shift_amount)
            elif tokens[1] == 'column':
                col = int(tokens[2].split('=')[1])
                shift_amount = int(tokens[4])
                instruction = ('rotate_col', col, shift_amount)
            instructions.append(instruction)
        result = instructions
        return result

    def get_display(self, instructions, rows, cols):
        display = {}
        for row in range(rows):
            for col in range(cols):
                display[(row, col)] = '.'
        for operation, a, b in instructions:
            if operation == 'rect':
                col_count, row_count = a, b
                for row in range(min(rows, row_count)):
                    for col in range(min(cols, col_count)):
                        display[(row, col)] = '#'
            elif operation == 'rotate_row':
                row, shift_amount = a, b
                shifted_row = {}
                for col in range(cols):
                    target_col = (col + shift_amount) % cols
                    shifted_row[target_col] = display[(row, col)]
                for col in range(cols):
                    display[(row, col)] = shifted_row[col]
            elif operation == 'rotate_col':
                col, shift_amount = a, b
                shifted_col = {}
                for row in range(rows):
                    target_row = (row + shift_amount) % rows
                    shifted_col[target_row] = display[(row, col)]
                for row in range(rows):
                    display[(row, col)] = shifted_col[row]
        result = display
        return result
    
    def solve(self, instructions):
        display = self.get_display(instructions, 6, 50)
        lit_pixel_count = sum(
            1 for
            cell in display.values() if
            cell == '#'
            )
        result = lit_pixel_count
        return result
    
    def solve2(self, instructions):
        display = self.get_display(instructions, 6, 50)
        display_string = []
        rows = max(row for row, col in display.keys())
        cols = max(col for row, col in display.keys())
        for row in range(rows + 1):
            row_data = []
            for col in range(cols + 1):
                cell = display[(row, col)]
                row_data.append(cell)
            display_string.append(''.join(row_data))
        result = '\n' + '\n'.join(display_string)
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

class Day07: # Internet Protocol Version 7
    '''
    Internet Protocol Version 7
    https://adventofcode.com/2016/day/7
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, parsed_input):
        tls_count = 0
        for line in parsed_input:
            supports_tls_ind = False
            hypernet_mode = False
            queue = collections.deque()
            for char in line:
                if char == '[':
                    hypernet_mode = True
                elif char == ']':
                    hypernet_mode = False
                queue.append(char)
                while len(queue) > 4:
                    queue.popleft()
                if len(queue) == 4:
                    if (
                        queue[0] == queue[3] and
                        queue[1] == queue[2] and
                        queue[0] != queue[1]
                    ):
                        if hypernet_mode:
                            supports_tls_ind = False
                            break
                        else:
                            supports_tls_ind = True
            if supports_tls_ind:
                tls_count += 1
        result = tls_count
        return result
    
    def solve2(self, parsed_input):
        ssl_count = 0
        for line in parsed_input:
            supernet_abas = set()
            hypernet_abas = set()
            mode = 'supernet'
            queue = collections.deque()
            for char in line:
                if char == '[':
                    mode = 'hypernet'
                elif char == ']':
                    mode = 'supernet'
                queue.append(char)
                while len(queue) > 3:
                    queue.popleft()
                if len(queue) == 3:
                    if (
                        queue[0] == queue[2] and
                        queue[0] != queue[1] and
                        queue[1] not in '[]'
                    ):
                        if mode == 'supernet':
                            supernet_abas.add(''.join(queue))
                        elif mode == 'hypernet':
                            hypernet_abas.add(''.join(queue))
            for aba in supernet_abas:
                bab = aba[1] + aba[0] + aba[1]
                if bab in hypernet_abas:
                    ssl_count += 1
                    break
        result = ssl_count
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

class Day06: # Signals and Noise
    '''
    Signals and Noise
    https://adventofcode.com/2016/day/6
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, parsed_input):
        cols = len(parsed_input[0])
        counts = []
        for word in parsed_input:
            for col, char in enumerate(word):
                counts.append(dict())
                if char not in counts[col]:
                    counts[col][char] = 0
                counts[col][char] += 1
        message = []
        for col in range(cols):
            most_common = [0, '?']
            for char, count in counts[col].items():
                if count > most_common[0]:
                    most_common[0] = count
                    most_common[1] = char
            message.append(most_common[1])
        result = ''.join(message)
        return result
    
    def solve2(self, parsed_input):
        cols = len(parsed_input[0])
        counts = []
        for word in parsed_input:
            for col, char in enumerate(word):
                counts.append(dict())
                if char not in counts[col]:
                    counts[col][char] = 0
                counts[col][char] += 1
        message = []
        for col in range(cols):
            least_common = [float('inf'), '?']
            for char, count in counts[col].items():
                if count < least_common[0]:
                    least_common[0] = count
                    least_common[1] = char
            message.append(least_common[1])
        result = ''.join(message)
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

class Day05: # How About a Nice Game of Chess?
    '''
    How About a Nice Game of Chess?
    https://adventofcode.com/2016/day/5
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = raw_input_lines[0]
        return result
    
    def solve(self, door_id):
        password = []
        num = 0
        while len(password) < 8:
            input_string = door_id + str(num)
            message = hashlib.md5(input_string.encode('utf-8')).hexdigest()
            if message[:5] == '00000':
                password.append(message[5])
            num += 1
        result = ''.join(password)
        return result
    
    def solve2(self, door_id):
        password = {}
        num = 0
        while len(password) < 8:
            input_string = door_id + str(num)
            message = hashlib.md5(input_string.encode('utf-8')).hexdigest()
            if message[:5] == '00000':
                position = int(message[5], base=16)
                if position not in password and position < 8:
                    password[position] = message[6]
            num += 1
        result = ''.join(
            password[position] for
            position in sorted(password.keys())
            )
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        door_id = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(door_id),
            self.solve2(door_id),
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
        5: (Day05, 'How About a Nice Game of Chess?'),
        6: (Day06, 'Signals and Noise'),
        7: (Day07, 'Internet Protocol Version 7'),
        8: (Day08, 'Two-Factor Authentication'),
        9: (Day09, 'Explosives in Cyberspace'),
       10: (Day10, 'Balance Bots'),
       11: (Day11, 'Radioisotope Thermoelectric Generators'),
       12: (Day12, 'Leonardo''s Monorail'),
       13: (Day13, 'A Maze of Twisty Little Cubicles'),
       14: (Day14, 'One-Time Pad'),
       15: (Day15, 'Timing is Everything'),
       16: (Day16, 'Dragon Checksum'),
       17: (Day17, 'Two Steps Forward'),
       18: (Day18, 'Like a Rogue'),
       19: (Day19, 'An Elephant Named Joseph'),
       20: (Day20, 'Firewall Rules'),
       21: (Day21, 'Scrambled Letters and Hash'),
       22: (Day22Incomplete, 'Grid Computing'),
       23: (Day23, 'Safe Cracking'),
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
