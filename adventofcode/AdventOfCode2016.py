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

class Day12: # Leonardo's Monorail
    '''
    Leonardo's Monorail
    https://adventofcode.com/2016/day/12
    '''
    def get_instructions(self, raw_input_lines: List[str]):
        instructions = []
        for raw_input_line in raw_input_lines:
            parts = raw_input_line.split(' ')
            for i in range(len(parts)):
                try:
                    parts[i] = int(parts[i])
                except ValueError:
                    pass
            instruction = tuple(parts)
            instructions.append(instruction)
        result = instructions
        return result

    def run(self, registers, instructions):
        pc = 0
        while pc < len(instructions):
            instruction = instructions[pc]
            op = instruction[0]
            if op == 'cpy':
                x = instruction[1]
                y = instruction[2]
                x_val = x if type(x) is int else registers[x]
                registers[y] = x_val
            elif op == 'inc':
                x = instruction[1]
                registers[x] += 1
            elif op == 'dec':
                x = instruction[1]
                registers[x] -= 1
            elif op == 'jnz':
                x = instruction[1]
                y = instruction[2]
                x_val = x if type(x) is int else registers[x]
                if x_val != 0:
                    pc += y - 1
            pc += 1
        result = registers['a']
        return result
    
    def solve(self, instructions):
        registers = {'a': 0, 'b': 0, 'c': 0, 'd': 0}
        self.run(registers, instructions)
        result = registers['a']
        return result
    
    def solve2(self, instructions):
        registers = {'a': 0, 'b': 0, 'c': 0, 'd': 0}
        registers['c'] = 1
        self.run(registers, instructions)
        result = registers['a']
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
