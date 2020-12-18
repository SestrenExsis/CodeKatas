'''
Created on Nov 24, 2020

@author: Sestren
'''
import argparse
import collections
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
    https://adventofcode.com/2020/day/?
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

class Day18: # Operation Order
    '''
    Operation Order
    https://adventofcode.com/2020/day/18
    '''
    def tokenize(self, chars):
        tokens = []
        num = 0
        for i, char in enumerate(chars + ' '):
            if char not in '0123456789':
                if i > 0 and chars[i - 1] in '0123456789':
                    tokens.append(num)
                    num = 0
            if char == ' ':
                continue
            elif char in '0123456789':
                num = 10 * num + int(char)
            else:
                tokens.append(char)
        result = tokens
        return result

    def get_parsed_input(self, raw_input_lines):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(self.tokenize(raw_input_line))
        return result
    
    def solve(self, parsed_input):
        # Use Shunting-Yard Algorithm
        # https://en.wikipedia.org/wiki/Shunting-yard_algorithm
        totals = []
        for tokens in parsed_input:
            output = []
            operators = []
            for token in tokens:
                if type(token) is int:
                    output.append(token)
                elif token in '+*':
                    while (
                        len(operators) > 0 and
                        operators[-1] + token in ('++', '**', '*+', '+*')
                        ):
                        output.append(operators.pop())
                    operators.append(token)
                elif token == '(':
                    operators.append(token)
                elif token == ')':
                    while operators[-1] != '(':
                        output.append(operators.pop())
                    if operators[-1] == '(':
                        operators.pop()
            while len(operators) > 0:
                output.append(operators.pop())
            stack = []
            for el in output:
                stack.append(el)
                if type(stack[-1]) is str and stack[-1] in '+*':
                    operator = stack.pop()
                    a = stack.pop()
                    b = stack.pop()
                    if operator == '+':
                        stack.append(a + b)
                    elif operator == '*':
                        stack.append(a * b)
            totals.append(sum(stack))
        result = sum(totals)
        return result
    
    def solve2(self, parsed_input):
        # Use Shunting-Yard Algorithm
        # https://en.wikipedia.org/wiki/Shunting-yard_algorithm
        totals = []
        for tokens in parsed_input:
            output = []
            operators = []
            for token in tokens:
                if type(token) is int:
                    output.append(token)
                elif token in '+*':
                    while (
                        len(operators) > 0 and
                        operators[-1] + token in ('++', '**', '+*')
                        ):
                        output.append(operators.pop())
                    operators.append(token)
                elif token == '(':
                    operators.append(token)
                elif token == ')':
                    while operators[-1] != '(':
                        output.append(operators.pop())
                    if operators[-1] == '(':
                        operators.pop()
            while len(operators) > 0:
                output.append(operators.pop())
            stack = []
            for el in output:
                stack.append(el)
                if type(stack[-1]) is str and stack[-1] in '+*':
                    operator = stack.pop()
                    a = stack.pop()
                    b = stack.pop()
                    if operator == '+':
                        stack.append(a + b)
                    elif operator == '*':
                        stack.append(a * b)
            totals.append(sum(stack))
        result = sum(totals)
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

class Day17: # Conway Cubes
    '''
    Conway Cubes
    https://adventofcode.com/2020/day/17
    '''
    def get_initial_state(self, raw_input_lines: List[str]):
        initial_state = set()
        for y, raw_input_line in enumerate(raw_input_lines):
            for x, cell in enumerate(raw_input_line):
                if cell == '#':
                    initial_state.add((x, y, 0))
        result = initial_state
        return result
    
    def solve(self, initial_state):
        curr_state = set(initial_state)
        for _ in range(6):
            neighbors = collections.defaultdict(int)
            for x, y, z in curr_state:
                for (dx, dy, dz) in (
                    (-1, -1, -1),
                    (-1, -1,  0),
                    (-1, -1,  1),
                    (-1,  0, -1),
                    (-1,  0,  0),
                    (-1,  0,  1),
                    (-1,  1, -1),
                    (-1,  1,  0),
                    (-1,  1,  1),
                    ( 0, -1, -1),
                    ( 0, -1,  0),
                    ( 0, -1,  1),
                    ( 0,  0, -1),
                    # ( 0,  0,  0),
                    ( 0,  0,  1),
                    ( 0,  1, -1),
                    ( 0,  1,  0),
                    ( 0,  1,  1),
                    ( 1, -1, -1),
                    ( 1, -1,  0),
                    ( 1, -1,  1),
                    ( 1,  0, -1),
                    ( 1,  0,  0),
                    ( 1,  0,  1),
                    ( 1,  1, -1),
                    ( 1,  1,  0),
                    ( 1,  1,  1),
                    ):
                    neighbors[(x + dx, y + dy, z + dz)] += 1
            next_state = set()
            for x, y, z in curr_state:
                if 2 <= neighbors[(x, y, z)] <= 3:
                    next_state.add((x, y, z))
            for x, y, z in neighbors:
                if neighbors[(x, y, z)] == 3 and (x, y, z) not in curr_state:
                    next_state.add((x, y, z))
            curr_state = next_state
        result = len(curr_state)
        return result
    
    def solve2(self, initial_state):
        curr_state = set()
        for (x, y, z) in initial_state:
            curr_state.add((x, y, z, 0))
        for _ in range(6):
            neighbors = collections.defaultdict(int)
            for x, y, z, w in curr_state:
                for (dx, dy, dz, dw) in (
                    (-1, -1, -1, -1),
                    (-1, -1,  0, -1),
                    (-1, -1,  1, -1),
                    (-1,  0, -1, -1),
                    (-1,  0,  0, -1),
                    (-1,  0,  1, -1),
                    (-1,  1, -1, -1),
                    (-1,  1,  0, -1),
                    (-1,  1,  1, -1),
                    ( 0, -1, -1, -1),
                    ( 0, -1,  0, -1),
                    ( 0, -1,  1, -1),
                    ( 0,  0, -1, -1),
                    ( 0,  0,  0, -1),
                    ( 0,  0,  1, -1),
                    ( 0,  1, -1, -1),
                    ( 0,  1,  0, -1),
                    ( 0,  1,  1, -1),
                    ( 1, -1, -1, -1),
                    ( 1, -1,  0, -1),
                    ( 1, -1,  1, -1),
                    ( 1,  0, -1, -1),
                    ( 1,  0,  0, -1),
                    ( 1,  0,  1, -1),
                    ( 1,  1, -1, -1),
                    ( 1,  1,  0, -1),
                    ( 1,  1,  1, -1),
                    
                    (-1, -1, -1,  0),
                    (-1, -1,  0,  0),
                    (-1, -1,  1,  0),
                    (-1,  0, -1,  0),
                    (-1,  0,  0,  0),
                    (-1,  0,  1,  0),
                    (-1,  1, -1,  0),
                    (-1,  1,  0,  0),
                    (-1,  1,  1,  0),
                    ( 0, -1, -1,  0),
                    ( 0, -1,  0,  0),
                    ( 0, -1,  1,  0),
                    ( 0,  0, -1,  0),
                    # ( 0,  0,  0,  0),
                    ( 0,  0,  1,  0),
                    ( 0,  1, -1,  0),
                    ( 0,  1,  0,  0),
                    ( 0,  1,  1,  0),
                    ( 1, -1, -1,  0),
                    ( 1, -1,  0,  0),
                    ( 1, -1,  1,  0),
                    ( 1,  0, -1,  0),
                    ( 1,  0,  0,  0),
                    ( 1,  0,  1,  0),
                    ( 1,  1, -1,  0),
                    ( 1,  1,  0,  0),
                    ( 1,  1,  1,  0),
                    
                    (-1, -1, -1,  1),
                    (-1, -1,  0,  1),
                    (-1, -1,  1,  1),
                    (-1,  0, -1,  1),
                    (-1,  0,  0,  1),
                    (-1,  0,  1,  1),
                    (-1,  1, -1,  1),
                    (-1,  1,  0,  1),
                    (-1,  1,  1,  1),
                    ( 0, -1, -1,  1),
                    ( 0, -1,  0,  1),
                    ( 0, -1,  1,  1),
                    ( 0,  0, -1,  1),
                    ( 0,  0,  0,  1),
                    ( 0,  0,  1,  1),
                    ( 0,  1, -1,  1),
                    ( 0,  1,  0,  1),
                    ( 0,  1,  1,  1),
                    ( 1, -1, -1,  1),
                    ( 1, -1,  0,  1),
                    ( 1, -1,  1,  1),
                    ( 1,  0, -1,  1),
                    ( 1,  0,  0,  1),
                    ( 1,  0,  1,  1),
                    ( 1,  1, -1,  1),
                    ( 1,  1,  0,  1),
                    ( 1,  1,  1,  1),
                    ):
                    neighbors[(x + dx, y + dy, z + dz, w + dw)] += 1
            next_state = set()
            for x, y, z, w in curr_state:
                if 2 <= neighbors[(x, y, z, w)] <= 3:
                    next_state.add((x, y, z, w))
            for x, y, z, w in neighbors:
                if (
                    neighbors[(x, y, z, w)] == 3 and
                    (x, y, z, w) not in curr_state
                ):
                    next_state.add((x, y, z, w))
            curr_state = next_state
        result = len(curr_state)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        initial_state = self.get_initial_state(raw_input_lines)
        solutions = (
            self.solve(initial_state),
            self.solve2(initial_state),
            )
        result = solutions
        return result

class Day16: # Ticket Translation
    '''
    Ticket Translation
    https://adventofcode.com/2020/day/16
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        rules = {}
        your_ticket = []
        nearby_tickets = []
        mode = 'rules'
        for raw_input_line in raw_input_lines:
            if ': ' in raw_input_line:
                field, raw_suffix = raw_input_line.split(': ')
                rules[field] = []
                valid_ranges = raw_suffix.split(' or ')
                for valid_range in valid_ranges:
                    rule = tuple(map(int, valid_range.split('-')))
                    rules[field].append(rule)
            else:
                if len(raw_input_line) == 0:
                    continue
                if raw_input_line in (
                    'your ticket:',
                    'nearby tickets:',
                    ):
                    mode = raw_input_line[:-1]
                elif mode == 'your ticket':
                    your_ticket = list(map(int, raw_input_line.split(',')))
                elif mode == 'nearby tickets':
                    nearby_tickets.append(
                        list(map(int, raw_input_line.split(',')))
                        )
        result = (
            rules,
            your_ticket,
            nearby_tickets,
        )
        return result
    
    def ticket_errors(self,
        ticket: List[int],
        rules: Dict[str, List[Tuple[int]]],
        ):
        errors = []
        for value in ticket:
            valid_ind = False
            for valid_ranges in rules.values():
                for valid_range in valid_ranges:
                    min_val, max_val = valid_range
                    if min_val <= value <= max_val:
                        valid_ind = True
                        break
                if valid_ind:
                    break
            if not valid_ind:
                errors.append(value)
        result = errors
        return result
    
    def solve(self,
        rules: Dict[str, List[Tuple[int]]],
        nearby_tickets: List[List[int]],
        ):
        errors = []
        for ticket in nearby_tickets:
            errors.extend(self.ticket_errors(ticket, rules))
        result = sum(errors)
        return result
    
    def solve2(self,
        rules: Dict[str, List[Tuple[int]]],
        your_ticket: List[int],
        nearby_tickets: List[List[int]],
        ):
        valid_tickets = list(
            ticket for
            ticket in nearby_tickets if
            len(self.ticket_errors(ticket, rules)) == 0
            )
        fields = {}
        for field in rules:
            fields[field] = set(range(len(your_ticket)))
        for field in fields:
            valid_ranges = rules[field]
            for field_id in range(len(your_ticket)):
                possible_ind = True
                for ticket in valid_tickets:
                    valid_ind = False
                    for valid_range in valid_ranges:
                        min_val, max_val = valid_range
                        if min_val <= ticket[field_id] <= max_val:
                            valid_ind = True
                            break
                    if not valid_ind:
                        possible_ind = False
                        break
                if not possible_ind:
                    fields[field].remove(field_id)
        fixed_fields = {}
        while len(fixed_fields) < len(your_ticket):
            fixed_field_id = -1
            for field, possible_ids in fields.items():
                if len(possible_ids) == 1:
                    fixed_field_id = next(iter(possible_ids))
                    break
            if fixed_field_id >= 0:
                fixed_fields[field] = fixed_field_id
                for field in fields:
                    if fixed_field_id in fields[field]:
                        fields[field].remove(fixed_field_id)
        result = functools.reduce(operator.mul,(
            your_ticket[fixed_fields['departure location']],
            your_ticket[fixed_fields['departure station']],
            your_ticket[fixed_fields['departure platform']],
            your_ticket[fixed_fields['departure track']],
            your_ticket[fixed_fields['departure date']],
            your_ticket[fixed_fields['departure time']],
            ), 1)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        parsed_input = self.get_parsed_input(raw_input_lines)
        rules, your_ticket, nearby_tickets = parsed_input
        solutions = (
            self.solve(rules, nearby_tickets),
            self.solve2(rules, your_ticket, nearby_tickets),
            )
        result = solutions
        return result

class Day15: # Rambunctious Recitation
    '''
    Rambunctious Recitation
    https://adventofcode.com/2020/day/15
    '''
    def get_starting_numbers(self, raw_input_lines: List[str]):
        result = list(map(int, raw_input_lines[0].split(',')))
        return result
    
    def bruteforce(self, starting_numbers, turns):
        spoken = starting_numbers[0]
        prev_spoken = None
        last_spoken = collections.defaultdict(collections.deque)
        for i in range(turns):
            if i < len(starting_numbers):
                spoken = starting_numbers[i]
            else:
                if len(last_spoken[prev_spoken]) == 1:
                    spoken = 0
                else:
                    times = last_spoken[prev_spoken]
                    spoken = times[-1] - times[-2]
            last_spoken[spoken].append(i)
            while len(last_spoken[spoken]) > 2:
                last_spoken[spoken].popleft()
            prev_spoken = spoken
        result = spoken
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        starting_numbers = self.get_starting_numbers(raw_input_lines)
        solutions = (
            self.bruteforce(starting_numbers, 2_020),
            self.bruteforce(starting_numbers, 30_000_000),
            )
        result = solutions
        return result

class Day14: # Docking Data
    '''
    Docking Data
    https://adventofcode.com/2020/day/14
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            a, b = raw_input_line.split(' = ')
            if a == 'mask':
                result.append((a, b))
            else:
                c = a.split('[')[1][:-1]
                result.append(('mem', int(c), int(b)))
        return result
    
    def solve(self, parsed_input):
        mem = collections.defaultdict(int)
        mask = 'X' * 36
        for row in parsed_input:
            if row[0] == 'mask':
                mask = row[1]
            elif row[0] == 'mem':
                num = row[2]
                value = 0
                for i in range(36):
                    power = 2 ** (36 - i - 1)
                    if mask[i] == '1' or mask[i] == 'X' and num & power > 0:
                        value += power
                mem[row[1]] = value
        result = sum(mem.values())
        return result

    def gen_addresses(self, mask):
        if 'X' in mask:
            for char in ('0', '1'):
                yield from self.gen_addresses(mask.replace('X', char, 1))
        else:
            yield mask
    
    def solve2(self, parsed_input):
        mem = collections.defaultdict(int)
        mask = 'X' * 36
        for row in parsed_input:
            if row[0] == 'mask':
                mask = row[1]
            elif row[0] == 'mem':
                masked_address = []
                for i in range(len(mask)):
                    if mask[i] == '0':
                        power = 2 ** (35 - i)
                        bit = 1 if row[1] & power > 0 else 0
                        masked_address.append(str(bit))
                    else:
                        masked_address.append(mask[i])
                masked_address = ''.join(masked_address)
                for address_str in self.gen_addresses(masked_address):
                    address = int(address_str, 2)
                    mem[address] = row[2]
        result = sum(mem.values())
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

class Day13: # Shuttle Search
    '''
    Shuttle Search
    https://adventofcode.com/2020/day/13
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        earliest_departure_time = int(raw_input_lines[0])
        buses = set()
        for i, bus_id in enumerate(raw_input_lines[1].split(',')):
            if bus_id != 'x':
                bus_id = int(bus_id)
                buses.add((bus_id, (bus_id - (i % bus_id)) % bus_id))
        result = (earliest_departure_time, buses)
        return result
    
    def solve(self, earliest_departure_time, buses):
        min_wait_time = float('inf')
        earliest_bus_id = 0
        for bus_id, _ in buses:
            wait_time = bus_id - (earliest_departure_time % bus_id)
            if wait_time < min_wait_time:
                min_wait_time = wait_time
                earliest_bus_id = bus_id
        result = earliest_bus_id * min_wait_time
        return result
    
    def solve2(self, earliest_departure_time, buses):
        modulo, remainder = buses.pop()
        while len(buses) > 0:
            next_modulo, next_remainder = buses.pop()
            while remainder % next_modulo != next_remainder:
                remainder += modulo
            modulo *= next_modulo
        result = remainder
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        earliest_departure_time, buses = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(earliest_departure_time, buses),
            self.solve2(earliest_departure_time, buses),
            )
        result = solutions
        return result

class Day12: # Rain Risk
    '''
    Rain Risk
    https://adventofcode.com/2020/day/12
    '''
    def get_instructions(self, raw_input_lines: List[str]):
        instructions = []
        for raw_input_line in raw_input_lines:
            instruction = (raw_input_line[0], int(raw_input_line[1:]))
            instructions.append(instruction)
        result = instructions
        return result
    
    def solve(self, instructions):
        facings = ['N', 'E', 'S', 'W']
        facing = 1 # ship
        offsets = {
            'N': (-1, 0),
            'S': ( 1, 0),
            'W': ( 0,-1),
            'E': ( 0, 1),
        }
        position = (0, 0) # NS, WE
        for instruction, amount in instructions:
            offset = None
            if instruction == 'F':
                offset = offsets[facings[facing]]
            elif instruction == 'L':
                facing = (facing - amount // 90) % len(facings)
            elif instruction == 'R':
                facing = (facing + amount // 90) % len(facings)
            elif instruction in offsets:
                offset = offsets[instruction]
            if offset is not None:
                position = (
                    position[0] + amount * offset[0],
                    position[1] + amount * offset[1],
                )
        result = sum(map(abs, position))
        return result
    
    def solve2(self, instructions):
        offsets = {
            'N': (-1, 0),
            'S': ( 1, 0),
            'W': ( 0,-1),
            'E': ( 0, 1),
        }
        waypoint = (-1, 10) # NS, WE
        ship = (0, 0) # NS, WE
        for instruction, amount in instructions:
            if instruction == 'F':
                ship = (
                    ship[0] + amount * waypoint[0],
                    ship[1] + amount * waypoint[1],
                )
            elif instruction == 'L':
                rotation_count = amount // 90
                for _ in range(rotation_count):
                    waypoint = (-waypoint[1], waypoint[0])
            elif instruction == 'R':
                rotation_count = amount // 90
                for _ in range(rotation_count):
                    waypoint = (waypoint[1], -waypoint[0])
            elif instruction in offsets:
                offset = offsets[instruction]
                waypoint = (
                    waypoint[0] + amount * offset[0],
                    waypoint[1] + amount * offset[1],
                )
        result = sum(map(abs, ship))
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

class Day11: # Seating System
    '''
    Seating System
    https://adventofcode.com/2020/day/11
    '''
    def get_seats(self, raw_input_lines: List[List[str]]):
        result = []
        for raw_input_line in raw_input_lines:
            rowdata = []
            for seat in raw_input_line:
                rowdata.append(seat)
            result.append(rowdata)
        return result
    
    def solve(self, seats):
        rows = len(seats)
        cols = len(seats[0])
        seats = [row[:] for row in seats]
        next_seats = [row[:] for row in seats]
        change_ind = True
        while change_ind:
            change_ind = False
            for row in range(rows):
                for col in range(cols):
                    occupied_count = 0
                    for (r, c) in (
                        (row - 1, col - 1),
                        (row - 1, col + 0),
                        (row - 1, col + 1),
                        (row + 0, col - 1),
                        (row + 0, col + 1),
                        (row + 1, col - 1),
                        (row + 1, col + 0),
                        (row + 1, col + 1),
                    ):
                        if (
                            0 <= r < rows and
                            0 <= c < cols and
                            seats[r][c] == '#'
                        ):
                            occupied_count += 1
                    if seats[row][col] == 'L':
                        if occupied_count == 0:
                            next_seats[row][col] = '#'
                            change_ind = True
                    elif seats[row][col] == '#':
                        if occupied_count >= 4:
                            next_seats[row][col] = 'L'
                            change_ind = True
            for row in range(rows):
                for col in range(cols):
                    seats[row][col] = next_seats[row][col]
        occupied_count = 0
        for row in range(rows):
            for col in range(cols):
                if seats[row][col] == '#':
                    occupied_count += 1
        result = occupied_count
        return result
    
    def solve2(self, seats):
        rows = len(seats)
        cols = len(seats[0])
        seats = [row[:] for row in seats]
        next_seats = [row[:] for row in seats]
        change_ind = True
        while change_ind:
            change_ind = False
            for row in range(rows):
                for col in range(cols):
                    occupied_count = 0
                    for (dr, dc) in (
                        (-1, -1),
                        (-1,  0),
                        (-1,  1),
                        ( 0, -1),
                        ( 0,  1),
                        ( 1, -1),
                        ( 1,  0),
                        ( 1,  1),
                    ):
                        dist = 1
                        while True:
                            if (
                                (row + dist * dr) < 0 or
                                (row + dist * dr) >= rows or
                                (col + dist * dc) < 0 or
                                (col + dist * dc) >= cols
                            ):
                                break
                            seat = seats[row + dist * dr][col + dist * dc]
                            if seat == '#':
                                occupied_count += 1
                                break
                            elif seat == 'L':
                                break
                            dist += 1
                    if seats[row][col] == 'L':
                        if occupied_count == 0:
                            next_seats[row][col] = '#'
                            change_ind = True
                    elif seats[row][col] == '#':
                        if occupied_count >= 5:
                            next_seats[row][col] = 'L'
                            change_ind = True
            for row in range(rows):
                for col in range(cols):
                    seats[row][col] = next_seats[row][col]
        occupied_count = 0
        for row in range(rows):
            for col in range(cols):
                if seats[row][col] == '#':
                    occupied_count += 1
        result = occupied_count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        seats = self.get_seats(raw_input_lines)
        solutions = (
            self.solve(seats),
            self.solve2(seats),
            )
        result = solutions
        return result

class Day10: # Adapter Array
    '''
    https://adventofcode.com/2020/day/10
    '''
    def get_adapters(self, raw_input_lines: List[str]) -> Set[int]:
        result = set()
        for raw_input_line in raw_input_lines:
            result.add(int(raw_input_line))
        return result
    
    def solve(self, adapters: Set[int]) -> int:
        adapters_left = set(adapters)
        chain = [0]
        while len(adapters_left) > 0:
            adapter = min(adapters_left)
            adapters_left.remove(adapter)
            chain.append(adapter)
        chain.append(chain[-1] + 3)
        diffs = [0, 0, 0, 0]
        for i in range(1, len(chain)):
            diff = chain[i] - chain[i - 1]
            assert 1 <= diff <= 3
            diffs[diff] += 1
        # sum of 1-jolt differences multiplied by sum of 3-jolt differences
        result = diffs[1] * diffs[3]
        return result
    
    def solve2(self, adapters: Set[int]) -> int:
        dp = [1] + [0] * max(adapters)
        for adapter in sorted(adapters):
            if adapter >= 1:
                dp[adapter] += dp[adapter - 1]
            if adapter >= 2:
                dp[adapter] += dp[adapter - 2]
            if adapter >= 3:
                dp[adapter] += dp[adapter - 3]
        result = dp[-1]
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        adapters = self.get_adapters(raw_input_lines)
        solutions = (
            self.solve(adapters),
            self.solve2(adapters),
            )
        result = solutions
        return result

class Day09: # Encoding Error
    '''
    Encoding Error
    https://adventofcode.com/2020/day/9
    '''
    def get_numbers(self, raw_input_lines: List[str]) -> List[int]:
        numbers = []
        for raw_input_line in raw_input_lines:
            numbers.append(int(raw_input_line))
        result = numbers
        return result
    
    def solve(self, numbers: List[int], span: int=25) -> int:
        invalid_num = None
        for i in range(span, len(numbers)):
            complements = set()
            for offset in range(1, span + 1):
                complement = numbers[i] - numbers[i - offset]
                if numbers[i - offset] in complements:
                    break
                complements.add(complement)
            else:
                invalid_num = numbers[i]
                break
        result = invalid_num
        return result
    
    def solve2(self, numbers: List[int], target: int) -> int:
        queue = collections.deque()
        left = 0
        right = 0
        queue.append(numbers[0])
        while True:
            if sum(queue) == target:
                break
            while right < len(numbers) and sum(queue) < target:
                right += 1
                queue.append(numbers[right])
            while left < right and sum(queue) > target:
                left += 1
                queue.popleft()
        result = min(queue) + max(queue)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        numbers = self.get_numbers(raw_input_lines)
        target = self.solve(numbers, 25)
        solutions = (
            target,
            self.solve2(numbers, target),
            )
        result = solutions
        return result

class Day08: # Handheld Halting
    '''
    Handheld Halting
    https://adventofcode.com/2020/day/8
    '''
    def get_instructions(self, raw_input_lines: List[str]) -> List[str]:
        instructions = []
        for raw_input_line in raw_input_lines:
            operation, raw_argument = raw_input_line.split(' ')
            instruction = (operation, int(raw_argument))
            instructions.append(instruction)
        result = instructions
        return result
    
    def solve(self, instructions: List[Tuple[str, int]]) -> int:
        acc = 0
        pc = 0
        seen = set()
        while True:
            if pc in seen or pc >= len(instructions):
                break
            seen.add(pc)
            operation, argument = instructions[pc]
            if operation == 'nop':
                pc += 1
            elif operation == 'acc':
                acc += argument
                pc += 1
            elif operation == 'jmp':
                pc += argument
        result = acc
        return result
    
    def solve2(self, instructions: List[Tuple[str, int]]) -> int:
        swapped = {
            'acc': 'acc',
            'nop': 'jmp',
            'jmp': 'nop',
        }
        acc = 0
        for i in range(len(instructions)):
            if instructions[i][0] not in ('nop', 'jmp'):
                continue
            acc = 0
            pc = 0
            seen = set()
            halted = False
            while True:
                if pc >= len(instructions):
                    halted = True
                    break
                if pc in seen:
                    break
                seen.add(pc)
                operation, argument = instructions[pc]
                if pc == i:
                    operation = swapped[operation]
                if operation == 'nop':
                    pc += 1
                elif operation == 'acc':
                    acc += argument
                    pc += 1
                elif operation == 'jmp':
                    pc += argument
            if halted:
                break
        result = acc
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

class Day07: # Handy Haversacks
    '''
    Handy Haversacks
    https://adventofcode.com/2020/day/7
    '''
    def get_bags(self, raw_input_lines: List[str]) -> List[str]:
        bags = {}
        for raw_input_line in raw_input_lines:
            bag, raw_contents = raw_input_line.split(' bags contain ')
            contents = {}
            if raw_contents != 'no other bags.':
                raw_contents = raw_contents.split(', ')
                for raw_content in raw_contents:
                    (raw_content)
                    words = raw_content.split(' ')
                    contents[' '.join(words[1:3])] = int(words[0])
            bags[bag] = contents
        result = bags
        return result

    def solve(self, bags: List[str]) -> int:
        containing_bags = set()
        work = ['shiny gold']
        while len(work) > 0:
            curr_bag = work.pop()
            for bag, contents in bags.items():
                for content in contents:
                    if content == curr_bag:
                        if bag not in containing_bags:
                            work.append(bag)
                        containing_bags.add(bag)
        result = len(containing_bags)
        return result
    
    def solve2(self, bags: List[str]) -> int:
        bag_count = 0
        work = [(1, 'shiny gold')]
        while len(work) > 0:
            curr_count, curr_bag = work.pop()
            contents = bags[curr_bag]
            for next_bag, next_count in contents.items():
                work.append((curr_count * next_count, next_bag))
            bag_count += curr_count
        result = bag_count - 1
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        bags = self.get_bags(raw_input_lines)
        solutions = (
            self.solve(bags),
            self.solve2(bags),
            )
        result = solutions
        return result

class Day06: # Custom Customs
    '''
    Custom Customs
    https://adventofcode.com/2020/day/6
    '''
    def get_groups(self, raw_input_lines: List[str]) -> List[List[str]]:
        groups = []
        group = []
        for raw_input_line in raw_input_lines:
            if len(raw_input_line) < 1:
                groups.append(group)
                group = []
            else:
                group.append(raw_input_line)
        groups.append(group)
        result = groups
        return result
    
    def solve(self, groups: List[List[str]]) -> int:
        answer_count = 0
        for group in groups:
            answers = set()
            for person in group:
                for answer in person:
                    answers.add(answer)
            answer_count += len(answers)
        result = answer_count
        return result
    
    def solve2(self, groups: List[List[str]]) -> int:
        answer_count = 0
        for group in groups:
            person_count = len(group)
            answers = collections.defaultdict(int)
            for person in group:
                for answer in person:
                    answers[answer] += 1
            for answer, count in answers.items():
                if count == person_count:
                    answer_count += 1
        result = answer_count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        groups = self.get_groups(raw_input_lines)
        solutions = (
            self.solve(groups),
            self.solve2(groups),
            )
        result = solutions
        return result

class Day05: # Binary Boarding
    '''
    Binary Boarding
    https://adventofcode.com/2020/day/5
    '''
    def get_parsed_input(self, raw_input_lines: List[str]) -> List[str]:
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, parsed_input: List[str]) -> int:
        seat_ids = []
        for code in parsed_input:
            row = 0
            row_val = 64
            for char in code[:7]:
                if char == 'B':
                    row += row_val
                row_val //= 2
            col = 0
            col_val = 4
            for char in code[7:]:
                if char == 'R':
                    col += col_val
                col_val //= 2
            seat_id = 8 * row + col
            seat_ids.append(seat_id)
        result = max(seat_ids)
        return result
    
    def solve2(self, parsed_input: List[str]) -> int:
        seat_ids = set()
        for code in parsed_input:
            row = 0
            row_val = 64
            for char in code[:7]:
                if char == 'B':
                    row += row_val
                row_val //= 2
            col = 0
            col_val = 4
            for char in code[7:]:
                if char == 'R':
                    col += col_val
                col_val //= 2
            seat_id = 8 * row + col
            seat_ids.add(seat_id)
        for middle_seat in range(min(seat_ids), max(seat_ids)):
            if all([
                middle_seat - 1 in seat_ids,
                middle_seat not in seat_ids,
                middle_seat + 1 in seat_ids,
            ]):
                result = middle_seat
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

class Day04: # Passport Processing
    '''
    Passport Processing
    https://adventofcode.com/2020/day/4
    '''
    def get_passports(self, raw_input_lines: List[str]) -> List[str]:
        passports = []
        passport = {}
        for raw_input_line in raw_input_lines:
            if len(raw_input_line) < 1:
                passports.append(passport)
                passport = {}
            else:
                pairs = raw_input_line.split(' ')
                for pair in pairs:
                    key, val = pair.split(':')
                    passport[key] = val
        passports.append(passport)
        result = passports
        return result
    
    def solve(self, passports: List[str]) -> int:
        valid_passport_count = 0
        for passport in passports:
            if all(key in passport for key in (
                'byr',
                'iyr',
                'eyr',
                'hgt',
                'hcl',
                'ecl',
                'pid',
                )):
                valid_passport_count += 1
        result = valid_passport_count
        return result
    
    def solve2(self, passports: List[str]) -> int:
        valid_passport_count = 0
        for passport in passports:
            try:
                # Validate birth year
                birth_year = int(passport['byr'])
                assert 1920 <= birth_year <= 2002
                # Validate issue year
                issue_year = int(passport['iyr'])
                assert 2010 <= issue_year <= 2020
                # Validate expiration year
                expiration_year = int(passport['eyr'])
                assert 2020 <= expiration_year <= 2030
                # Validate height
                height_unit = passport['hgt'][-2:]
                height_amt = int(passport['hgt'][:-2])
                assert (
                    height_unit == 'cm' and 150 <= height_amt <= 193 or
                    height_unit == 'in' and 59 <= height_amt <= 76
                    )
                # Validate hair color
                hair_color = passport['hcl']
                assert hair_color[0] == '#'
                assert all(
                    digit in '0123456789abcdef' for 
                    digit in hair_color[1:]
                    )
                # Validate eye color
                eye_color = passport['ecl']
                assert eye_color in (
                    'amb',
                    'blu',
                    'brn',
                    'gry',
                    'grn',
                    'hzl',
                    'oth',
                    )
                # Validate passport ID
                passport_id = passport['pid']
                assert len(passport_id) == 9
                assert all(
                    digit in '0123456789' for 
                    digit in passport_id
                    )
                # All validations passed
                valid_passport_count += 1
            except (AssertionError, KeyError, ValueError):
                continue
        result = valid_passport_count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        passports = self.get_passports(raw_input_lines)
        solutions = (
            self.solve(passports),
            self.solve2(passports),
            )
        result = solutions
        return result

class Day03: # Toboggan Trajectory
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
    
    def solve2(self, parsed_input: List[str]) -> int:
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

class Day02: # Password Philosophy
    '''
    Password Philosophy
    https://adventofcode.com/2020/day/2
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            a, b, c = raw_input_line.split(' ')
            num_a, num_b = map(int, a.split('-'))
            char = b[0]
            password = c
            result.append((num_a, num_b, char, password))
        return result
    
    def solve(self, parsed_input: List[str]) -> int:
        valid_password_count = 0
        for min_count, max_count, char, password in parsed_input:
            char_count = password.count(char)
            if min_count <= char_count <= max_count:
                valid_password_count += 1
        result = valid_password_count
        return result
    
    def solve2(self, parsed_input: List[str]) -> int:
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

class Day01: # Report Repair
    '''
    Report Repair
    https://adventofcode.com/2020/day/1
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(int(raw_input_line))
        return result
    
    def solve(self, parsed_input: List[str]) -> int:
        result = -1
        seen = set()
        for num in parsed_input:
            target = 2020 - num
            if target in seen:
                result = num * target
            else:
                seen.add(num)
        return result
    
    def solve2(self, parsed_input: List[str]) -> int:
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
    python Solver.py 18 < day18.in
    '''
    solvers = {
        1: (Day01, 'Report Repair'),
        2: (Day02, 'Password Philosophy'),
        3: (Day03, 'Toboggan Trajectory'),
        4: (Day04, 'Passport Processing'),
        5: (Day05, 'Binary Boarding'),
        6: (Day06, 'Custom Customs'),
        7: (Day07, 'Handy Haversacks'),
        8: (Day08, 'Handheld Halting'),
        9: (Day09, 'Encoding Error'),
       10: (Day10, 'Adapter Array'),
       11: (Day11, 'Seating System'),
       12: (Day12, 'Rain Risk'),
       13: (Day13, 'Shuttle Search'),
       14: (Day14, 'Docking Data'),
       15: (Day15, 'Rambunctious Recitation'),
       16: (Day16, 'Ticket Translation'),
       17: (Day17, 'Conway Cubes'),
       18: (Day18, 'Operation Order'),
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
