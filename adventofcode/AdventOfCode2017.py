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

class KnotHash:
    def __init__(self, input_str: str):
        self.hash = self.get_hash(input_str)
    
    def get_hash(self, input_str: str) -> str:
        lengths = []
        for char in input_str:
            lengths.append(ord(char))
        lengths += [17, 31, 73, 47, 23]
        nums = list(range(256))
        cursor = 0
        skip_size = 0
        for _ in range(64):
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
            char = hex(xors[i])[2:]
            if len(char) == 1:
                char = '0' + char
            chars.append(char)
        result = ''.join(chars)
        return result

class DuetVM:
    def __init__(self, instructions):
        self.registers = {}
        for char in 'abcdefghijklmnopqrstuvwxyz':
            self.registers[char] = 0
        self.instructions = instructions
        self.pc = 0
        self.sends = collections.deque()
        self.receiving_register = None
    
    def get_value(self, operand) -> int:
        value = operand
        if type(operand) == str:
            value = self.registers[operand]
        return value
    
    def run(self):
        while (
            self.pc < len(self.instructions) and
            self.receiving_register is None
        ):
            self.step()
    
    def receive(self, value):
        self.registers[self.receiving_register] = value
        self.receiving_register = None
    
    def step(self):
        operation, operands = self.instructions[self.pc]
        if operation == 'snd':
            x = operands[0]
            send = self.get_value(x)
            self.sends.append(send)
            self.pc += 1
        elif operation == 'set':
            x = operands[0]
            y = operands[1]
            self.registers[x] = self.get_value(y)
            self.pc += 1
        elif operation == 'add':
            x = operands[0]
            y = operands[1]
            self.registers[x] += self.get_value(y)
            self.pc += 1
        elif operation == 'mul':
            x = operands[0]
            y = operands[1]
            self.registers[x] *= self.get_value(y)
            self.pc += 1
        elif operation == 'mod':
            x = operands[0]
            y = operands[1]
            self.registers[x] %= self.get_value(y)
            self.pc += 1
        elif operation == 'rcv':
            x = operands[0]
            self.receiving_register = x
            self.pc += 1
        elif operation == 'jgz':
            x = operands[0]
            y = operands[1]
            if self.get_value(x) > 0:
                self.pc += self.get_value(y)
            else:
                self.pc += 1

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

class Day20: # Particle Swarm
    '''
    https://adventofcode.com/2017/day/20
    '''
    def get_particles(self, raw_input_lines: List[str]):
        particles = []
        for raw_input_line in raw_input_lines:
            particle = {}
            parts = raw_input_line.split(', ')
            for part in parts:
                key = part[0]
                raw_x, raw_y, raw_z = part[3:-1].split(',')
                x = int(raw_x)
                y = int(raw_y)
                z = int(raw_z)
                particle[key] = [x, y, z]
            particles.append(particle)
        result = particles
        return result
    
    def solve(self, particles):
        result = len(particles)
        return result
    
    def solve2(self, particles):
        result = len(particles)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        particles = self.get_particles(raw_input_lines)
        solutions = (
            self.solve(particles),
            self.solve2(particles),
            )
        result = solutions
        return result

class Day19: # A Series of Tubes
    '''
    https://adventofcode.com/2017/day/19
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def traverse(self, parsed_input):
        path = []
        step_count = 0
        rows = len(parsed_input)
        cols = len(parsed_input[0])
        col = 0
        for i in range(cols):
            if parsed_input[0][i] == '|':
                col = i
                break
        row = 0
        direction = 'SOUTH'
        path = []
        while True:
            cell = parsed_input[row][col]
            if cell == ' ':
                break
            elif cell in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                path.append(cell)
            elif cell == '+':
                if direction in ('NORTH', 'SOUTH'):
                    if col < 1 or parsed_input[row][col + 1] != ' ':
                        direction = 'EAST'
                    else:
                        direction = 'WEST'
                elif direction in ('EAST', 'WEST'):
                    if row < 1 or parsed_input[row + 1][col] != ' ':
                        direction = 'SOUTH'
                    else:
                        direction = 'NORTH'
            if direction == 'NORTH':
                row -= 1
            elif direction == 'EAST':
                col += 1
            elif direction == 'SOUTH':
                row += 1
            elif direction == 'WEST':
                col -= 1
            step_count += 1
        result = (step_count, ''.join(path))
        return result
    
    def solve(self, parsed_input):
        _, result = self.traverse(parsed_input)
        return result
    
    def solve2(self, parsed_input):
        result, _ = self.traverse(parsed_input)
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

class Day18: # Duet
    '''
    https://adventofcode.com/2017/day/18
    '''
    def get_instructions(self, raw_input_lines: List[str]):
        instructions = []
        for raw_input_line in raw_input_lines:
            parts = raw_input_line.split(' ')
            operation = parts[0]
            operands = parts[1:]
            for i in range(len(operands)):
                try:
                    operands[i] = int(operands[i])
                except ValueError:
                    pass
            instruction = (operation, operands)
            instructions.append(instruction)
        result = instructions
        return result
    
    def solve(self, instructions):
        vm = DuetVM(instructions)
        vm.run()
        result = vm.sends[-1]
        return result
    
    def solve2(self, instructions):
        vm0 = DuetVM(instructions)
        vm0.registers['p'] = 0
        vm1 = DuetVM(instructions)
        vm1.registers['p'] = 1
        vm1_send_step_count = 0
        while (
            vm0.receiving_register is None or
            vm1.receiving_register is None
        ):
            vm0.run()
            vm1.run()
            if (
                vm0.receiving_register is not None and
                len(vm1.sends) > 0
            ):
                vm0.receive(vm1.sends.popleft())
                vm1_send_step_count += 1
            if (
                vm1.receiving_register is not None and
                len(vm0.sends) > 0
            ):
                vm1.receive(vm0.sends.popleft())
        result = vm1_send_step_count + len(vm1.sends)
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

class Day17: # Spinlock
    '''
    https://adventofcode.com/2017/day/17
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = int(raw_input_lines[0])
        return result
    
    def solve(self, step_count, iteration_count, target):
        buffer = [0]
        cursor = 0
        for num in range(1, iteration_count + 1):
            cursor = (cursor + step_count) % len(buffer)
            buffer = buffer[:cursor + 1] + [num] + buffer[cursor + 1:]
            cursor += 1
        result = buffer[0]
        index = buffer.index(target)
        if index < len(buffer) - 1:
            result = buffer[index + 1]
        return result
    
    def solve2(self, step_count, iteration_count):
        '''
        [0]             0: 0 -> (0 + 3) % 1 = 0 + 1 = 1
        [0 1]           1: 1 -> (1 + 3) % 2 = 0 + 1 = 1
        [0 2 1]         2: 1 -> (1 + 3) % 3 = 1 + 1 = 2
        [0 2 3 1]       3: 2 -> (2 + 3) % 4 = 1 + 1 = 2
        [0 2 4 3 1]     4: 2 -> (2 + 3) % 5 = 0 + 1 = 1
        [0 5 2 4 3 1]   5: 1 -> (1 + 3) % 6 = 4 + 1 = 5
        [0 5 2 4 3 6 1] 6: 5 -> (5 + 3) % 7 = 1 + 1 = 2
        '''
        value_after_zero = 1
        spinlock_size = 2
        cursor = 1
        for num in range(2, iteration_count):
            cursor = (cursor + step_count) % spinlock_size + 1
            if cursor == 1:
                value_after_zero = num
            spinlock_size += 1
        result = value_after_zero
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        parsed_input = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(parsed_input, 2017, 2017),
            self.solve2(parsed_input, 50_000_000),
            )
        result = solutions
        return result

class Day16: # Permutation Promenade
    '''
    https://adventofcode.com/2017/day/16
    TODO: Consider if cycle detection will make this any faster
    '''
    def get_dance_moves(self, raw_input_lines: List[str]):
        dance_moves = []
        for line in raw_input_lines[0].split(','):
            parameters = line[1:].split('/')
            dance_move = [line[0]]
            for parameter in parameters:
                try:
                    value = int(parameter)
                    dance_move.append(value)
                except ValueError:
                    dance_move.append(parameter)
            dance_moves.append(tuple(dance_move))
        result = dance_moves
        return result
    
    def solve(self, state, dance_moves):
        programs = list(state)
        for dance_move in dance_moves:
            if dance_move[0] == 's':
                X = dance_move[1]
                programs = programs[-X:] + programs[:-X]
            elif dance_move[0] == 'x':
                A = dance_move[1]
                B = dance_move[2]
                programs[A], programs[B] = programs[B], programs[A]
            elif dance_move[0] == 'p':
                A = programs.index(dance_move[1])
                B = programs.index(dance_move[2])
                programs[A], programs[B] = programs[B], programs[A]
        result = ''.join(programs)
        return result
    
    def memoized(self, dance_moves, iteration_count):
        @functools.lru_cache(maxsize=1024)
        def perform_dance(current_state: str) -> str:
            next_state = self.solve(current_state, dance_moves)
            return next_state
        state = 'abcdefghijklmnop'
        for i in range(iteration_count):
            if i % 10_000_000 == 0:
                progress = int(100 * i / iteration_count)
                print(f'{progress}% done')
            next_state = perform_dance(state)
            state = next_state
        result = state
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        dance_moves = self.get_dance_moves(raw_input_lines)
        solutions = (
            self.solve('abcdefghijklmnop', dance_moves),
            self.memoized(dance_moves, 1_000_000_000),
            )
        result = solutions
        return result


class Day15: # Dueling Generators
    '''
    https://adventofcode.com/2017/day/15
    '''
    class Generator:
        MODULO = 2147483647

        def __init__(self, value: int, factor: int, multiple: int=0):
            self.value = value
            self.factor = factor
            self.multiple = multiple
        
        def next(self) -> int:
            while True:
                self.value = (self.value * self.factor) % self.MODULO
                if self.multiple == 0 or self.value % self.multiple == 0:
                    break
            result = self.value
            return result
    
    MASK = 2 ** 16 - 1

    def get_starts(self, raw_input_lines: List[str]):
        starts = {}
        for raw_input_line in raw_input_lines:
            parts = raw_input_line.split()
            key = parts[1]
            start = int(parts[-1])
            starts[key] = start
        result = starts
        return result
    
    def solve_slowly(self, starts):
        generators = {
            self.Generator(starts['A'], 16807),
            self.Generator(starts['B'], 48271),
        }
        match_count = 0
        for i in range(40_000_000):
            for gen in generators:
                gen.next()
            bits = None
            for gen in generators:
                if bits is None:
                    bits = gen.value & self.MASK
                if bits != (gen.value & self.MASK):
                    break
            else:
                match_count += 1
        result = match_count
        return result
    
    def solve2_slowly(self, starts):
        generators = {
            self.Generator(starts['A'], 16807, 4),
            self.Generator(starts['B'], 48271, 8),
        }
        match_count = 0
        for i in range(5_000_000):
            for gen in generators:
                gen.next()
            bits = None
            for gen in generators:
                if bits is None:
                    bits = gen.value & self.MASK
                if bits != (gen.value & self.MASK):
                    break
            else:
                match_count += 1
        result = match_count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        starts = self.get_starts(raw_input_lines)
        solutions = (
            self.solve_slowly(starts),
            self.solve2_slowly(starts),
            )
        result = solutions
        return result

class Day14: # Disk Defragmentation
    '''
    https://adventofcode.com/2017/day/14
    '''
    def get_grid(self, raw_input_lines: List[str]):
        parsed_input = raw_input_lines[0]
        grid = []
        for index in range(128):
            input_str = parsed_input + '-' + str(index)
            hash = KnotHash(input_str).hash
            bin_hash = str(bin(int(hash, 16))[2:].zfill(128))
            grid.append(bin_hash)
        result = grid
        return result
    
    def solve(self, grid):
        result = sum(1 for char in ''.join(grid) if char == '1')
        return result
    
    def solve2(self, grid):
        cells = set()
        for row, row_data in enumerate(grid):
            for col, cell in enumerate(row_data):
                if cell == '1':
                    cells.add((row, col))
        region_count = 0
        while len(cells) > 0:
            work = [cells.pop()]
            while len(work) > 0:
                row, col = work.pop()
                for (nrow, ncol) in (
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                ):
                    if (nrow, ncol) in cells:
                        cells.remove((nrow, ncol))
                        work.append((nrow, ncol))
            region_count += 1
        result = region_count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        grid = self.get_grid(raw_input_lines)
        solutions = (
            self.solve(grid),
            self.solve2(grid),
            )
        result = solutions
        return result

class Day13: # Packet Scanners
    '''
    https://adventofcode.com/2017/day/13
    '''
    def get_layers(self, raw_input_lines: List[str]):
        layers = {}
        for raw_input_line in raw_input_lines:
            a, b = raw_input_line.split(': ')
            layers[int(a)] = int(b)
        result = layers
        return result
    
    def solve(self, layers):
        severity = 0
        for depth, range in layers.items():
            if depth % (2 * (range - 1)) == 0:
                severity += depth * range
        result = severity
        return result
    
    def solve2(self, layers):
        min_delay = float('inf')
        delay = 0
        while True:
            detected_ind = False
            for depth, range in layers.items():
                if (depth + delay) % (2 * (range - 1)) == 0:
                    detected_ind = True
                    break
            if not detected_ind:
                min_delay = delay
                break
            delay += 1
        result = min_delay
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        layers = self.get_layers(raw_input_lines)
        solutions = (
            self.solve(layers),
            self.solve2(layers),
            )
        result = solutions
        return result

class Day12: # Digital Plumber
    '''
    https://adventofcode.com/2017/day/12
    '''
    def get_graph(self, raw_input_lines: List[str]):
        graph = {}
        for raw_input_line in raw_input_lines:
            a, b = raw_input_line.split(' <-> ')
            node = int(a)
            connections = set(map(int, b.split(', ')))
            connections -= { node }
            graph[node] = connections
        result = graph
        return result
    
    def solve(self, graph):
        nodes_visited = set()
        work = [0]
        while len(work) > 0:
            node = work.pop()
            nodes_visited.add(node)
            for next_node in graph[node]:
                if next_node not in nodes_visited:
                    work.append(next_node)
        nodes_visited.add(0)
        result = len(nodes_visited)
        return result
    
    def solve2(self, graph):
        group_count = 0
        nodes_left = set(graph.keys())
        while len(nodes_left) > 0:
            work = [nodes_left.pop()]
            while len(work) > 0:
                node = work.pop()
                for next_node in graph[node]:
                    if next_node in nodes_left:
                        work.append(next_node)
                        nodes_left.remove(next_node)
            group_count += 1
        result = group_count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        graph = self.get_graph(raw_input_lines)
        solutions = (
            self.solve(graph),
            self.solve2(graph),
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
        result = KnotHash(chars).hash
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
       12: (Day12, 'Digital Plumber'),
       13: (Day13, 'Packet Scanners'),
       14: (Day14, 'Disk Defragmentation'),
       15: (Day15, 'Dueling Generators'),
       16: (Day16, 'Permutation Promenade'),
       17: (Day17, 'Spinlock'),
       18: (Day18, 'Duet'),
       19: (Day19, 'A Series of Tubes'),
       20: (Day20, 'Particle Swarm'),
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
