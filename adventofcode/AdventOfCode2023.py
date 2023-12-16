'''
Created on 2023-11-28

@author: Sestren
'''
import argparse
import collections
import copy
import functools
import heapq
import itertools
import random
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
    https://adventofcode.com/2023/day/?
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

class Day15: # Lens Library
    '''
    https://adventofcode.com/2023/day/15
    '''
    def get_sequences(self, raw_input_lines: List[str]):
        sequences = list(raw_input_lines[0].split(','))
        result = sequences
        return result
    
    def solve(self, sequences):
        hashes = []
        for sequence in sequences:
            hash = 0
            for char in sequence:
                hash += ord(char)
                hash *= 17
                hash %= 256
            hashes.append(hash)
        result = sum(hashes)
        return result
    
    def solve2(self, sequences):
        result = len(sequences)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        sequences = self.get_sequences(raw_input_lines)
        solutions = (
            self.solve(sequences),
            self.solve2(sequences),
            )
        result = solutions
        return result

class Day14: # Parabolic Reflector Dish
    '''
    https://adventofcode.com/2023/day/14
    '''
    def get_rocks(self, raw_input_lines: List[str]):
        rows = len(raw_input_lines)
        cols = len(raw_input_lines[0])
        rocks = {}
        for row, raw_input_line in enumerate(raw_input_lines):
            for col, char in enumerate(raw_input_line):
                if char in 'O#':
                    rocks[(row, col)] = char
        result = (rocks, rows, cols)
        return result
    
    def solve(self, rocks, rows, cols):
        # Tilt north
        while True:
            moves = set(
                (row, col) for (row, col), rock in rocks.items() if
                rock == 'O' and row > 0 and (row - 1, col) not in rocks
            )
            for (row, col) in moves:
                del rocks[(row, col)]
                rocks[(row - 1, col)] = 'O'
            if len(moves) == 0:
                break
        # Calculate total load
        total_load = sum(
            rows - row for (row, col), rock in rocks.items() if
            rock == 'O'
        )
        result = total_load
        return result

    def cycle(self, rocks, rows, cols):
        # Tilt north
        while True:
            moves = set(
                (row, col) for (row, col), rock in rocks.items() if
                rock == 'O' and row > 0 and (row - 1, col) not in rocks
            )
            for (row, col) in moves:
                del rocks[(row, col)]
                rocks[(row - 1, col)] = 'O'
            if len(moves) == 0:
                break
        # Tilt west
        while True:
            moves = set(
                (row, col) for (row, col), rock in rocks.items() if
                rock == 'O' and col > 0 and (row, col - 1) not in rocks
            )
            for (row, col) in moves:
                del rocks[(row, col)]
                rocks[(row, col - 1)] = 'O'
            if len(moves) == 0:
                break
        # Tilt south
        while True:
            moves = set(
                (row, col) for (row, col), rock in rocks.items() if
                rock == 'O' and (row + 1) < rows and (row + 1, col) not in rocks
            )
            for (row, col) in moves:
                del rocks[(row, col)]
                rocks[(row + 1, col)] = 'O'
            if len(moves) == 0:
                break
        # Tilt east
        while True:
            moves = set(
                (row, col) for (row, col), rock in rocks.items() if
                rock == 'O' and (col + 1) < cols and (row, col + 1) not in rocks
            )
            for (row, col) in moves:
                del rocks[(row, col)]
                rocks[(row, col + 1)] = 'O'
            if len(moves) == 0:
                break
    
    def get_total_load(self, rocks, rows):
        total_load = sum(
            rows - row for (row, col), rock in rocks.items() if
            rock == 'O'
        )
        result = total_load
        return result
    
    def solve2(self, rocks, rows, cols):
        meta_cycle_key_length = 100
        cycles_needed = 1_000_000_000
        total_cycles = 0
        # Get first meta cycle key
        recent_loads = collections.deque()
        for _ in range(meta_cycle_key_length):
            self.cycle(rocks, rows, cols)
            total_load = self.get_total_load(rocks, rows)
            total_cycles += 1
            recent_loads.append(total_load)
            if len(recent_loads) > meta_cycle_key_length:
                recent_loads.popleft()
        # Compute meta cycle size
        meta_cycle_key = tuple(recent_loads)
        cycles = {}
        while meta_cycle_key not in cycles:
            cycles[meta_cycle_key] = total_cycles
            self.cycle(rocks, rows, cols)
            total_load = self.get_total_load(rocks, rows)
            total_cycles += 1
            if len(recent_loads) > meta_cycle_key_length:
                recent_loads.popleft()
            meta_cycle_key = tuple(recent_loads)
            print(total_cycles, 'meta_cycle', meta_cycle_key)
        meta_cycle_size = total_cycles - cycles[meta_cycle_key]
        # Skip several meta cycles
        while total_cycles + 10000 * meta_cycle_size <= cycles_needed:
            total_cycles += 10000 * meta_cycle_size
            print(total_cycles, 'load10000', total_load)
        while total_cycles + 1000 * meta_cycle_size <= cycles_needed:
            total_cycles += 1000 * meta_cycle_size
            print(total_cycles, 'load1000 ', total_load)
        while total_cycles + 100 * meta_cycle_size <= cycles_needed:
            total_cycles += 100 * meta_cycle_size
            print(total_cycles, 'load100  ', total_load)
        while total_cycles + 10 * meta_cycle_size <= cycles_needed:
            total_cycles += 10 * meta_cycle_size
            print(total_cycles, 'load10   ', total_load)
        while total_cycles + meta_cycle_size <= cycles_needed:
            total_cycles += meta_cycle_size
            print(total_cycles, 'load1    ', total_load)
        # Finish the rest of the cycles
        while total_cycles < cycles_needed:
            self.cycle(rocks, rows, cols)
            total_load = self.get_total_load(rocks, rows)
            total_cycles += 1
        result = total_load
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        (rocks, rows, cols) = self.get_rocks(raw_input_lines)
        solutions = (
            self.solve(copy.deepcopy(rocks), rows, cols),
            self.solve2(copy.deepcopy(rocks), rows, cols),
            )
        result = solutions
        return result

class Day13: # Point of Incidence
    '''
    https://adventofcode.com/2023/day/13
    '''
    def get_patterns(self, raw_input_lines: List[str]):
        patterns = []
        pattern = set()
        row = 0
        cols = len(raw_input_lines[0])
        for raw_input_line in raw_input_lines:
            if len(raw_input_line) < 1:
                patterns.append((pattern, row, cols))
                pattern = set()
                row = 0
            else:
                for col, char in enumerate(raw_input_line):
                    if char == '#':
                        pattern.add((row, col))
                cols = len(raw_input_line)
                row += 1
        patterns.append((pattern, row, cols))
        result = patterns
        return result
    
    def find_reflections(self, pattern, rows, cols):
        reflections = set()
        # Find reflected row
        for line in range(1, rows):
            height = min(line, rows - line)
            for i in range(height):
                top = line - i - 1
                bottom = line + i
                top_rocks = set(
                    col for (row, col) in
                    pattern if row == top
                )
                bottom_rocks = set(
                    col for (row, col) in
                    pattern if row == bottom
                )
                if top_rocks != bottom_rocks:
                    break
            else:
                reflections.add(('Row', line))
        # Find reflected column
        for line in range(1, cols):
            width = min(line, cols - line)
            for i in range(width):
                left = line - i - 1
                right = line + i
                left_rocks = set(
                    row for (row, col) in
                    pattern if col == left
                )
                right_rocks = set(
                    row for (row, col) in
                    pattern if col == right
                )
                if left_rocks != right_rocks:
                    break
            else:
                reflections.add(('Column', line))
        result = reflections
        return result
    
    def solve(self, patterns):
        reflected_rows = []
        reflected_cols = []
        for (pattern, rows, cols) in patterns:
            reflections = self.find_reflections(pattern, rows, cols)
            (reflection_type, reflection_line) = reflections.pop()
            if reflection_type == 'Row':
                reflected_rows.append(reflection_line)
            elif reflection_type == 'Column':
                reflected_cols.append(reflection_line)
        result = 100 * sum(reflected_rows) + sum(reflected_cols)
        return result
    
    def solve2(self, patterns):
        reflected_rows = []
        reflected_cols = []
        for (pattern, rows, cols) in patterns:
            # Try all possible smudge locations
            new_reflection_found = False
            old_reflections = self.find_reflections(pattern, rows, cols)
            for row in range(rows):
                if new_reflection_found:
                    break
                for col in range(cols):
                    smudged_pattern = set(pattern)
                    if (row, col) in smudged_pattern:
                        smudged_pattern.remove((row, col))
                    else:
                        smudged_pattern.add((row, col))
                    assert pattern != smudged_pattern
                    reflections = self.find_reflections(smudged_pattern, rows, cols)
                    new_reflections = reflections - old_reflections
                    if len(new_reflections) > 0:
                        assert len(new_reflections) == 1
                        (new_reflection_type, new_reflection_line) = new_reflections.pop()
                        if new_reflection_type == 'Row':
                            new_reflection_found = True
                            reflected_rows.append(new_reflection_line)
                            break
                        elif new_reflection_type == 'Column':
                            new_reflection_found = True
                            reflected_cols.append(new_reflection_line)
                            break
        result = 100 * sum(reflected_rows) + sum(reflected_cols)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        patterns = self.get_patterns(raw_input_lines)
        solutions = (
            self.solve(patterns),
            self.solve2(patterns),
            )
        result = solutions
        return result

class Day12: # Hot Springs
    '''
    https://adventofcode.com/2023/day/12
    '''
    def get_records(self, raw_input_lines: List[str]):
        records = []
        for raw_input_line in raw_input_lines:
            condition, b = raw_input_line.split(' ')
            summary = tuple(map(int, b.split(',')))
            records.append((condition, summary))
        result = records
        return result
    
    def possible_arrangements(self, condition):
        wildcards = []
        for index, char in enumerate(condition):
            if char == '?':
                wildcards.append(index)
        if len(wildcards) < 1:
            return [condition]
        arrangements = []
        arrangement_count = 2 ** len(wildcards)
        for code_id in range(arrangement_count):
            arrangement = list(condition)
            code = bin(code_id)[2:][::-1]
            for i in range(len(wildcards)):
                char = '.'
                if i < len(code) and code[i] == '1':
                    char = '#'
                arrangement[wildcards[i]] = char
            arrangements.append(''.join(arrangement))
        result = arrangements
        return result

    def summarize(self, arrangement):
        start = 0
        while start < len(arrangement) and arrangement[start] == '.':
            start += 1
        summary = [0]
        for i in range(start, len(arrangement)):
            if arrangement[i] == '.':
                if summary[-1] > 0:
                    summary.append(0)
            else:
                summary[-1] += 1
        if len(summary) > 1 and summary[-1] == 0:
            summary.pop()
        result = tuple(summary)
        return result
    
    def solve(self, records):
        valid_arrangement_counts = []
        for (condition, summary) in records:
            valid_arrangement_count = 0
            for arrangement in self.possible_arrangements(condition):
                new_summary = self.summarize(arrangement)
                if new_summary == summary:
                    valid_arrangement_count += 1
            valid_arrangement_counts.append(valid_arrangement_count)
            print(valid_arrangement_count, condition, summary)
        result = sum(valid_arrangement_counts)
        return result
    
    def solve2(self, records):
        result = len(records)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        records = self.get_records(raw_input_lines)
        solutions = (
            self.solve(records),
            self.solve2(records),
            )
        result = solutions
        return result

class Day11: # Cosmic Expansion
    '''
    https://adventofcode.com/2023/day/11
    '''
    def get_galaxies(self, raw_input_lines: List[str]):
        galaxies = set()
        for row, raw_input_line in enumerate(raw_input_lines):
            for col, char in enumerate(raw_input_line):
                if char == '#':
                    galaxies.add((row, col))
        result = galaxies
        return result
    
    def solve(self, galaxies, expansions=2):
        filled_rows = set()
        filled_cols = set()
        for (row, col) in galaxies:
            filled_rows.add(row)
            filled_cols.add(col)
        min_row = min(filled_rows)
        max_row = max(filled_rows)
        empty_rows = set(range(min_row, max_row + 1)) - filled_rows
        min_col = min(filled_cols)
        max_col = max(filled_cols)
        empty_cols = set(range(min_col, max_col + 1)) - filled_cols
        expanded_galaxies = set()
        for (row, col) in galaxies:
            expanded_row = row + sum(
                expansions - 1 for _ in (
                    empty_row for empty_row in
                    empty_rows if empty_row < row
                )
            )
            expanded_col = col + sum(
                expansions - 1 for _ in (
                    empty_col for empty_col in
                    empty_cols if empty_col < col
                )
            )
            expanded_galaxies.add((expanded_row, expanded_col))
        distances = {}
        for (row1, col1) in expanded_galaxies:
            for (row2, col2) in expanded_galaxies:
                if (row2 == row1 and col2 == col1):
                    continue
                pairing_key = (
                    min((row1, col1), (row2, col2)),
                    max((row1, col1), (row2, col2)),
                )
                distance = abs(row2 - row1) + abs(col2 - col1)
                distances[pairing_key] = distance
        result = sum(distances.values())
        return result
    
    def solve2(self, galaxies):
        result = self.solve(galaxies, 1_000_000)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        galaxies = self.get_galaxies(raw_input_lines)
        solutions = (
            self.solve(galaxies),
            self.solve2(galaxies),
            )
        result = solutions
        return result

class Day10: # Pipe Maze
    '''
    https://adventofcode.com/2023/day/10
    '''
    ROW, COL = 0, 1
    DIRECTIONS = {
        'N': (-1,  0),
        'E': ( 0,  1),
        'S': ( 1,  0),
        'W': ( 0, -1),
    }
    CONNECTIONS = {
        '|': {'N', 'S'},
        '-': {'E', 'W'},
        'L': {'N', 'E'},
        'J': {'N', 'W'},
        '7': {'S', 'W'},
        'F': {'S', 'E'},
    }

    def get_parsed_input(self, raw_input_lines: List[str]):
        start = None
        pipes = {}
        for (row, raw_input_line) in enumerate(raw_input_lines):
            for (col, char) in enumerate(raw_input_line):
                if char == 'S':
                    start = (row, col)
                    directions = set()
                    # Figure out what type of pipe S is based on its neighbors
                    if row > 0:
                        north = raw_input_lines[row - 1][col]
                        if 'S' in self.CONNECTIONS[north]:
                            directions.add('N')
                    if row < len(raw_input_lines) - 1:
                        south = raw_input_lines[row + 1][col]
                        if 'N' in self.CONNECTIONS[south]:
                            directions.add('S')
                    if col > 0:
                        west = raw_input_lines[row][col - 1]
                        if 'E' in self.CONNECTIONS[west]:
                            directions.add('W')
                    if col < len(raw_input_line) - 1:
                        east = raw_input_lines[row][col + 1]
                        if 'W' in self.CONNECTIONS[east]:
                            directions.add('E')
                    for new_char, new_directions in self.CONNECTIONS.items():
                        if len(new_directions & directions) == 2:
                            pipes[(row, col)] = new_char
                            break
                    else:
                        raise Exception('No replacement for pipe found')
                else:
                    pipes[(row, col)] = char
        result = (start, pipes)
        return result
    
    def get_main_loop(self, start, pipes):
        (row, col) = start
        main_loop = set()
        work = [(row, col)]
        while len(work) > 0:
            (row, col) = work.pop()
            main_loop.add((row, col))
            connection = pipes[(row, col)]
            for direction in self.CONNECTIONS[connection]:
                next_row = row + self.DIRECTIONS[direction][self.ROW]
                next_col = col + self.DIRECTIONS[direction][self.COL]
                if (next_row, next_col) not in main_loop:
                    work.append((next_row, next_col))
        result = main_loop
        return result

    def get_boundaries(self, pipes):
        min_row = min(row for (row, col) in pipes)
        max_row = max(row for (row, col) in pipes)
        min_col = min(col for (row, col) in pipes)
        max_col = max(col for (row, col) in pipes)
        result = (min_row, max_row, min_col, max_col)
        return result

    def simplify_pipes(self, pipes, main_loop):
        (min_row, max_row, min_col, max_col) = self.get_boundaries(main_loop)
        simplified_pipes = {}
        for (row, col), char in pipes.items():
            if (
                row < min_row or
                row > max_row or
                col < min_col or
                col > max_col
            ):
                continue
            if (row, col) not in main_loop:
                char = '.'
            simplified_pipes[(row, col)] = char
        result = simplified_pipes
        return result

    def expand_pipes(self, pipes):
        expanded_pipes = {}
        for (row, col), char in pipes.items():
            expanded_pipes[(2 * row, 2 * col)] = char
            if char not in self.CONNECTIONS:
                continue
            for direction in self.CONNECTIONS[char]:
                next_row = 2 * row + self.DIRECTIONS[direction][self.ROW]
                next_col = 2 * col + self.DIRECTIONS[direction][self.COL]
                expanded_pipes[(next_row, next_col)] = '+'
        result = expanded_pipes
        return result
    
    def solve(self, start, pipes):
        (row, col) = start
        main_loop = self.get_main_loop(start, pipes)
        result = len(main_loop) // 2
        return result
    
    def solve2(self, start, pipes):
        main_loop = self.get_main_loop(start, pipes)
        simplified_pipes = self.simplify_pipes(pipes, main_loop)
        expanded_pipes = self.expand_pipes(simplified_pipes)
        (min_row, max_row, min_col, max_col) = self.get_boundaries(expanded_pipes)
        enclosed_tiles = set()
        for ((row, col), char) in expanded_pipes.items():
            if char != '.':
                continue
            seen = set()
            visited = set()
            enclosed_ind = True
            work = collections.deque()
            work.append((row, col))
            seen.add((row, col))
            while len(work) > 0:
                (curr_row, curr_col) = work.pop()
                if (
                    curr_row < min_row or
                    curr_row > max_row or
                    curr_col < min_col or
                    curr_col > max_col
                ):
                    enclosed_ind = False
                    break
                visited.add((curr_row, curr_col))
                for (next_row, next_col) in (
                    (curr_row - 1, curr_col   ),
                    (curr_row + 1, curr_col   ),
                    (curr_row    , curr_col - 1),
                    (curr_row    , curr_col + 1),
                ):
                    if (next_row, next_col) in seen:
                        continue
                    if (
                        (next_row, next_col) in expanded_pipes and
                        expanded_pipes[(next_row, next_col)] != '.'
                    ):
                        continue
                    work.appendleft((next_row, next_col))
                    seen.add((next_row, next_col))
            if enclosed_ind:
                enclosed_tiles |= visited
                break
        result = len(
            enclosed_tiles &
            set((row, col) for
                (row, col) in expanded_pipes if
                expanded_pipes[(row, col)] == '.'
            )
        )
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        start, pipes = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(start, pipes),
            self.solve2(start, pipes),
            )
        result = solutions
        return result

class Day09: # Mirage Maintenance
    '''
    https://adventofcode.com/2023/day/9
    '''
    def get_histories(self, raw_input_lines: List[str]):
        histories = []
        for raw_input_line in raw_input_lines:
            history = tuple(map(int, raw_input_line.split()))
            histories.append(history)
        result = histories
        return result

    def get_differences(self, history):
        differences = [list(history)]
        while len(set(differences[-1])) > 1 or differences[-1][0] != 0:
            next_difference = []
            for i in range(len(differences[-1]) - 1):
                a = differences[-1][i]
                b = differences[-1][i + 1]
                next_difference.append(b - a)
            differences.append(next_difference)
        result = differences
        return result

    def get_extrapolation(self, history):
        differences = self.get_differences(history)
        extrapolation = 0
        for i in range(len(differences)):
            extrapolation += differences[i][-1]
        result = extrapolation
        return result

    def get_reverse_extrapolation(self, history):
        differences = self.get_differences(history)
        extrapolation = 0
        for i in reversed(range(len(differences) - 1)):
            next_extrapolation = differences[i][0] - extrapolation
            extrapolation = next_extrapolation
        result = extrapolation
        return result
    
    def solve(self, histories):
        extrapolations = list()
        for history in histories:
            extrapolations.append(self.get_extrapolation(history))
        result = sum(extrapolations)
        return result
    
    def solve2(self, histories):
        extrapolations = list()
        for history in histories:
            extrapolations.append(self.get_reverse_extrapolation(history))
        result = sum(extrapolations)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        histories = self.get_histories(raw_input_lines)
        solutions = (
            self.solve(histories),
            self.solve2(histories),
            )
        result = solutions
        return result

class Day08: # Haunted Wasteland
    '''
    https://adventofcode.com/2023/day/8
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        instructions = raw_input_lines[0]
        network = {}
        for raw_input_line in raw_input_lines[2:]:
            (node, raw_elements) = raw_input_line.split(' = ')
            (left_node, right_node) = raw_elements[1:-1].split(', ')
            network[node] = (left_node, right_node)
        result = (instructions, network)
        return result
    
    def solve(self, instructions, network, starting_node='AAA', ending_pattern='ZZZ'):
        LEFT, RIGHT = 0, 1
        step_count = 0
        node = starting_node
        while True:
            instruction = instructions[step_count % len(instructions)]
            next_node = node
            if instruction == 'L':
                next_node = network[node][LEFT]
            else:
                next_node = network[node][RIGHT]
            step_count += 1
            node = next_node
            ending_pattern_matched = True
            for i in range(len(node)):
                if ending_pattern[i] == '*' or ending_pattern[i] == node[i]:
                    pass
                else:
                    ending_pattern_matched = False
                    break
            if ending_pattern_matched:
                break
        result = step_count
        return result
    
    def solve2(self, instructions, network):
        LEFT, RIGHT = 0, 1
        step_counts = []
        for node in network.keys():
            if node[-1] == 'A':
                step_counts.append(
                    self.solve(instructions, network, node, '**Z')
                )
        total_step_count = 1
        for step_count in sorted(step_counts):
            multiplier = 1
            while ((multiplier * total_step_count) % step_count) != 0:
                multiplier += 1
            total_step_count *= multiplier
        result = total_step_count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        (instructions, network) = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(instructions, network),
            self.solve2(instructions, network),
            )
        result = solutions
        return result

class Day07: # Camel Cards
    '''
    https://adventofcode.com/2023/day/7
    '''
    joker_rule = False
    rank_values = {
        'A': 14,
        'K': 13,
        'Q': 12,
        'J': 11,
        'T': 10,
        '9': 9,
        '8': 8,
        '7': 7,
        '6': 6,
        '5': 5,
        '4': 4,
        '3': 3,
        '2': 2,
    }
    hand_type_values = {
        'Five of a kind': 6,
        'Four of a kind': 5,
        'Full house': 4,
        'Three of a kind': 3,
        'Two pair': 2,
        'One pair': 1,
        'High card': 0,
    }
    def get_bidding_hands(self, raw_input_lines: List[str]):
        bidding_hands = []
        for raw_input_line in raw_input_lines:
            (hand, raw_bid) = raw_input_line.split()
            bid = int(raw_bid)
            bidding_hands.append((hand, bid))
        result = bidding_hands
        return result

    def get_hand_type(self, hand):
        counts = collections.Counter(hand)
        RANK, COUNT = 0, 1
        ranks = []
        for rank, count in counts.most_common():
            ranks.append((rank, count))
        hand_type = 'UNKNOWN'
        if ranks[0][COUNT] == 5:
            hand_type = 'Five of a kind'
        elif ranks[0][COUNT] == 4:
            hand_type = 'Four of a kind'
        elif ranks[0][COUNT] == 3:
            if len(ranks) > 1 and ranks[1][COUNT] == 2:
                hand_type = 'Full house'
            else:
                hand_type = 'Three of a kind'
        elif ranks[0][COUNT] == 2:
            if len(ranks) > 1 and ranks[1][COUNT] == 2:
                hand_type = 'Two pair'
            else:
                hand_type = 'One pair'
        else:
            hand_type = 'High card'
        result = hand_type
        return result
    
    def get_bidding_hand_sort(self, bidding_hand):
        HAND, BID = 0, 1
        hand = bidding_hand[HAND]
        best_hand_type_value = -1
        if self.joker_rule and 'J' in hand:
            for rank in '23456789TJQKA':
                new_hand = hand.replace('J', rank)
                best_hand_type_value = max(
                    best_hand_type_value,
                    self.hand_type_values[self.get_hand_type(new_hand)],
                )
        else:
            best_hand_type_value = self.hand_type_values[self.get_hand_type(hand)]
        # Temporarily change the value of J
        if self.joker_rule and 'J' in hand:
            self.rank_values['J'] = 1
        bidding_hand_value = (
            best_hand_type_value,
            self.rank_values[hand[0]],
            self.rank_values[hand[1]],
            self.rank_values[hand[2]],
            self.rank_values[hand[3]],
            self.rank_values[hand[4]],
        )
        result = bidding_hand_value
        # Change the value of J back
        self.rank_values['J'] = 11
        return result
    
    def solve(self, bidding_hands):
        self.joker_rule = False
        bidding_hands.sort(key=self.get_bidding_hand_sort)
        winnings = []
        for multiplier, (hand, bid) in enumerate(bidding_hands, start=1):
            winnings.append(multiplier * bid)
        result = sum(winnings)
        return result
    
    def solve2(self, bidding_hands):
        self.joker_rule = True
        bidding_hands.sort(key=self.get_bidding_hand_sort)
        winnings = []
        for multiplier, (hand, bid) in enumerate(bidding_hands, start=1):
            winnings.append(multiplier * bid)
        result = sum(winnings)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        bidding_hands = self.get_bidding_hands(raw_input_lines)
        solutions = (
            self.solve(bidding_hands),
            self.solve2(bidding_hands),
            )
        result = solutions
        return result

class Day06: # Wait For It
    '''
    https://adventofcode.com/2023/day/6
    '''
    def get_races(self, raw_input_lines: List[str]):
        races = []
        raw_time = raw_input_lines[0].split()
        raw_distance = raw_input_lines[1].split()
        for i in range(1, len(raw_time)):
            races.append((int(raw_time[i]), int(raw_distance[i])))
        result = races
        return result
    
    def solve(self, races):
        ways_to_win = []
        for (time, record) in races:
            wins = {}
            for speed in range(time + 1):
                time_left = time - speed
                distance = speed * time_left
                if distance > record:
                    wins[speed] = distance
            ways_to_win.append(wins)
        margin_of_error = 1
        for wins in ways_to_win:
            margin_of_error *= len(wins)
        result = margin_of_error
        return result
    
    def solve2(self, races):
        combined_time = ''
        combined_distance = ''
        for (time, record) in races:
            combined_time += str(time)
            combined_distance += str(record)
        new_races = []
        new_races.append((int(combined_time), int(combined_distance)))
        result = self.solve(new_races)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        races = self.get_races(raw_input_lines)
        solutions = (
            self.solve(races),
            self.solve2(races),
            )
        result = solutions
        return result

class Day05: # If You Give A Seed A Fertilizer
    '''
    https://adventofcode.com/2023/day/5
    '''
    def get_almanac(self, raw_input_lines: List[str]):
        seeds = list(map(int, raw_input_lines[0].split(': ')[1].split(' ')))
        almanac = {
            'seeds': seeds,
            'seed-to-soil': {},
            'soil-to-fertilizer': {},
            'fertilizer-to-water': {},
            'water-to-light': {},
            'light-to-temperature': {},
            'temperature-to-humidity': {},
            'humidity-to-location': {},
        }
        section = None
        for raw_input_line in raw_input_lines[1:]:
            if len(raw_input_line) < 1:
                pass
            elif raw_input_line[-1] == ':':
                section = raw_input_line.split(' ')[0]
            else:
                mapping = tuple(map(int, raw_input_line.split(' ')))
                map_key = (mapping[1], mapping[1] + mapping[2] - 1)
                map_value = mapping[0] - mapping[1]
                almanac[section][map_key] = map_value
        result = almanac
        return result
    
    def solve(self, almanac):
        locations = []
        seeds = almanac['seeds']
        # Convert seed to soil
        soils = []
        for seed in seeds:
            for (range_start, range_end), offset in almanac['seed-to-soil'].items():
                if range_start <= seed <= range_end:
                    soils.append(seed + offset)
                    break
            else:
                soils.append(seed)
        # Convert soil to fertilizer
        fertilizers = []
        for soil in soils:
            for (range_start, range_end), offset in almanac['soil-to-fertilizer'].items():
                if range_start <= soil <= range_end:
                    fertilizers.append(soil + offset)
                    break
            else:
                fertilizers.append(soil)
        # Convert fertilizer to water
        waters = []
        for fertilizer in fertilizers:
            for (range_start, range_end), offset in almanac['fertilizer-to-water'].items():
                if range_start <= fertilizer <= range_end:
                    waters.append(fertilizer + offset)
                    break
            else:
                waters.append(fertilizer)
        # Convert water to light
        lights = []
        for water in waters:
            for (range_start, range_end), offset in almanac['water-to-light'].items():
                if range_start <= water <= range_end:
                    lights.append(water + offset)
                    break
            else:
                lights.append(water)
        # Convert light to temperature
        temperatures = []
        for light in lights:
            for (range_start, range_end), offset in almanac['light-to-temperature'].items():
                if range_start <= light <= range_end:
                    temperatures.append(light + offset)
                    break
            else:
                temperatures.append(light)
        # Convert temperature to humidity
        humiditys = []
        for temperature in temperatures:
            for (range_start, range_end), offset in almanac['temperature-to-humidity'].items():
                if range_start <= temperature <= range_end:
                    humiditys.append(temperature + offset)
                    break
            else:
                humiditys.append(temperature)
        # Convert humidity to location
        locations = []
        for humidity in humiditys:
            for (range_start, range_end), offset in almanac['humidity-to-location'].items():
                if range_start <= humidity <= range_end:
                    locations.append(humidity + offset)
                    break
            else:
                locations.append(humidity)
        result = min(locations)
        return result

    def solve2(self, almanac):
        # Work backwards to figure out seeds of interest
        # Get initial points of interest from humidity
        humiditys_of_interest = set()
        for (start, end), offset in almanac['humidity-to-location'].items():
            humiditys_of_interest.add(start)
            humiditys_of_interest.add(end)
        # Convert humidity to temperature
        temperatures_of_interest = set()
        for humidity in humiditys_of_interest:
            for (start, end), offset in almanac['temperature-to-humidity'].items():
                if (start + offset <= humidity <= end + offset):
                    temperatures_of_interest.add(start)
                    temperatures_of_interest.add(end)
                    temperatures_of_interest.add(humidity - offset)
            temperatures_of_interest.add(humidity)
        # Convert temperature to light
        lights_of_interest = set()
        for temperature in temperatures_of_interest:
            for (start, end), offset in almanac['light-to-temperature'].items():
                if (start + offset <= temperature <= end + offset):
                    lights_of_interest.add(start)
                    lights_of_interest.add(end)
                    lights_of_interest.add(temperature - offset)
            lights_of_interest.add(humidity)
        # Convert light to water
        waters_of_interest = set()
        for light in lights_of_interest:
            for (start, end), offset in almanac['water-to-light'].items():
                if (start + offset <= light <= end + offset):
                    waters_of_interest.add(start)
                    waters_of_interest.add(end)
                    waters_of_interest.add(light - offset)
            waters_of_interest.add(light)
        # Convert water to fertilizer
        fertilizers_of_interest = set()
        for water in waters_of_interest:
            for (start, end), offset in almanac['fertilizer-to-water'].items():
                if (start + offset <= water <= end + offset):
                    fertilizers_of_interest.add(start)
                    fertilizers_of_interest.add(end)
                    fertilizers_of_interest.add(water - offset)
            fertilizers_of_interest.add(water)
        # Convert fertilizer to soil
        soils_of_interest = set()
        for fertilizer in fertilizers_of_interest:
            for (start, end), offset in almanac['soil-to-fertilizer'].items():
                if (start + offset <= fertilizer <= end + offset):
                    soils_of_interest.add(start)
                    soils_of_interest.add(end)
                    soils_of_interest.add(fertilizer - offset)
            soils_of_interest.add(fertilizer)
        # Convert soil to seed
        seeds_of_interest = set()
        for soil in soils_of_interest:
            for (start, end), offset in almanac['seed-to-soil'].items():
                if (start + offset <= soil <= end + offset):
                    seeds_of_interest.add(start)
                    seeds_of_interest.add(end)
                    seeds_of_interest.add(soil - offset)
            seeds_of_interest.add(soil)
        valid_seeds = set()
        # Find seeds of interest in original seeds
        for seed in seeds_of_interest:
            for index in range(0, len(almanac['seeds']), 2):
                seed_start = almanac['seeds'][index]
                seed_count = almanac['seeds'][index + 1]
                seed_end = seed_start + seed_count - 1
                if seed_start <= seed <= seed_end:
                    valid_seeds.add(seed)
                    break
        revised_almanac = copy.deepcopy(almanac)
        revised_almanac['seeds'] = list(valid_seeds)
        min_location = self.solve(revised_almanac)
        result = min_location
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        almanac = self.get_almanac(raw_input_lines)
        solutions = (
            self.solve(almanac),
            self.solve2(almanac),
            )
        result = solutions
        return result

class Day04: # Scratchcards
    '''
    https://adventofcode.com/2023/day/4
    '''
    def get_cards(self, raw_input_lines: List[str]):
        cards = {}
        for raw_input_line in raw_input_lines:
            a, b = raw_input_line.split(': ')
            card_id = int(a.split()[1])
            c, d = b.split(' | ')
            winners = set(map(int, c.split()))
            numbers = set(map(int, d.split()))
            card = {
                'winners': winners,
                'numbers': numbers,
            }
            cards[card_id] = card
        result = cards
        return result
    
    def solve(self, cards):
        winnings = []
        for card in cards.values():
            winning_numbers = card['winners'] & card['numbers']
            points = 0
            for _ in winning_numbers:
                if points < 1:
                    points = 1
                else:
                    points *= 2
            winnings.append(points)
        result = sum(winnings)
        return result
    
    def solve2(self, cards):
        copies = {}
        # Calculate copies in reverse order by card ID
        for card_id in reversed(sorted(cards.keys())):
            card = cards[card_id]
            matches = len(card['winners'] & card['numbers'])
            copies[card_id] = 1
            for offset in range(1, matches + 1):
                copies[card_id] += copies[card_id + offset]
        result = sum(copies.values())
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        cards = self.get_cards(raw_input_lines)
        solutions = (
            self.solve(cards),
            self.solve2(cards),
            )
        result = solutions
        return result

class Day03: # Gear Ratios
    '''
    https://adventofcode.com/2023/day/3
    '''
    def get_schematic(self, raw_input_lines: List[str]):
        schematic = {}
        for row, raw_input_line in enumerate(raw_input_lines):
            for col, char in enumerate(raw_input_line):
                if char != '.':
                    schematic[(row, col)] = char
        result = schematic
        return result
    
    def solve(self, schematic):
        work = set()
        for (row, col), char in schematic.items():
            if char not in '0123456789':
                for (r, c) in (
                    (-1, -1),
                    (-1,  0),
                    (-1,  1),
                    ( 0, -1),
                    ( 0,  1),
                    ( 1, -1),
                    ( 1,  0),
                    ( 1,  1),
                ):
                    if (
                        (row + r, col + c) in schematic and
                        schematic[(row + r, col + c)] in '0123456789'
                    ):
                        work.add((row + r, col + c))
        valid_part_numbers = []
        while len(work) > 0:
            (row, col) = work.pop()
            # Expand left to find the start
            start_col = col
            while (
                (row, start_col - 1) in schematic and 
                schematic[(row, start_col - 1)] in '0123456789'
            ):
                start_col -= 1
                work.discard((row, start_col))
            # Expand right to find the end
            end_col = col
            while (
                (row, end_col + 1) in schematic and 
                schematic[(row, end_col + 1)] in '0123456789'
            ):
                end_col += 1
                work.discard((row, end_col))
            # Calculate the whole part number
            valid_part_number = 0
            for col in range(start_col, end_col + 1):
                number = int(schematic[(row, col)])
                valid_part_number = 10 * valid_part_number + number
            valid_part_numbers.append(valid_part_number)
        result = sum(valid_part_numbers)
        return result
    
    def solve2(self, schematic):
        work = set()
        for (row, col), char in schematic.items():
            if char == '*':
                for (r, c) in (
                    (-1, -1),
                    (-1,  0),
                    (-1,  1),
                    ( 0, -1),
                    ( 0,  1),
                    ( 1, -1),
                    ( 1,  0),
                    ( 1,  1),
                ):
                    if (
                        (row + r, col + c) in schematic and
                        schematic[(row + r, col + c)] in '0123456789'
                    ):
                        work.add((row + r, col + c))
        # Locate part numbers
        part_number_map = {} # (row, col) -> part_number_id
        part_numbers = {} # part_number_id -> part_number
        while len(work) > 0:
            part_number_id = len(part_numbers)
            (row, col) = work.pop()
            part_number_map[(row, col)] = part_number_id
            # Expand left to find the start
            start_col = col
            while (
                (row, start_col - 1) in schematic and 
                schematic[(row, start_col - 1)] in '0123456789'
            ):
                start_col -= 1
                work.discard((row, start_col))
                part_number_map[(row, start_col)] = part_number_id
            # Expand right to find the end
            end_col = col
            while (
                (row, end_col + 1) in schematic and 
                schematic[(row, end_col + 1)] in '0123456789'
            ):
                end_col += 1
                work.discard((row, end_col))
                part_number_map[(row, start_col)] = part_number_id
            # Calculate the whole part number
            part_number = 0
            for col in range(start_col, end_col + 1):
                number = int(schematic[(row, col)])
                part_number = 10 * part_number + number
            part_numbers[part_number_id] = part_number
        gear_ratios = []
        for (row, col), char in schematic.items():
            if char == '*':
                neighbors = set()
                for (r, c) in (
                    (-1, -1),
                    (-1,  0),
                    (-1,  1),
                    ( 0, -1),
                    ( 0,  1),
                    ( 1, -1),
                    ( 1,  0),
                    ( 1,  1),
                ):
                    if (row + r, col + c) in part_number_map:
                        neighbor = part_number_map[(row + r, col + c)]
                        neighbors.add(neighbor)
                if len(neighbors) == 2:
                    gear_ratio = part_numbers[neighbors.pop()]
                    gear_ratio *= part_numbers[neighbors.pop()]
                    gear_ratios.append(gear_ratio)
        result = sum(gear_ratios)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        schematic = self.get_schematic(raw_input_lines)
        solutions = (
            self.solve(schematic),
            self.solve2(schematic),
            )
        result = solutions
        return result

class Day02: # Cube Conundrum
    '''
    https://adventofcode.com/2023/day/2
    '''
    def get_games(self, raw_input_lines: List[str]):
        games = {}
        for raw_input_line in raw_input_lines:
            raw_game_id, raw_game_info = raw_input_line.split(': ')
            game_id = int(raw_game_id.split(' ')[1])
            game = []
            for raw_trial_info in raw_game_info.split('; '):
                trial = {}
                for raw_color_info in raw_trial_info.split(', '):
                    raw_amount, color = raw_color_info.split(' ')
                    amount = int(raw_amount)
                    trial[color] = amount
                game.append(trial)
            games[game_id] = game
        result = games
        return result
    
    def solve(self, games):
        valid_game_ids = set()
        max_cubes = {
            'red': 12,
            'green': 13,
            'blue': 14,
        }
        for game_id, game in games.items():
            valid_game = True
            for trial in game:
                for color, amount in trial.items():
                    if amount > max_cubes[color]:
                        valid_game = False
                        break
            if valid_game:
                valid_game_ids.add(game_id)
        result = sum(valid_game_ids)
        return result
    
    def solve2(self, games):
        min_cubes = {}
        for game_id, game in games.items():
            min_cube = collections.defaultdict(int)
            for trial in game:
                for color, amount in trial.items():
                    min_cube[color] = max(min_cube[color], amount)
            min_cubes[game_id] = min_cube
        powers = {}
        for cube_id, min_cube in min_cubes.items():
            power = 1
            for amount in min_cube.values():
                power *= amount
            powers[cube_id] = power
        result = sum(powers.values())
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        games = self.get_games(raw_input_lines)
        solutions = (
            self.solve(games),
            self.solve2(games),
            )
        result = solutions
        return result

class Day01: # Trebuchet?!
    '''
    https://adventofcode.com/2023/day/1
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, parsed_input):
        calibration_values = []
        for row_data in parsed_input:
            calibration_value = 0
            for char in row_data:
                try:
                    value = int(char)
                    calibration_value = 10 * value
                    break
                except ValueError:
                    pass
            for char in reversed(row_data):
                try:
                    value = int(char)
                    calibration_value += value
                    break
                except ValueError:
                    pass
            calibration_values.append(calibration_value)
        result = sum(calibration_values)
        return result
    
    def solve2(self, parsed_input):
        numbers = {
            'one': 1,
            'two': 2,
            'three': 3,
            'four': 4,
            'five': 5,
            'six': 6,
            'seven': 7,
            'eight': 8,
            'nine': 9,
        }
        calibration_values = []
        for row_data in parsed_input:
            parts = []
            for i in range(len(row_data)):
                try:
                    value = int(row_data[i])
                    parts.append(value)
                    break
                except ValueError:
                    for number in sorted(numbers, key=len):
                        if row_data[i:].startswith(number):
                            value = numbers[number]
                            parts.append(value)
                            break
                    if len(parts) > 0:
                        break
            for i in reversed(range(len(row_data))):
                try:
                    value = int(row_data[i])
                    parts.append(value)
                    break
                except ValueError:
                    for number in sorted(numbers, key=len):
                        if row_data[i:].startswith(number):
                            value = numbers[number]
                            parts.append(value)
                            break
                    if len(parts) > 1:
                        break
            calibration_values.append(10 * parts[0] + parts[1])
        result = sum(calibration_values)
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
    python AdventOfCode2023.py 2 < inputs/2023day02.in
    '''
    solvers = {
        1: (Day01, 'Trebuchet?!'),
        2: (Day02, 'Cube Conundrum'),
        3: (Day03, 'Gear Ratios'),
        4: (Day04, 'Scratchcards'),
        5: (Day05, 'If You Give A Seed A Fertilizer'),
        6: (Day06, 'Wait For It'),
        7: (Day07, 'Camel Cards'),
        8: (Day08, 'Haunted Wasteland'),
        9: (Day09, 'Mirage Maintenance'),
       10: (Day10, 'Pipe Maze'),
       11: (Day11, 'Cosmic Expansion'),
       12: (Day12, 'Hot Springs'),
       13: (Day13, 'Point of Incidence'),
       14: (Day14, 'Parabolic Reflector Dish'),
       15: (Day15, 'Lens Library'),
    #    16: (Day16, 'Unknown'),
    #    17: (Day17, 'Unknown'),
    #    18: (Day18, 'Unknown'),
    #    19: (Day19, 'Unknown'),
    #    20: (Day20, 'Unknown'),
    #    21: (Day21, 'Unknown'),
    #    22: (Day22, 'Unknown'),
    #    23: (Day23, 'Unknown'),
    #    24: (Day24, 'Unknown'),
    #    25: (Day25, 'Unknown'),
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
