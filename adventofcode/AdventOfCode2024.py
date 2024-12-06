'''
Created on 2024-11-30

@author: Sestren
'''
import argparse
import functools

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
    https://adventofcode.com/2024/day/?
    '''
    def get_parsed_input(self, raw_input_lines: list[str]):
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

class Day06: # Guard Gallivant
    '''
    https://adventofcode.com/2024/day/6
    '''
    directions = {
        'UP'   : (-1,  0, 'RIGHT'),
        'RIGHT': ( 0,  1, 'DOWN'),
        'DOWN' : ( 1,  0, 'LEFT'),
        'LEFT' : ( 0, -1, 'UP'),
    }
    def get_parsed_input(self, raw_input_lines: list[str]):
        rows = len(raw_input_lines)
        cols = len(raw_input_lines[0])
        obstacles = set()
        guard = (0, 0, 'UNKNOWN')
        result = []
        for (row, raw_input_line) in enumerate(raw_input_lines):
            for (col, char) in enumerate(raw_input_line):
                if char == '#':
                    obstacles.add((row, col))
                elif char in '>^v<':
                    directions = {
                        '>': 'RIGHT',
                        '^': 'UP',
                        'v': 'DOWN',
                        '<': 'LEFT',
                    }
                    guard = (row, col, directions[char])
            result.append(raw_input_line)
        result = (rows, cols, obstacles, guard)
        return result
    
    def solve(self, rows: int, cols: int, obstacles: set, guard: tuple):
        visits = set()
        while True:
            (row, col, direction) = guard
            if not (0 <= row < rows and 0 <= col < cols):
                break
            visits.add((row, col))
            while True:
                facing_row = row + self.directions[direction][0]
                facing_col = col + self.directions[direction][1]
                if (facing_row, facing_col) in obstacles:
                    # Turn 90 degrees
                    direction = self.directions[direction][2]
                else:
                    # Take a step forward
                    guard = (facing_row, facing_col, direction)
                    break
        result = len(visits)
        return result
    
    def solve2(self, rows: int, cols: int, original_obstacles: set, original_guard: tuple):
        positions = set()
        for obstacle_row in range(rows):
            for obstacle_col in range(cols):
                if (obstacle_row, obstacle_col) in original_obstacles:
                    continue
                if (obstacle_row, obstacle_col) == (original_guard[0], original_guard[1]):
                    continue
                obstacles = set(original_obstacles)
                obstacles.add((obstacle_row, obstacle_col))
                guard = original_guard
                visits = set()
                while True:
                    (row, col, direction) = guard
                    if not (0 <= row < rows and 0 <= col < cols):
                        break
                    if (row, col, direction) in visits:
                        positions.add((obstacle_row, obstacle_col))
                        break
                    visits.add((row, col, direction))
                    while True:
                        facing_row = row + self.directions[direction][0]
                        facing_col = col + self.directions[direction][1]
                        if (facing_row, facing_col) in obstacles:
                            # Turn 90 degrees
                            direction = self.directions[direction][2]
                        else:
                            # Take a step forward
                            guard = (facing_row, facing_col, direction)
                            break
        result = len(positions)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        (rows, cols, obstacles, guard) = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(rows, cols, obstacles, guard),
            self.solve2(rows, cols, obstacles, guard),
        )
        result = solutions
        return result

class Day05: # Print Queue
    '''
    https://adventofcode.com/2024/day/5
    '''
    def get_parsed_input(self, raw_input_lines: list[str]):
        rules = set()
        updates = []
        mode = 'RULES'
        for raw_input_line in raw_input_lines:
            if len(raw_input_line) == 0:
                mode = 'UPDATES'
            else:
                if mode == 'RULES':
                    rules.add(tuple(map(int, raw_input_line.split('|'))))
                elif mode == 'UPDATES':
                    updates.append(list(map(int, raw_input_line.split(','))))
                else:
                    raise Exception('Unknown mode')
        result = (rules, updates)
        return result
    
    def collated(self, rules, update):
        def lt(a, b):
            result = 0
            if (a, b) in rules:
                result = -1
            elif (b, a) in rules:
                result = 1
            return result
        result = sorted(update, key=functools.cmp_to_key(lt))
        return result
    
    def solve(self, rules, updates):
        result = 0
        for update in updates:
            collated_update = self.collated(rules, update)
            if collated_update == update:
                n = len(collated_update) // 2
                result += collated_update[n]
        return result
    
    def solve2(self, rules, updates):
        result = 0
        for update in updates:
            collated_update = self.collated(rules, update)
            if collated_update != update:
                n = len(collated_update) // 2
                result += collated_update[n]
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        (rules, updates) = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(rules, updates),
            self.solve2(rules, updates),
        )
        result = solutions
        return result

class Day04: # Ceres Search
    '''
    https://adventofcode.com/2024/day/4
    '''
    def get_parsed_input(self, raw_input_lines: list[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, parsed_input):
        word = 'XMAS'
        word_count = 0
        for start_row in range(len(parsed_input)):
            for start_col in range(len(parsed_input[start_row])):
                if parsed_input[start_row][start_col] == word[0]:
                    for (row_mul, col_mul) in (
                        (-1, -1),
                        (-1,  0),
                        (-1,  1),
                        ( 0, -1),
                        ( 0,  1),
                        ( 1, -1),
                        ( 1,  0),
                        ( 1,  1),
                    ):
                        for i in range(len(word)):
                            row = start_row + row_mul * i
                            col = start_col + col_mul * i
                            if not(0 <= row < len(parsed_input)):
                                break
                            if not(0 <= col < len(parsed_input[row])):
                                break
                            if parsed_input[row][col] != word[i]:
                                break
                        else:
                            word_count += 1
        result = word_count
        return result
    
    def solve2(self, parsed_input):
        word_count = 0
        for start_row in range(len(parsed_input)):
            for start_col in range(len(parsed_input[start_row])):
                if parsed_input[start_row][start_col] == 'A':
                    corners = []
                    for (row_offset, col_offset) in (
                        (-1, -1),
                        (-1,  1),
                        ( 1, -1),
                        ( 1,  1),
                    ):
                        corners.append('')
                        row = start_row + row_offset
                        col = start_col + col_offset
                        if not(0 <= row < len(parsed_input)):
                            continue
                        if not(0 <= col < len(parsed_input[row])):
                            continue
                        corners[-1] = parsed_input[row][col]
                    if (
                        (
                            corners[0] + 'A' + corners[3] == 'MAS' or
                            corners[0] + 'A' + corners[3] == 'SAM'
                        ) and
                        (
                            corners[1] + 'A' + corners[2] == 'MAS' or
                            corners[1] + 'A' + corners[2] == 'SAM'
                        )
                    ):
                        word_count += 1
        result = word_count
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

class Day03: # Mull It Over
    '''
    https://adventofcode.com/2024/day/3
    '''
    def get_parsed_input(self, raw_input_lines: list[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, parsed_input):
        multiplications = []
        for line in parsed_input:
            for i in range(len(line)):
                if line[i:].startswith('mul('):
                    length = line[i:i + 12].find(')')
                    if length >= 0:
                        segment = line[i:i + length + 1]
                        try:
                            multiplication = tuple(map(int, segment[4:-1].split(',')))
                            multiplications.append(multiplication)
                        except ValueError:
                            pass
        result = sum((a * b for (a, b) in multiplications))
        return result
    
    def solve2(self, parsed_input):
        instructions = []
        for line in parsed_input:
            for i in range(len(line)):
                if line[i:].startswith('mul('):
                    length = line[i:i + 12].find(')')
                    if length >= 0:
                        segment = line[i:i + length + 1]
                        try:
                            (a, b) = tuple(map(int, segment[4:-1].split(',')))
                            instructions.append(('MUL', a, b))
                        except ValueError:
                            pass
                elif line[i:].startswith('do()'):
                    instructions.append(('DO', ))
                elif line[i:].startswith('don\'t()'):
                    instructions.append(('DONT', ))
        result = 0
        do = True
        for instruction in instructions:
            op = instruction[0]
            if op == 'MUL':
                (a, b) = instruction[1:]
                if do:
                    result += a * b
            elif op == 'DO':
                do = True
            elif op == 'DONT':
                do = False
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

class Day02: # Red-Nosed Reports
    '''
    https://adventofcode.com/2024/day/2
    '''
    def get_reports(self, raw_input_lines: list[str]):
        reports = []
        for raw_input_line in raw_input_lines:
            report = tuple(map(int, raw_input_line.split()))
            reports.append(report)
        result = reports
        return result
    
    def check_report(self, report):
        safe_ind = True
        prev_diff = 0
        for i in range(1, len(report)):
            diff = report[i] - report[i - 1]
            if not(1 <= abs(diff) <= 3):
                safe_ind = False
                break
            if (
                (prev_diff < 0 and diff > 0) or
                (prev_diff > 0 and diff < 0)
            ):
                safe_ind = False
                break
            prev_diff = diff
        result = safe_ind
        return result
    
    def solve(self, reports):
        safe_count = 0
        for report in reports:
            if self.check_report(report):
                safe_count += 1
        result = safe_count
        return result
    
    def solve2(self, reports):
        safe_count = 0
        for report in reports:
            safe_ind = False
            if self.check_report(report):
                safe_ind = True
            else:
                for i in range(len(report)):
                    modified_report = tuple(report[:i] + report[i + 1:])
                    if self.check_report(modified_report):
                        safe_ind = True
                        break
            if safe_ind:
                safe_count += 1
        result = safe_count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        reports = self.get_reports(raw_input_lines)
        solutions = (
            self.solve(reports),
            self.solve2(reports),
        )
        result = solutions
        return result

class Day01: # Historian Hysteria
    '''
    https://adventofcode.com/2024/day/1
    '''
    def get_list_pairs(self, raw_input_lines: list[str]):
        left = []
        right = []
        for raw_input_line in raw_input_lines:
            (a, b) = map(int, raw_input_line.split())
            left.append(a)
            right.append(b)
        result = (left, right)
        return result
    
    def solve(self, left, right):
        sorted_left = sorted(left)
        sorted_right = sorted(right)
        assert len(sorted_left) == len(sorted_right)
        diffs = []
        for i in range(len(sorted_left)):
            diff = abs(sorted_left[i] - sorted_right[i])
            diffs.append(diff)
        result = sum(diffs)
        return result
    
    def solve2(self, left, right):
        right_counts = {}
        for num in right:
            if num not in right_counts:
                right_counts[num] = 0
            right_counts[num] += 1
        similarity_scores = []
        for num in left:
            if num not in right_counts:
                right_counts[num] = 0
            similarity_score = num * right_counts[num]
            similarity_scores.append(similarity_score)
        result = sum(similarity_scores)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        (left, right) = self.get_list_pairs(raw_input_lines)
        solutions = (
            self.solve(left, right),
            self.solve2(left, right),
        )
        result = solutions
        return result

if __name__ == '__main__':
    '''
    Usage
    python AdventOfCode2024.py 5 < inputs/2024day05.in
    '''
    solvers = {
        1: (Day01, 'Historian Hysteria'),
        2: (Day02, 'Red-Nosed Reports'),
        3: (Day03, 'Mull It Over'),
        4: (Day04, 'Ceres Search'),
        5: (Day05, 'Print Queue'),
        6: (Day06, 'Guard Gallivant'),
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
