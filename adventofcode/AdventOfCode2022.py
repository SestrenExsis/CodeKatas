'''
Created on 2022-11-30

@author: Sestren
'''
import argparse
import collections
import copy
import functools
import heapq
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

class DeviceCPU:
    def __init__(self, instructions):
        self.x = 1
        self.instructions = instructions
        self.pc = 0
        self.cycles_remaining = 0
    
    def step(self):
        instruction = self.instructions[self.pc]
        if instruction[0] == 'noop':
            # NOOP takes 1 clock cycle to compute
            self.pc += 1
        elif instruction[0] == 'addx':
            # ADDX takes 2 clock cycles to compute
            if self.cycles_remaining == 0:
                self.cycles_remaining = 1
            else:
                self.cycles_remaining -= 1
                if self.cycles_remaining < 1:
                    self.x += instruction[1]
                    self.pc += 1

class DeviceCRT:
    def __init__(self):
        self.rows = 6
        self.cols = 40
        self.display = [['.'] * self.cols for _ in range(self.rows)]
        self.pos = (0, 0)
    
    def step(self, x):
        row, col = self.pos
        col += 1
        if col >= self.cols:
            col = 0
            row += 1
            if row >= self.rows:
                row = 0
        cell = '.'
        if col in (x - 1, x, x + 1):
            cell = '#'
        self.display[row][col] = cell
        self.pos = (row, col)
    
    def get_display(self):
        display = []
        for row in range(self.rows):
            row_data = ''.join(self.display[row])
            display.append(row_data)
        result = display
        return result

class Template: # Template
    '''
    https://adventofcode.com/2022/day/?
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

class Day11: # Monkey in the Middle
    '''
    https://adventofcode.com/2022/day/11
    '''
    def get_monkeys(self, raw_input_lines: List[str]):
        monkeys = {}
        index = 0
        while index < len(raw_input_lines):
            monkey_id = raw_input_lines[index][-2]
            items = list(reversed(list(map(int,
                raw_input_lines[index + 1].split(': ')[1].split(', ')
            ))))
            operation = raw_input_lines[index + 2].split(' = ')[1]
            factor = int(raw_input_lines[index + 3].split(' ')[-1])
            true_monkey = raw_input_lines[index + 4].split(' ')[-1]
            false_monkey = raw_input_lines[index + 5].split(' ')[-1]
            monkey = {
                'items': collections.deque(items),
                'operation': operation,
                'factor': factor,
                'if_true': true_monkey,
                'if_false': false_monkey,
                'inspections': 0,
            }
            monkeys[monkey_id] = monkey
            index += 7
        result = monkeys
        return result
    
    def solve(self, monkeys):
        for round_id in range(20):
            for monkey_id, monkey in monkeys.items():
                while len(monkey['items']) > 0:
                    old = monkey['items'].pop()
                    expression = monkey['operation'].replace('old', str(old))
                    new = eval(expression) // 3
                    monkey['inspections'] += 1
                    target = monkey['if_false']
                    if new % monkey['factor'] == 0:
                        target = monkey['if_true']
                    monkeys[target]['items'].appendleft(new)
        monkey_business = []
        for monkey in monkeys.values():
            heapq.heappush(monkey_business, -1 * monkey['inspections'])
        result = -1 * heapq.heappop(monkey_business)
        result *= -1 * heapq.heappop(monkey_business)
        return result
    
    def solve2(self, monkeys):
        modulo = 1
        for monkey in monkeys.values():
            modulo *= monkey['factor']
        for round_id in range(10_000):
            for monkey_id, monkey in monkeys.items():
                while len(monkey['items']) > 0:
                    old = monkey['items'].pop()
                    expression = monkey['operation'].replace('old', str(old))
                    new = eval(expression) % modulo
                    monkey['inspections'] += 1
                    target = monkey['if_false']
                    if new % monkey['factor'] == 0:
                        target = monkey['if_true']
                    monkeys[target]['items'].appendleft(new)
        monkey_business = []
        for monkey in monkeys.values():
            heapq.heappush(monkey_business, -1 * monkey['inspections'])
        result = -1 * heapq.heappop(monkey_business)
        result *= -1 * heapq.heappop(monkey_business)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        monkeys = self.get_monkeys(raw_input_lines)
        solutions = (
            self.solve(copy.deepcopy(monkeys)),
            self.solve2(copy.deepcopy(monkeys)),
            )
        result = solutions
        return result

class Day10: # Cathode-Ray Tube
    '''
    https://adventofcode.com/2022/day/10
    '''
    def get_instructions(self, raw_input_lines: List[str]):
        instructions = []
        for raw_input_line in raw_input_lines:
            parts = raw_input_line.split()
            instruction = tuple(parts)
            if parts[0] == 'addx':
                instruction = (parts[0], int(parts[1]))
            instructions.append(instruction)
        result = instructions
        return result
    
    def solve(self, instructions):
        cpu = DeviceCPU(instructions)
        cycles = {}
        cycle_id = 1
        while cpu.pc < len(cpu.instructions):
            signal_strength = cpu.x * cycle_id
            cycles[cycle_id] = signal_strength
            cpu.step()
            cycle_id += 1
        result = sum(
            signal_strength for
            (cycle, signal_strength) in cycles.items() if
            cycle in (20, 60, 100, 140, 180, 220)
        )
        return result
    
    def solve2(self, instructions):
        cpu = DeviceCPU(instructions)
        crt = DeviceCRT()
        while cpu.pc < len(cpu.instructions):
            cpu.step()
            crt.step(cpu.x)
        result = '\n' + '\n'.join(crt.get_display())
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

class Day09: # Rope Bridge
    '''
    https://adventofcode.com/2022/day/9
    '''
    dirs = {
        'U': (-1,  0),
        'D': ( 1,  0),
        'L': ( 0, -1),
        'R': ( 0,  1),
    }

    def get_steps(self, raw_input_lines: List[str]):
        steps = []
        for raw_input_line in raw_input_lines:
            dir, count = raw_input_line.split(' ')
            steps.append((dir, int(count)))
        result = steps
        return result
    
    def solve(self, steps):
        tail_visits = set()
        tail = (0, 0)
        head = (0, 0)
        tail_visits.add(tail)
        for (dir, count) in steps:
            (drow, dcol) = self.dirs[dir]
            for _ in range(count):
                head = (head[0] + drow, head[1] + dcol)
                rowdiff = abs(head[0] - tail[0])
                coldiff = abs(head[1] - tail[1])
                if rowdiff > 1 or coldiff > 1 or sum((rowdiff, coldiff)) > 2:
                    if head[0] < tail[0]:
                        tail = (tail[0] - 1, tail[1])
                    elif head[0] > tail[0]:
                        tail = (tail[0] + 1, tail[1])
                    if head[1] < tail[1]:
                        tail = (tail[0], tail[1] - 1)
                    elif head[1] > tail[1]:
                        tail = (tail[0], tail[1] + 1)
                tail_visits.add(tail)
        result = len(tail_visits)
        return result
    
    def solve2(self, steps, knot_count: int=10):
        knots = []
        for _ in range(knot_count):
            knots.append((0, 0))
        head_visits = set()
        tail_visits = set()
        tail_visits.add(knots[-1])
        head_visits.add(knots[0])
        for (dir, count) in steps:
            (drow, dcol) = self.dirs[dir]
            for _ in range(count):
                knots[0] = (knots[0][0] + drow, knots[0][1] + dcol)
                head_visits.add(knots[0])
                for knot_id in range(1, knot_count):
                    rowdiff = abs(knots[knot_id - 1][0] - knots[knot_id][0])
                    coldiff = abs(knots[knot_id - 1][1] - knots[knot_id][1])
                    if rowdiff > 1 or coldiff > 1 or sum((rowdiff, coldiff)) > 2:
                        if knots[knot_id - 1][0] < knots[knot_id][0]:
                            knots[knot_id] = (knots[knot_id][0] - 1, knots[knot_id][1])
                        elif knots[knot_id - 1][0] > knots[knot_id][0]:
                            knots[knot_id] = (knots[knot_id][0] + 1, knots[knot_id][1])
                        if knots[knot_id - 1][1] < knots[knot_id][1]:
                            knots[knot_id] = (knots[knot_id][0], knots[knot_id][1] - 1)
                        elif knots[knot_id - 1][1] > knots[knot_id][1]:
                            knots[knot_id] = (knots[knot_id][0], knots[knot_id][1] + 1)
                tail_visits.add(knots[-1])
        result = len(tail_visits)
        # self.visualize(tail_visits)
        return result
    
    def visualize(self, visits: set):
        min_row = min(visit[0] for visit in visits)
        min_col = min(visit[1] for visit in visits)
        max_row = max(visit[0] for visit in visits)
        max_col = max(visit[1] for visit in visits)
        for row in range(min_row, max_row + 1):
            row_data = ''
            for col in range(min_col, max_col + 1):
                cell = '.'
                if (row, col) in visits:
                    cell = '#'
                row_data += cell
            print(row_data)
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        steps = self.get_steps(raw_input_lines)
        solutions = (
            self.solve(steps),
            self.solve2(steps),
            )
        result = solutions
        return result

class Day08: # Treetop Tree House
    '''
    https://adventofcode.com/2022/day/8
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, trees):
        rows = len(trees)
        cols = len(trees[0])
        visible_trees = set()
        for row in range(rows):
            # Look from the left
            tallest_tree_from_left = -1
            for col in range(cols):
                tree = int(trees[row][col])
                if tree > tallest_tree_from_left:
                    visible_trees.add((row, col))
                    tallest_tree_from_left = tree
            # Look from the right
            tallest_tree_from_right = -1
            for col in reversed(range(cols)):
                tree = int(trees[row][col])
                if tree > tallest_tree_from_right:
                    visible_trees.add((row, col))
                    tallest_tree_from_right = tree
        for col in range(cols):
            # Look from the top
            tallest_tree_from_top = -1
            for row in range(rows):
                tree = int(trees[row][col])
                if tree > tallest_tree_from_top:
                    visible_trees.add((row, col))
                    tallest_tree_from_top = tree
            # Look from the bottom
            tallest_tree_from_bottom = -1
            for row in reversed(range(rows)):
                tree = int(trees[row][col])
                if tree > tallest_tree_from_bottom:
                    visible_trees.add((row, col))
                    tallest_tree_from_bottom = tree
        result = len(visible_trees)
        return result
    
    def solve2(self, trees):
        rows = len(trees)
        cols = len(trees[0])
        scenic_scores = {}
        for row in range(rows):
            for col in range(cols):
                viewing_height = int(trees[row][col])
                # Look left
                left_tree_count = 0
                for c in reversed(range(col)):
                    left_tree_count += 1
                    tree = int(trees[row][c])
                    if tree >= viewing_height:
                        break
                # Look right
                right_tree_count = 0
                for c in range(col + 1, cols):
                    right_tree_count += 1
                    tree = int(trees[row][c])
                    if tree >= viewing_height:
                        break
                # Look up
                up_tree_count = 0
                for r in reversed(range(row)):
                    up_tree_count += 1
                    tree = int(trees[r][col])
                    if tree >= viewing_height:
                        break
                # Look down
                down_tree_count = 0
                for r in range(row + 1, rows):
                    down_tree_count += 1
                    tree = int(trees[r][col])
                    if tree >= viewing_height:
                        break
                scenic_score = left_tree_count * right_tree_count * up_tree_count * down_tree_count
                scenic_scores[(row, col)] = scenic_score
        result = max(scenic_scores.values())
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

class Day07: # No Space Left On Device
    '''
    https://adventofcode.com/2022/day/7
    '''
    def get_sizes(self, raw_input_lines: List[str]):
        working_dir = '/'
        lists = {}
        sizes = {}
        cursor = 0
        while cursor < len(raw_input_lines):
            if raw_input_lines[cursor][0] != '$':
                raise Exception('Invalid row. Row must start with $')
            command = tuple(raw_input_lines[cursor][2:].split(' '))
            if command[0] == 'cd':
                if command[1] == '/':
                    working_dir = '/'
                elif command[1] == '..':
                    if working_dir != '/':
                        working_dir = '/'.join(working_dir.split('/')[:-2]) + '/'
                else:
                    working_dir += command[1] + '/'
                cursor += 1
            elif command[0] == 'ls':
                cursor += 1
                dirs = []
                files = {}
                while (
                    cursor < len(raw_input_lines) and
                    raw_input_lines[cursor][0] != '$'
                ):
                    parts = raw_input_lines[cursor].split(' ')
                    if parts[0] == 'dir':
                        dirs.append(parts[1])
                    else:
                        file = parts[1]
                        filesize = int(parts[0])
                        files[file] = int(parts[0])
                        sizes[working_dir + file] = filesize
                    cursor += 1
                lists[working_dir] = (dirs, files)
        def sizeof(path: str):
            if path not in sizes:
                total_size = 0
                (dirs, files) = lists[path]
                for file in files:
                    total_size += sizes[path + file]
                for dir in dirs:
                    total_size += sizeof(path + dir + '/')
                sizes[path] = total_size
            return sizes[path]
        for path in lists:
            sizes[path] = sizeof(path)
        result = sizes
        return result
    
    def solve(self, sizes):
        result = sum(
            size for
            path, size in sizes.items() if
            path[-1] == '/' and size <= 100_000
        )
        return result
    
    def solve2(self, sizes):
        total_disk_space = 70_000_000
        available_disk_space = total_disk_space - sizes['/']
        disk_space_needed = max(0, 30_000_000 - available_disk_space)
        result = min(
            size for
            _, size in sizes.items() if
            size >= disk_space_needed
        )
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        sizes = self.get_sizes(raw_input_lines)
        solutions = (
            self.solve(sizes),
            self.solve2(sizes),
            )
        result = solutions
        return result

class Day06: # Tuning Trouble
    '''
    https://adventofcode.com/2022/day/6
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = raw_input_lines[0]
        return result
    
    def solve(self, signal):
        stream = iter(signal)
        result = ''
        chars = collections.deque()
        while len(chars) < 4:
            chars.append(next(stream))
        start = 4
        while len(set(chars)) < 4:
            chars.popleft()
            chars.append(next(stream))
            start += 1
        result = start
        return result
    
    def solve2(self, signal):
        stream = iter(signal)
        result = ''
        chars = collections.deque()
        while len(chars) < 14:
            chars.append(next(stream))
        start = 14
        while len(set(chars)) < 14:
            chars.popleft()
            chars.append(next(stream))
            start += 1
        result = start
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        signal = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(signal),
            self.solve2(signal),
            )
        result = solutions
        return result

class Day05: # Supply Stacks
    '''
    https://adventofcode.com/2022/day/5
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        stack_height = 0
        while '[' in raw_input_lines[stack_height]:
            stack_height += 1
        stack_count = max(map(int, raw_input_lines[stack_height].split()))
        stacks = []
        for stack_id in range(stack_count):
            col = 1 + 4 * stack_id
            stacks.append('')
            for row in reversed(range(stack_height)):
                cell = raw_input_lines[row][col]
                if cell != ' ':
                    stacks[-1] += raw_input_lines[row][col]
        instructions = []
        for raw_input_line in raw_input_lines[stack_height + 2:]:
            parts = raw_input_line.split(' ')
            crate_count = int(parts[1])
            source_column = int(parts[3]) - 1
            target_column = int(parts[5]) - 1
            instruction = (crate_count, source_column, target_column)
            instructions.append(instruction)
        result = (stacks, instructions)
        return result
    
    def solve(self, stacks, instructions):
        for (crate_count, source_column, target_column) in instructions:
            for _ in range(crate_count):
                crate = stacks[source_column][-1]
                stacks[source_column] = stacks[source_column][:-1]
                stacks[target_column] += crate
        result = ''
        for stack in stacks:
            result += stack[-1]
        return result
    
    def solve2(self, stacks, instructions):
        for (crate_count, source_column, target_column) in instructions:
            crates = stacks[source_column][-crate_count:]
            stacks[source_column] = stacks[source_column][:-crate_count]
            stacks[target_column] += crates
        result = ''
        for stack in stacks:
            result += stack[-1]
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        stacks, instructions = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(stacks[:], instructions),
            self.solve2(stacks[:], instructions),
            )
        result = solutions
        return result

class Day04: # Camp Cleanup
    '''
    https://adventofcode.com/2022/day/4
    '''
    def get_assignments(self, raw_input_lines: List[str]):
        assignments = []
        for raw_input_line in raw_input_lines:
            a, b = raw_input_line.split(',')
            a1, a2 = tuple(map(int, a.split('-')))
            b1, b2 = tuple(map(int, b.split('-')))
            assignments.append(((a1, a2), (b1, b2)))
        result = assignments
        return result
    
    def solve(self, assignments):
        containment_count = 0
        for (a1, a2), (b1, b2) in assignments:
            if (
                (a1 <= b1 and a2 >= b2) or
                (b1 <= a1 and b2 >= a2)
            ):
                containment_count += 1
        result = containment_count
        return result
    
    def solve2(self, assignments):
        overlap_count = 0
        for (a1, a2), (b1, b2) in assignments:
            if len(set(range(a1, a2 + 1)) & set(range(b1, b2 + 1))) > 0:
                overlap_count += 1
        result = overlap_count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        assignments = self.get_assignments(raw_input_lines)
        solutions = (
            self.solve(assignments),
            self.solve2(assignments),
            )
        result = solutions
        return result

class Day03: # Rucksack Reorganization
    '''
    https://adventofcode.com/2022/day/3
    '''
    def get_rucksacks(self, raw_input_lines: List[str]):
        rucksacks = []
        for raw_input_line in raw_input_lines:
            compartment_size = len(raw_input_line) // 2
            compartment_a = raw_input_line[:compartment_size]
            compartment_b = raw_input_line[compartment_size:]
            rucksacks.append((compartment_a, compartment_b))
        result = rucksacks
        return result
    
    def solve(self, rucksacks):
        priorities = []
        for (compartment_a, compartment_b) in rucksacks:
            common_items = set(compartment_a) & set(compartment_b)
            for item in common_items:
                priority = 0
                if item in 'abcdefghijklmnopqrstuvwxyz':
                    priority = 1 + ord(item) - ord('a')
                elif item in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    priority = 27 + ord(item) - ord('A')
                priorities.append(priority)
        result = sum(priorities)
        return result
    
    def solve2(self, rucksacks):
        priorities = []
        for index in range(0, len(rucksacks), 3):
            rucksack_a = rucksacks[index][0] + rucksacks[index][1]
            rucksack_b = rucksacks[index + 1][0] + rucksacks[index + 1][1]
            rucksack_c = rucksacks[index + 2][0] + rucksacks[index + 2][1]
            common_items = set(rucksack_a) & set(rucksack_b) & set(rucksack_c)
            for item in common_items:
                priority = 0
                if item in 'abcdefghijklmnopqrstuvwxyz':
                    priority = 1 + ord(item) - ord('a')
                elif item in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    priority = 27 + ord(item) - ord('A')
                priorities.append(priority)
        result = sum(priorities)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        rucksacks = self.get_rucksacks(raw_input_lines)
        solutions = (
            self.solve(rucksacks),
            self.solve2(rucksacks),
            )
        result = solutions
        return result

class Day02: # Rock Paper Scissors
    '''
    https://adventofcode.com/2022/day/2
    '''
    scoring_values = {
        'Rock': 1,
        'Paper': 2,
        'Scissors': 3,
        'Loss': 0,
        'Draw': 3,
        'Win': 6,
    }

    messages = {
        'A': 'Rock',
        'B': 'Paper',
        'C': 'Scissors',
    }

    responses = {
        'X': 'Rock',
        'Y': 'Paper',
        'Z': 'Scissors',
        ('Rock', 'Loss'): 'Scissors',
        ('Paper', 'Loss'): 'Rock',
        ('Scissors', 'Loss'): 'Paper',
        ('Rock', 'Draw'): 'Rock',
        ('Paper', 'Draw'): 'Paper',
        ('Scissors', 'Draw'): 'Scissors',
        ('Rock', 'Win'): 'Paper',
        ('Paper', 'Win'): 'Scissors',
        ('Scissors', 'Win'): 'Rock',
    }

    desired_outcomes = {
        'X': 'Loss',
        'Y': 'Draw',
        'Z': 'Win',
    }

    outcomes = {
        ('Rock', 'Rock'): 'Draw',
        ('Rock', 'Paper'): 'Win',
        ('Rock', 'Scissors'): 'Loss',
        ('Paper', 'Rock'): 'Loss',
        ('Paper', 'Paper'): 'Draw',
        ('Paper', 'Scissors'): 'Win',
        ('Scissors', 'Rock'): 'Win',
        ('Scissors', 'Paper'): 'Loss',
        ('Scissors', 'Scissors'): 'Draw',
    }

    def get_strategy_guide(self, raw_input_lines: List[str]):
        strategy_guide = []
        for raw_input_line in raw_input_lines:
            message, response = raw_input_line.split(' ')
            strategy_guide.append(
                (
                    self.messages[message],
                    self.responses[response],
                )
            )
        result = strategy_guide
        return result

    def get_strategy_guide_v2(self, raw_input_lines: List[str]):
        strategy_guide_v2 = []
        for raw_input_line in raw_input_lines:
            message, desired_outcome = raw_input_line.split(' ')
            strategy_guide_v2.append(
                (
                    self.messages[message],
                    self.desired_outcomes[desired_outcome],
                )
            )
        result = strategy_guide_v2
        return result
    
    def solve(self, strategy_guide):
        total_score = 0
        for (message, response) in strategy_guide:
            total_score += self.scoring_values[response]
            outcome = self.outcomes[(message, response)]
            total_score += self.scoring_values[outcome]
        result = total_score
        return result
    
    def solve2(self, strategy_guide_v2):
        total_score = 0
        for (message, desired_outcome) in strategy_guide_v2:
            outcome = desired_outcome
            response = self.responses[(message, outcome)]
            total_score += self.scoring_values[response]
            total_score += self.scoring_values[outcome]
        result = total_score
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        strategy_guide = self.get_strategy_guide(raw_input_lines)
        strategy_guide_v2 = self.get_strategy_guide_v2(raw_input_lines)
        solutions = (
            self.solve(strategy_guide),
            self.solve2(strategy_guide_v2),
            )
        result = solutions
        return result

class Day01: # Calorie Counting
    '''
    https://adventofcode.com/2022/day/1
    '''
    def get_elves(self, raw_input_lines: List[str]):
        elves = [[]]
        for raw_input_line in raw_input_lines:
            if len(raw_input_line) < 1:
                elves.append([])
            else:
                elves[-1].append(int(raw_input_line))
        result = elves
        return result
    
    def solve(self, elves):
        result = max(sum(calories) for calories in elves)
        return result
    
    def solve2(self, elves):
        heap = []
        for calories in elves:
            heapq.heappush(heap, -1 * sum(calories))
        top_calories = []
        top_calories.append(-1 * heapq.heappop(heap))
        top_calories.append(-1 * heapq.heappop(heap))
        top_calories.append(-1 * heapq.heappop(heap))
        result = sum(top_calories)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        elves = self.get_elves(raw_input_lines)
        solutions = (
            self.solve(elves),
            self.solve2(elves),
            )
        result = solutions
        return result

if __name__ == '__main__':
    '''
    Usage
    python AdventOfCode2022.py 11 < inputs/2022day11.in
    '''
    solvers = {
        1: (Day01, 'Calorie Counting'),
        2: (Day02, 'Rock Paper Scissors'),
        3: (Day03, 'Rucksack Reorganization'),
        4: (Day04, 'Camp Cleanup'),
        5: (Day05, 'Supply Stacks'),
        6: (Day06, 'Tuning Trouble'),
        7: (Day07, 'No Space Left On Device'),
        8: (Day08, 'Treetop Tree House'),
        9: (Day09, 'Rope Bridge'),
       10: (Day10, 'Cathode-Ray Tube'),
       11: (Day11, 'Monkey in the Middle'),
    #    12: (Day12, 'Day12'),
    #    13: (Day13, 'Day13'),
    #    14: (Day14, 'Day14'),
    #    15: (Day15, 'Day15'),
    #    16: (Day16, 'Day16'),
    #    17: (Day17, 'Day17'),
    #    18: (Day18, 'Day18'),
    #    19: (Day19, 'Day19'),
    #    20: (Day20, 'Day20'),
    #    21: (Day21, 'Day21'),
    #    22: (Day22, 'Day22'),
    #    23: (Day23, 'Day23'),
    #    24: (Day24, 'Day24'),
    #    25: (Day25, 'Day25'),
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
