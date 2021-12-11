'''
Created on 2021-11-29

@author: Sestren
'''
import argparse
import collections
import copy
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
    https://adventofcode.com/2021/day/?
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

class Day11: # Dumbo Octopus
    '''
    https://adventofcode.com/2021/day/11
    '''
    def get_octopuses(self, raw_input_lines: List[str]):
        octopuses = {}
        for row, raw_input_line in enumerate(raw_input_lines):
            for col, cell in enumerate(raw_input_line):
                octopuses[(row, col)] = int(cell)
        result = octopuses
        return result
    
    class Simulation:
        def __init__(self, octopuses):
            self.octopuses = copy.deepcopy(octopuses)
            self.flashes = set()

        def step(self):
            self.flashes = set()
            # Increase energies by 1
            for octopus in self.octopuses.keys():
                self.octopuses[octopus] += 1
                if self.octopuses[octopus] > 9:
                    self.flashes.add(octopus)
            # Propogate flashing of octopuses at 10+ energy
            flashes_left = set(self.flashes)
            while len(flashes_left) > 0:
                new_flashes = set()
                for (row, col) in flashes_left:
                    for (nrow, ncol) in (
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
                            (nrow, ncol) in self.octopuses and
                            (nrow, ncol) not in self.flashes and
                            (nrow, ncol) not in new_flashes
                        ):
                            self.octopuses[(nrow, ncol)] += 1
                            if self.octopuses[(nrow, ncol)] > 9:
                                new_flashes.add((nrow, ncol))
                self.flashes.update(new_flashes)
                flashes_left = new_flashes
            # Reset flashed octopuses to 0 energy
            for flash in self.flashes:
                self.octopuses[flash] = 0
    
    def solve(self, octopuses):
        flash_count = 0
        sim = self.Simulation(octopuses)
        for _ in range(100):
            sim.step()
            flash_count += len(sim.flashes)
        result = flash_count
        return result
    
    def solve2(self, octopuses):
        target_turn_id = None
        turn_id = 1
        sim = self.Simulation(octopuses)
        while True:
            sim.step()
            if len(sim.flashes) == len(octopuses):
                target_turn_id = turn_id
                break
            turn_id += 1
        result = target_turn_id
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        octopuses = self.get_octopuses(raw_input_lines)
        solutions = (
            self.solve(copy.deepcopy(octopuses)),
            self.solve2(copy.deepcopy(octopuses)),
            )
        result = solutions
        return result

class Day10: # Syntax Scoring
    '''
    https://adventofcode.com/2021/day/10
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, lines):
        values = {
            ')': 3,
            ']': 57,
            '}': 1197,
            '>': 25137,
        }
        pairs = {
            ')': '(',
            ']': '[',
            '}': '{',
            '>': '<',
        }
        score = 0
        for line in lines:
            stack = []
            for char in line:
                if char in '([{<':
                    stack.append(char)
                elif char in ')]}>':
                    if len(stack) < 1 or stack[-1] != pairs[char]:
                        score += values[char]
                        break
                    stack.pop()
        result = score
        return result
    
    def solve2(self, lines):
        values = {
            ')': 1,
            ']': 2,
            '}': 3,
            '>': 4,
        }
        opens = {
            ')': '(',
            ']': '[',
            '}': '{',
            '>': '<',
        }
        closes = {
            '(': ')',
            '[': ']',
            '{': '}',
            '<': '>',
        }
        scores = []
        for line in lines:
            corrupt_ind = False
            stack = []
            for char in line:
                if char in '([{<':
                    stack.append(char)
                elif char in ')]}>':
                    if len(stack) < 1 or stack[-1] != opens[char]:
                        corrupt_ind = True
                        break
                    stack.pop()
            if corrupt_ind:
                continue
            print(''.join(stack))
            score = 0
            while len(stack) > 0:
                char = closes[stack.pop()]
                score = 5 * score + values[char]
            scores.append(score)
        result = sorted(scores)[len(scores) // 2]
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        lines = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(lines),
            self.solve2(lines),
            )
        result = solutions
        return result

class Day09: # Smoke Basin
    '''
    https://adventofcode.com/2021/day/9
    '''
    def get_height_map(self, raw_input_lines: List[str]):
        height_map = []
        for raw_input_line in raw_input_lines:
            row_data = []
            for cell in raw_input_line:
                row_data.append(int(cell))
            height_map.append(row_data)
        result = height_map
        return result
    
    def solve(self, height_map):
        rows = len(height_map)
        cols = len(height_map[0])
        low_point_risk_levels = []
        for row in range(rows):
            for col in range(cols):
                low_point = height_map[row][col]
                for (nrow, ncol) in (
                    (row - 1, col    ),
                    (row + 1, col    ),
                    (row    , col - 1),
                    (row    , col + 1),
                ):
                    if not (0 <= nrow < rows and 0 <= ncol < cols):
                        continue
                    if height_map[nrow][ncol] <= low_point:
                        break
                else:
                    low_point_risk_levels.append(1 + low_point)
        result = sum(low_point_risk_levels)
        return result
    
    def solve2(self, height_map):
        rows = len(height_map)
        cols = len(height_map[0])
        unvisited = set()
        for row in range(rows):
            for col in range(cols):
                unvisited.add((row, col))
        basins = []
        while len(unvisited) > 0:
            row, col = unvisited.pop()
            if height_map[row][col] == 9:
                continue
            basin = 0
            work = [(row, col)]
            while len(work) > 0:
                row, col = work.pop()
                basin += 1
                for (nrow, ncol) in (
                    (row - 1, col    ),
                    (row + 1, col    ),
                    (row    , col - 1),
                    (row    , col + 1),
                ):
                    if (
                        0 <= nrow < rows and
                        0 <= ncol < cols and
                        (nrow, ncol) in unvisited and
                        height_map[nrow][ncol] != 9
                    ):
                        unvisited.remove((nrow, ncol))
                        work.append((nrow, ncol))
            basins.append(basin)
        basins.sort()
        result = basins[-1] * basins[-2] * basins[-3]
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        height_map = self.get_height_map(raw_input_lines)
        solutions = (
            self.solve(height_map),
            self.solve2(height_map),
            )
        result = solutions
        return result

class Day08: # Seven Segment Search
    '''
    https://adventofcode.com/2021/day/8

    2 of 7 segments (1):    1
    3 of 7 segments (1):                7
    4 of 7 segments (1):          4
    5 of 7 segments (3):      2 3   5
    6 of 7 segments (3):  0           6     9
    7 of 7 segments (1):  8

    Has canonical A:  0   2 3   5 6 7 8 9
    Has canonical B:  0       4 5 6   8 9
    Has canonical C:  0 1 2 3 4     7 8 9
    Has canonical D:      2 3 4 5 6   8 9
    Has canonical E:  0   2       6   8
    Has canonical F:  0 1   3 4 5 6 7 8 9
    Has canonical G:  0   2 3   5 6   8 9
    '''
    def get_entries(self, raw_input_lines: List[str]):
        entries = []
        for raw_input_line in raw_input_lines:
            parts = raw_input_line.split(' ')
            unique_signal_patterns = parts[:10]
            output_value = parts[-4:]
            entry = (unique_signal_patterns, output_value)
            entries.append(entry)
        result = entries
        return result
    
    def solve(self, entries):
        easy_digits = []
        for _, output_value in entries:
            for element in output_value:
                if len(element) in (2, 3, 4, 7):
                    easy_digits.append(element)
        result = len(easy_digits)
        return result
    
    def decode_output(self, patterns, output) -> int:
        digits_in_segment_count = {
            2: {'1', },
            3: {'7', },
            4: {'4', },
            5: {'2', '3', '5', },
            6: {'0', '6', '9', },
            7: {'8', },
        }
        # Let S be the non-canonical segments found in a given string
        # A string can be one of the following:
        #   a segment ('a' through 'f')
        #   a digit ('0' through '9')
        #   the intersection of multiple digits (e.g., '025')
        S = {}
        for segment_count, digits in digits_in_segment_count.items():
            code = ''.join(sorted(digits))
            if code not in S:
                # Start with all possible segments
                S[code] = set('abcdefg')
            for pattern in patterns:
                # Take the intersection of each pattern with the same length
                if len(pattern) == segment_count:
                    S[code] &= set(pattern)
        S['3'] = S['235'] | S['1']
        S['d'] = S['235'] & S['4']
        S['0'] = S['8'] - S['d']
        S['f'] = S['069'] & S['1']
        S['c'] = S['1'] - S['f']
        S['6'] = S['8'] - S['c']
        S['9'] = S['069'] | S['4']
        S['5'] = S['9'] - S['c']
        S['e'] = S['8'] - S['9']
        S['2'] = S['235'] | S['c'] | S['e']
        # Let D be the inverse of S for single digit lookups
        # i.e., the keys will be the segment keys that produce a digit value
        D = {}
        for digits, segments in S.items():
            if len(digits) == 1:
                key = ''.join(sorted(segments))
                D[key] = next(iter(digits))
        decoded_output = 0
        for segment_code in output:
            key = ''.join(sorted(set(segment_code)))
            decoded_output = 10 * decoded_output + int(D[key])
        result = decoded_output
        return result
    
    def solve2(self, entries):
        decoded_outputs = []
        for patterns, output in entries:
            decoded_output = self.decode_output(patterns, output)
            decoded_outputs.append(decoded_output)
        result = sum(decoded_outputs)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        entries = self.get_entries(raw_input_lines)
        solutions = (
            self.solve(entries),
            self.solve2(entries),
            )
        result = solutions
        return result

class Day07: # The Treachery of Whales
    '''
    https://adventofcode.com/2021/day/7
    '''
    def get_crabs(self, raw_input_lines: List[str]):
        result = list(map(int, raw_input_lines[0].split(',')))
        return result
    
    def solve(self, crabs):
        min_col = min(crabs)
        max_col = max(crabs)
        min_cost = float('inf')
        for target_col in range(min_col, max_col + 1):
            cost = 0
            for col in crabs:
                cost += abs(col - target_col)
            min_cost = min(min_cost, cost)
        result = min_cost
        return result
    
    def solve2(self, crabs):
        min_col = min(crabs)
        max_col = max(crabs)
        triangle_sums = [0]
        for num in range(1, abs(max_col - min_col + 1)):
            triangle_sums.append(triangle_sums[-1] + num)
        min_cost = float('inf')
        for target_col in range(min_col, max_col + 1):
            cost = 0
            for col in crabs:
                steps = abs(col - target_col)
                cost += triangle_sums[steps]
            min_cost = min(min_cost, cost)
        result = min_cost
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        crabs = self.get_crabs(raw_input_lines)
        solutions = (
            self.solve(crabs),
            self.solve2(crabs),
            )
        result = solutions
        return result

class Day06: # Lanternfish
    '''
    https://adventofcode.com/2021/day/6
    '''
    def get_starting_fish(self, raw_input_lines: List[str]):
        starting_fish = list(map(int, raw_input_lines[0].split(',')))
        result = starting_fish
        return result
    
    def solve(self, starting_fish, day_count):
        fish_timers = collections.defaultdict(int)
        for fish in starting_fish:
            fish_timers[fish] += 1
        for _ in range(day_count):
            next_fish_timers = collections.defaultdict(int)
            for timer, count in fish_timers.items():
                if timer == 0:
                    next_fish_timers[8] += count
                    next_fish_timers[6] += count
                else:
                    next_fish_timers[timer - 1] += count
            fish_timers = next_fish_timers
        result = sum(fish_timers.values())
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        starting_fish = self.get_starting_fish(raw_input_lines)
        solutions = (
            self.solve(starting_fish, 80),
            self.solve(starting_fish, 256),
            )
        result = solutions
        return result

class Day05: # Hydrothermal Venture
    '''
    https://adventofcode.com/2021/day/5
    '''
    def get_vents(self, raw_input_lines: List[str]):
        vents = []
        for raw_input_line in raw_input_lines:
            a, b = raw_input_line.split(' -> ')
            start = tuple(reversed(tuple(map(int, a.split(',')))))
            end = tuple(reversed(tuple(map(int, b.split(',')))))
            vents.append(tuple(sorted((start, end))))
        result = vents
        return result
    
    def solve(self, vents):
        orthogonal_vents = []
        for start, end in vents:
            if start[0] == end[0] or start[1] == end[1]:
                orthogonal_vents.append((start, end))
        points = collections.defaultdict(int)
        for start, end in orthogonal_vents:
            for row in range(start[0], end[0] + 1):
                for col in range(start[1], end[1] + 1):
                    points[(row, col)] += 1
        result = sum(1 for count in points.values() if count > 1)
        return result
    
    def solve2(self, vents):
        points = collections.defaultdict(int)
        for start, end in vents:
            row_delta = 0
            if end[0] > start[0]:
                row_delta = 1
            elif end[0] < start[0]:
                row_delta = -1
            col_delta = 0
            if end[1] > start[1]:
                col_delta = 1
            elif end[1] < start[1]:
                col_delta = -1
            distance = 1 + max(abs(end[0] - start[0]), abs(end[1] - start[1]))
            for step in range(distance):
                row = start[0] + step * row_delta
                col = start[1] + step * col_delta
                points[(row, col)] += 1
        result = sum(1 for count in points.values() if count > 1)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        vents = self.get_vents(raw_input_lines)
        solutions = (
            self.solve(vents),
            self.solve2(vents),
            )
        result = solutions
        return result

class Day04: # Giant Squid
    '''
    https://adventofcode.com/2021/day/4
    '''
    def get_bingo_subsystem(self, raw_input_lines: List[str]):
        numbers = []
        for num_str in raw_input_lines[0].split(','):
            number = int(num_str)
            numbers.append(number)
        boards = []
        board = []
        for raw_input_line in raw_input_lines[1:]:
            if len(raw_input_line) == 0:
                if len(board) > 0:
                    boards.append(board)
                board = []
            else:
                board_row = list(map(int, raw_input_line.split()))
                board.append(board_row)
        if len(board) > 0:
            boards.append(board)
        result = (boards, numbers)
        return result
    
    def solve(self, boards, numbers):
        rows = len(boards[0])
        cols = len(boards[0][0])
        called_numbers = set()
        winning_board_id = -1
        latest_number = -1
        for latest_number in numbers:
            called_numbers.add(latest_number)
            for board_id in range(len(boards)):
                for row in range(rows):
                    nums = set(boards[board_id][row])
                    if len(nums & called_numbers) == rows:
                        winning_board_id = board_id
                        break
                for col in range(cols):
                    nums = set()
                    for row in range(rows):
                        nums.add(boards[board_id][row][col])
                    if len(nums & called_numbers) == rows:
                        winning_board_id = board_id
                        break
            if winning_board_id >= 0:
                break
        winning_board_score = 0
        for row in range(rows):
            for col in range(cols):
                if boards[winning_board_id][row][col] not in called_numbers:
                    winning_board_score += boards[winning_board_id][row][col]
        result = latest_number * winning_board_score
        return result
    
    def solve2(self, boards, numbers):
        rows = len(boards[0])
        cols = len(boards[0][0])
        called_numbers = set()
        winning_board_ids = set()
        latest_win = {
            'board_id': -1,
            'round_id': -1,
            'number': -1,
        }
        for round_id, number in enumerate(numbers, start=1):
            called_numbers.add(number)
            for board_id in range(len(boards)):
                if board_id in winning_board_ids:
                    continue
                for row in range(rows):
                    nums = set(boards[board_id][row])
                    if len(nums & called_numbers) == rows:
                        latest_win = {
                            'board_id': board_id,
                            'round_id': round_id,
                            'number': number,
                        }
                        winning_board_ids.add(board_id)
                for col in range(cols):
                    nums = set()
                    for row in range(rows):
                        nums.add(boards[board_id][row][col])
                    if len(nums & called_numbers) == rows:
                        latest_win = {
                            'board_id': board_id,
                            'round_id': round_id,
                            'number': number,
                        }
                        winning_board_ids.add(board_id)
        called_numbers = set(numbers[:latest_win['round_id']])
        latest_winning_board_score = 0
        for row in range(rows):
            for col in range(cols):
                number = boards[latest_win['board_id']][row][col]
                if number not in called_numbers:
                    latest_winning_board_score += number
        result = latest_win['number'] * latest_winning_board_score
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        boards, numbers = self.get_bingo_subsystem(raw_input_lines)
        solutions = (
            self.solve(boards, numbers),
            self.solve2(boards, numbers),
            )
        result = solutions
        return result

class Day03: # Binary Diagnostic
    '''
    https://adventofcode.com/2021/day/3
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, parsed_input):
        N = len(parsed_input)
        counts = [0] * len(parsed_input[0])
        for line in parsed_input:
            for i, char in enumerate(line):
                if char == '1':
                    counts[i] += 1
        gamma_chars = []
        epsilon_chars = []
        for i, count in enumerate(counts):
            if count >= N // 2:
                gamma_chars.append('1')
                epsilon_chars.append('0')
            else:
                gamma_chars.append('0')
                epsilon_chars.append('1')
        gamma = int(''.join(gamma_chars), 2)
        epsilon = int(''.join(epsilon_chars), 2)
        power = gamma * epsilon
        result = power
        return result
    
    def solve2(self, parsed_input):
        N = len(parsed_input[0])
        nums = sorted(parsed_input)
        # Calculate oxygen generator rating
        # majority per bit, 1s win in ties
        left = 0
        right = len(nums)
        for i in range(N):
            counts = [0, 0]
            for num in nums[left:right]:
                digit = int(num[i])
                counts[digit] += 1
            assert sum(counts) == right - left
            if counts[1] >= counts[0]:
                left += counts[0]
            else:
                right -= counts[1]
        oxy = int(nums[left], 2)
        assert left == right - 1
        # Calculate C02 scrubber rating
        # minority per bit, 0s win in ties
        left = 0
        right = len(nums)
        for i in range(N):
            if left == right - 1:
                break
            counts = [0, 0]
            for num in nums[left:right]:
                digit = int(num[i])
                counts[digit] += 1
            assert sum(counts) == right - left
            if counts[0] <= counts[1]:
                right -= counts[1]
            else:
                left += counts[0]
        co2 = int(nums[left], 2)
        assert left == right - 1
        life_support = oxy * co2
        result = life_support
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

class Day02: # Dive!
    '''
    https://adventofcode.com/2021/day/2
    '''
    def get_commands(self, raw_input_lines: List[str]):
        commands = []
        for raw_input_line in raw_input_lines:
            command, raw_amount = raw_input_line.split(' ')
            amount = int(raw_amount)
            commands.append((command, amount))
        result = commands
        return result
    
    def solve(self, commands):
        x_pos = 0
        depth = 0
        for command, amount in commands:
            if command == 'forward':
                x_pos += amount
            elif command == 'down':
                depth += amount
            elif command == 'up':
                depth -= amount
        result = x_pos * depth
        return result
    
    def solve2(self, commands):
        x_pos = 0
        depth = 0
        aim = 0
        for command, amount in commands:
            if command == 'forward':
                x_pos += amount
                depth += aim * amount
            elif command == 'down':
                aim += amount
            elif command == 'up':
                aim -= amount
        result = x_pos * depth
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        commands = self.get_commands(raw_input_lines)
        solutions = (
            self.solve(commands),
            self.solve2(commands),
            )
        result = solutions
        return result

class Day01: # Sonar Sweep
    '''
    https://adventofcode.com/2021/day/1
    '''
    def get_sonar_sweeps(self, raw_input_lines: List[str]):
        sonar_sweeps = []
        for raw_input_line in raw_input_lines:
            sonar_sweeps.append(int(raw_input_line))
        result = sonar_sweeps
        return result
    
    def solve(self, sonar_sweeps):
        count = 0
        for i in range(1, len(sonar_sweeps)):
            if sonar_sweeps[i] > sonar_sweeps[i - 1]:
                count += 1
        result = count
        return result
    
    def solve2(self, sonar_sweeps):
        count = 0
        a, b, c = sonar_sweeps[0], sonar_sweeps[1], sonar_sweeps[2]
        for i in range(3, len(sonar_sweeps)):
            d = sonar_sweeps[i]
            if d > a:
                count += 1
            a, b, c = b, c, d
        result = count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        sonar_sweeps = self.get_sonar_sweeps(raw_input_lines)
        solutions = (
            self.solve(sonar_sweeps),
            self.solve2(sonar_sweeps),
            )
        result = solutions
        return result

if __name__ == '__main__':
    '''
    Usage
    python AdventOfCode2021.py 1 < inputs/2021day01.in
    '''
    solvers = {
        1: (Day01, 'Sonar Sweep'),
        2: (Day02, 'Dive!'),
        3: (Day03, 'Binary Diagnostic'),
        4: (Day04, 'Giant Squid'),
        5: (Day05, 'Hydrothermal Venture'),
        6: (Day06, 'Lanternfish'),
        7: (Day07, 'The Treachery of Whales'),
        8: (Day08, 'Seven Segment Search'),
        9: (Day09, 'Smoke Basin'),
       10: (Day10, 'Syntax Scoring'),
       11: (Day11, 'XXX'),
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
