'''
Created on Dec 30, 2020

@author: Sestren
'''
import argparse
import collections
import copy
import datetime
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
    Template
    https://adventofcode.com/2018/day/?
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

class Day09: # Marble Mania
    '''
    Marble Mania
    https://adventofcode.com/2018/day/9
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        parts = raw_input_lines[0].split(' ')
        player_count = int(parts[0])
        marble_count = int(parts[6])
        result = (player_count, marble_count)
        return result
    
    def solve(self, player_count, marble_count):
        scores = [0] * player_count
        marbles = collections.deque()
        marbles.append(0)
        player_id = 0
        for marble in range(1, marble_count + 1):
            if marble % 23 == 0:
                scores[player_id] += marble
                marbles.rotate(7)
                captured = marbles.popleft()
                scores[player_id] += captured
            else:
                marbles.rotate(-2)
                marbles.appendleft(marble)
            player_id = (player_id + 1) % player_count
        result = max(scores)
        return result
    
    def solve2(self, player_count, marble_count):
        result = self.solve(player_count, 100 * marble_count)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        player_count, marble_count = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(player_count, marble_count),
            self.solve2(player_count, marble_count),
            )
        result = solutions
        return result

class Day08: # Memory Maneuver
    '''
    Memory Maneuver
    https://adventofcode.com/2018/day/8
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = collections.deque(map(int, raw_input_lines[0].split(' ')))
        return result
    
    def solve(self, nums):
        child_node_count = nums.popleft()
        metadata_entry_count = nums.popleft()
        metadata_entry_sum = 0
        for _ in range(child_node_count):
            metadata_entry_sum += self.solve(nums)
        for _ in range(metadata_entry_count):
            metadata_entry_sum += nums.popleft()
        result = metadata_entry_sum
        return result
    
    def solve2(self, nums):
        child_node_count = nums.popleft()
        metadata_entry_count = nums.popleft()
        values = [
            self.solve2(nums) for _ in
            range(child_node_count)
            ]
        metadata_entries = [
            nums.popleft() for _ in
            range(metadata_entry_count)
            ]
        total_value = 0
        if child_node_count == 0:
            total_value = sum(metadata_entries)
        else:
            total_value = sum(
                values[i - 1] for 
                i in metadata_entries if 
                i - 1 in range(child_node_count)
                )
        result = total_value
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        parsed_input = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(copy.deepcopy(parsed_input)),
            self.solve2(copy.deepcopy(parsed_input)),
            )
        result = solutions
        return result

class Day07: # The Sum of Its Parts
    '''
    The Sum of Its Parts
    https://adventofcode.com/2018/day/7
    '''
    def get_dependencies(self, raw_input_lines: List[str]):
        dependencies = {}
        for raw_input_line in raw_input_lines:
            a = raw_input_line[36]
            b = raw_input_line[5]
            if a not in dependencies:
                dependencies[a] = set()
            if b not in dependencies:
                dependencies[b] = set()
            dependencies[a].add(b)
        result = dependencies
        return result
    
    def solve(self, dependencies):
        finished_steps = set()
        instructions = []
        while len(instructions) < len(dependencies):
            next_step = min(
                step for step, required_steps in
                dependencies.items() if
                len(required_steps) == 0 and
                step not in finished_steps
            )
            for instruction in dependencies:
                if next_step in dependencies[instruction]:
                    dependencies[instruction].remove(next_step)
            instructions.append(next_step)
            finished_steps.add(next_step)
        result = ''.join(instructions)
        return result
    
    def solve2(self, dependencies, worker_count: int=5, min_cost: int=60):
        tasks = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        costs = {
            task: min_cost + tasks.index(task) + 1 for task in tasks
            }
        workers = [(0, None)] * worker_count
        finished_steps = set()
        time = 0
        instructions = []
        while len(finished_steps) < len(dependencies):
            for worker_id in range(len(workers)):
                if workers[worker_id][0] > time:
                    continue
                try:
                    next_step = next(iter(sorted(
                        step for step, required_steps in
                        dependencies.items() if
                        len(required_steps) == 0 and
                        step not in finished_steps and
                        step not in instructions
                    )))
                    instructions.append(next_step)
                    workers[worker_id] = (time + costs[next_step], next_step)
                except StopIteration:
                    pass
            time += 1
            for worker_id in range(len(workers)):
                work_end, step = workers[worker_id]
                if time >= work_end and step is not None:
                    finished_steps.add(step)
                    for instruction in dependencies:
                        if step in dependencies[instruction]:
                            dependencies[instruction].remove(step)
        result = time
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        dependencies = self.get_dependencies(raw_input_lines)
        solutions = (
            self.solve(copy.deepcopy(dependencies)),
            self.solve2(copy.deepcopy(dependencies)),
            )
        result = solutions
        return result

class Day06: # Chronal Coordinates
    '''
    Chronal Coordinates
    https://adventofcode.com/2018/day/6
    '''
    def get_coordinates(self, raw_input_lines: List[str]):
        coordinates = []
        for raw_input_line in raw_input_lines:
            coordinates.append(tuple(map(int, raw_input_line.split(', '))))
        result = coordinates
        return result
    
    def print_grid(self, bounds, grid, tied):
        (left, right, top, bottom) = bounds
        for y in range(top, bottom + 1):
            row_data = []
            for x in range(left, right + 1):
                cell = '.'
                if (x, y) in tied:
                    cell = '_'
                elif (x, y) in grid:
                    cell = str(grid[(x, y)][1])
                row_data.append(cell)
            print(''.join(row_data))

    def solve(self, coordinates):
        left = float('inf')
        right = float('-inf')
        top = float('inf')
        bottom = float('-inf')
        for x, y in coordinates:
            left = min(left, x)
            right = max(right, x)
            top = min(top, y)
            bottom = max(bottom, y)
        # cells that are closest to more than one coordinate are tied
        tied = set()
        grid = {}
        work = collections.deque()
        for i, coordinate in enumerate(coordinates):
            work.append((0, i, coordinate))
        # Use BFS to fill the grid
        while len(work) > 0:
            step_count = len(work)
            for _ in range(step_count):
                (distance, index, coordinate) = work.pop()
                if coordinate in grid:
                    other_distance = grid[coordinate][0]
                    other_index = grid[coordinate][1]
                    if index != other_index and distance == other_distance:
                        tied.add(coordinate)
                    continue
                grid[coordinate] = (distance, index)
                for (x, y) in (
                    (coordinate[0] + 1, coordinate[1]    ),
                    (coordinate[0] - 1, coordinate[1]    ),
                    (coordinate[0]    , coordinate[1] + 1),
                    (coordinate[0]    , coordinate[1] - 1),
                    ):
                    if (
                        x < left or
                        x > right or
                        y < top or
                        y > bottom
                    ):
                        continue
                    work.appendleft((distance + 1, index, (x, y)))
        infinite = set()
        area = collections.defaultdict(int)
        for (x, y), (distance, index) in grid.items():
            if (x, y) not in tied:
                area[index] += 1
                # cells along the border that aren't tied
                # are guaranteed to go on infinitely
                if (
                    x <= left or
                    x >= right or
                    y <= top or
                    y >= bottom
                    ):
                    infinite.add(index)
        for index in infinite:
            del area[index]
        # self.print_grid((left, right, top, bottom), grid, tied)
        result = max(area.values())
        return result
    
    def solve2(self, coordinates, max_distance: int=10_000):
        left = float('inf')
        right = float('-inf')
        top = float('inf')
        bottom = float('-inf')
        for x, y in coordinates:
            left = min(left, x)
            right = max(right, x)
            top = min(top, y)
            bottom = max(bottom, y)
        N = len(coordinates)
        # Calculate distances horizontally
        cols = collections.defaultdict(int)
        for col in range(
            left - (1 + max_distance // N),
            right + (1 + max_distance // N),
            ):
            for coordinate in coordinates:
                cols[col] += abs(coordinate[0] - col)
        # Calculate distances vertically
        rows = collections.defaultdict(int)
        for row in range(
            top - (1 + max_distance // N),
            bottom + (1 + max_distance // N),
            ):
            for coordinate in coordinates:
                rows[row] += abs(coordinate[1] - row)
        # Combine horizontal and vertical distances to find cells
        # within the target distance
        count = 0
        for row, row_distance in rows.items():
            for col, col_distance in cols.items():
                if row_distance + col_distance < max_distance:
                    count += 1
        result = count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        coordinates = self.get_coordinates(raw_input_lines)
        solutions = (
            self.solve(coordinates),
            self.solve2(coordinates, 10_000),
            )
        result = solutions
        return result

class Day05: # Alchemical Reduction
    '''
    Alchemical Reduction
    https://adventofcode.com/2018/day/5
    '''
    def get_polymer(self, raw_input_lines: List[str]):
        polymer = list(raw_input_lines[0])
        result = polymer
        return result
    
    def get_fully_reacted_polymer(self, polymer):
        reaction_ind = True
        prev_polymer = polymer
        while reaction_ind:
            reaction_ind = False
            polymer = prev_polymer[:]
            i = len(polymer) - 1
            while i > 0:
                if (
                    prev_polymer[i] != prev_polymer[i - 1] and
                    prev_polymer[i].upper() == prev_polymer[i - 1].upper()
                    ):
                    polymer.pop(i)
                    polymer.pop(i - 1)
                    i -= 1
                    reaction_ind = True
                i -= 1
            prev_polymer = polymer
        result = polymer
        return result
    
    def solve(self, polymer):
        fully_reacted_polymer = self.get_fully_reacted_polymer(polymer)
        result = len(fully_reacted_polymer)
        return result
    
    def solve2(self, initial_polymer):
        shortest_polymer = initial_polymer
        for char in 'abcdefghijklmnopqrstuvwxyz':
            polymer = []
            for unit in initial_polymer:
                if unit == char or unit.lower() == char:
                    continue
                else:
                    polymer.append(unit)
            fully_reacted_polymer = self.get_fully_reacted_polymer(polymer)
            if len(fully_reacted_polymer) < len(shortest_polymer):
                shortest_polymer = fully_reacted_polymer
        result = len(shortest_polymer)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        polymer = self.get_polymer(raw_input_lines)
        solutions = (
            self.solve(polymer),
            self.solve2(polymer),
            )
        result = solutions
        return result

class Day04: # Repose Record
    '''
    Repose Record
    https://adventofcode.com/2018/day/4
    '''
    def get_sleep_times(self, raw_input_lines: List[str]):
        raw_records = []
        for raw_input_line in raw_input_lines:
            (a, b) = raw_input_line.split('] ')
            datetime_str = a[1:]
            year = int(datetime_str[:4])
            month = int(datetime_str[5:7])
            day = int(datetime_str[8:10])
            hour = int(datetime_str[11:13])
            minute = int(datetime_str[14:16])
            ordinal_day = datetime.date(year, month, day).toordinal()
            if hour == 23:
                ordinal_day += 1
                minute = 0
            raw_records.append((ordinal_day, minute, b))
        raw_records.sort()
        sleep_times = set()
        guard_id = -1
        for i in range(len(raw_records) - 1):
            day0, minute0, event0 = raw_records[i]
            _, minute1, event1 = raw_records[i + 1]
            if '#' in event0:
                guard_id = int(event0.split(' ')[1][1:])
            elif event0 == 'falls asleep':
                start = minute0
                end = minute1
                if '#' in event1:
                    end = 60
                for minute_id in range(start, end):
                    sleep_times.add((day0, minute_id, guard_id))
        result = sleep_times
        return result
    
    def solve(self, sleep_times):
        guards = collections.defaultdict(int)
        for _, minute_id, guard_id in sorted(sleep_times):
            guards[guard_id] += 1
        best_guard_id = max(guards, key=guards.get)
        minutes = collections.defaultdict(int)
        for _, minute_id, guard_id in sleep_times:
            if guard_id == best_guard_id:
                minutes[minute_id] += 1
        best_minute_id = max(minutes, key=minutes.get)
        result = best_guard_id * best_minute_id
        return result
    
    def solve2(self, sleep_times):
        guards = collections.defaultdict(int)
        for _, minute_id, guard_id in sorted(sleep_times):
            guards[(guard_id, minute_id)] += 1
        best_guard_id, best_minute_id = max(guards, key=guards.get)
        result = best_guard_id * best_minute_id
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        sleep_times = self.get_sleep_times(raw_input_lines)
        solutions = (
            self.solve(sleep_times),
            self.solve2(sleep_times),
            )
        result = solutions
        return result

class Day03: # No Matter How You Slice It
    '''
    No Matter How You Slice It
    https://adventofcode.com/2018/day/3
    '''
    def get_claims(self, raw_input_lines: List[str]):
        claims = {}
        for raw_input_line in raw_input_lines:
            parts = raw_input_line.split(' ')
            claim_id = int(parts[0][1:])
            left, top = tuple(map(int, parts[2][:-1].split(',')))
            width, height = tuple(map(int, parts[3].split('x')))
            claims[claim_id] = (left, top, width, height)
        result = claims
        return result
    
    def solve(self, claims):
        claimed_fabric = collections.defaultdict(set)
        for claim_id, (left, top, width, height) in claims.items():
            for row in range(top, top + height):
                for col in range(left, left + width):
                    claimed_fabric[(row, col)].add(claim_id)
        result = sum(
            1 for
            _, claims in
            claimed_fabric.items() if
            len(claims) >= 2
            )
        return result
    
    def solve2(self, claims):
        claimed_fabric = collections.defaultdict(set)
        for claim_id, (left, top, width, height) in claims.items():
            for row in range(top, top + height):
                for col in range(left, left + width):
                    claimed_fabric[(row, col)].add(claim_id)
        result = -1
        for claim_id, (left, top, width, height) in claims.items():
            overlap_ind = False
            for row in range(top, top + height):
                for col in range(left, left + width):
                    claims = claimed_fabric[(row, col)]
                    if len(claims) > 1:
                        overlap_ind = True
                        break
                if overlap_ind:
                    break
            if not overlap_ind:
                result = claim_id
                break
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        claims = self.get_claims(raw_input_lines)
        solutions = (
            self.solve(claims),
            self.solve2(claims),
            )
        result = solutions
        return result

class Day02: # Inventory Management System
    '''
    Inventory Management System
    https://adventofcode.com/2018/day/2
    '''
    def get_box_ids(self, raw_input_lines: List[str]):
        box_ids = []
        for raw_input_line in raw_input_lines:
            box_ids.append(raw_input_line)
        result = box_ids
        return result
    
    def solve(self, box_ids):
        counts = collections.defaultdict(set)
        for box_id in box_ids:
            chars = collections.Counter(box_id)
            for count in chars.values():
                counts[count].add(box_id)
        result = len(counts[2]) * len(counts[3])
        return result
    
    def solve2(self, box_ids):
        for i in range(len(box_ids)):
            box_id_1 = box_ids[i]
            for j in range(i + 1, len(box_ids)):
                box_id_2 = box_ids[j]
                idx = -1
                for k in range(len(box_id_1)):
                    if box_id_1[k] != box_id_2[k]:
                        if idx >= 0:
                            break
                        idx = k
                else:
                    if idx >= 0:
                        result = box_id_1[:idx] + box_id_1[idx + 1:]
                        return result
        return 'Solution not found!'
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        box_ids = self.get_box_ids(raw_input_lines)
        solutions = (
            self.solve(box_ids),
            self.solve2(box_ids),
            )
        result = solutions
        return result

class Day01: # Chronal Calibration
    '''
    Chronal Calibration
    https://adventofcode.com/2018/day/1
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(int(raw_input_line))
        return result
    
    def solve(self, parsed_input):
        result = sum(parsed_input)
        return result
    
    def solve2(self, parsed_input):
        seen = set()
        frequency = 0
        seen.add(frequency)
        result = None
        while result is None:
            for change in parsed_input:
                frequency += change
                if frequency in seen:
                    result = frequency
                    break
                seen.add(frequency)
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
    python AdventOfCode2018.py 9 < inputs/2018day09.in
    '''
    solvers = {
        1: (Day01, 'Chronal Calibration'),
        2: (Day02, 'Inventory Management System'),
        3: (Day03, 'No Matter How You Slice It'),
        4: (Day04, 'Repose Record'),
        5: (Day05, 'Alchemical Reduction'),
        6: (Day06, 'Chronal Coordinates'),
        7: (Day07, 'The Sum of Its Parts'),
        8: (Day08, 'Memory Maneuver'),
        9: (Day09, 'Marble Mania'),
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
