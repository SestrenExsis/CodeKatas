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

class Day05: # Alchemical Reduction
    '''
    Alchemical Reduction
    https://adventofcode.com/2018/day/5
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = raw_input_lines[0]
        return result
    
    def solve(self, initial_polymer):
        polymer = list(initial_polymer)
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
        result = len(polymer)
        return result
    
    def solve2(self, initial_polymer):
        result = len(initial_polymer)
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
            day1, minute1, event1 = raw_records[i + 1]
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
    python AdventOfCode2018.py 5 < inputs/2018day05.in
    '''
    solvers = {
        1: (Day01, 'Chronal Calibration'),
        2: (Day02, 'Inventory Management System'),
        3: (Day03, 'No Matter How You Slice It'),
        4: (Day04, 'Repose Record'),
        5: (Day05, 'Alchemical Reduction'),
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
