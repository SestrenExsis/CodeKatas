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
    #     5: (Day05, 'Unknown'),
    #     6: (Day06, 'Unknown'),
    #     7: (Day07, 'Unknown'),
    #     8: (Day08, 'Unknown'),
    #     9: (Day09, 'Unknown'),
    #    10: (Day10, 'Unknown'),
    #    11: (Day11, 'Unknown'),
    #    12: (Day12, 'Unknown'),
    #    13: (Day13, 'Unknown'),
    #    14: (Day14, 'Unknown'),
    #    15: (Day15, 'Unknown'),
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
