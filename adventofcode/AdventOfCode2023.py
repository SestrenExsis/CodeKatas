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
    #     3: (Day03, 'Unknown'),
    #     4: (Day04, 'Unknown'),
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
