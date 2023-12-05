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
    
    def solve2_slowly(self, almanac):
        almanac2 = copy.deepcopy(almanac)
        seeds = []
        for index in range(0, len(almanac['seeds']), 2):
            seed_start = almanac['seeds'][index]
            seed_count = almanac['seeds'][index + 1]
            for seed in range(seed_start, seed_start + seed_count):
                seeds.append(seed)
        almanac2['seeds'] = seeds
        result = self.solve(almanac2)
        return result

    def split_range(self, source_range, mapping):
        (source_start, source_end) = source_range
        target_ranges = []
        for (start, end), offset in mapping.items():
            if start <= source_start:
                if end >= source_start:
                    target_range = (
                        offset + max(start, source_start),
                        offset + min(end, source_end),
                    )
                    target_ranges.append(target_range)
                else:
                    pass # no overlap
            elif end <= source_end:
                if end >= source_start:
                    target_range = (
                        offset + max(start, source_start),
                        offset + min(end, source_end),
                    )
                    target_ranges.append(target_range)
                pass
            else:
                pass # no overlap
        result = target_ranges
        return result
    
    def solve2_with_ranges(self, almanac):
        seed_ranges = []
        for index in range(0, len(almanac['seeds']), 2):
            seed_start = almanac['seeds'][index]
            seed_count = almanac['seeds'][index + 1]
            seed_ranges.append((seed_start, seed_start + seed_count - 1))
        # Convert seed to soil
        soil_ranges = []
        for source_range in seed_ranges:
            target_ranges = self.split_range(source_range, almanac['seed-to-soil'])
            soil_ranges.extend(target_ranges)
        print('soil_ranges:', soil_ranges)
        # Convert soil to fertilizer
        fertilizer_ranges = []
        for source_range in soil_ranges:
            target_ranges = self.split_range(source_range, almanac['soil-to-fertilizer'])
            fertilizer_ranges.extend(target_ranges)
        print('fertilizer_ranges:', fertilizer_ranges)
        # Convert fertilizer to water
        water_ranges = []
        for source_range in fertilizer_ranges:
            target_ranges = self.split_range(source_range, almanac['fertilizer-to-water'])
            water_ranges.extend(target_ranges)
        # Convert water to light
        light_ranges = []
        for source_range in water_ranges:
            target_ranges = self.split_range(source_range, almanac['water-to-light'])
            light_ranges.extend(target_ranges)
        # Convert light to temperature
        temperature_ranges = []
        for source_range in light_ranges:
            target_ranges = self.split_range(source_range, almanac['light-to-temperature'])
            temperature_ranges.extend(target_ranges)
        # Convert temperature to humidity
        humidity_ranges = []
        for source_range in temperature_ranges:
            target_ranges = self.split_range(source_range, almanac['temperature-to-humidity'])
            humidity_ranges.extend(target_ranges)
        # Convert humidity to location
        location_ranges = []
        for source_range in humidity_ranges:
            target_ranges = self.split_range(source_range, almanac['humidity-to-location'])
            location_ranges.extend(target_ranges)
        # Calculate minimum location
        result = float('inf')
        for start, end in location_ranges:
            result = min(result, start)
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
