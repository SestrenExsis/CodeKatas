'''
Created 2021-04-04

@author: Sestren
'''
import argparse
import collections
import copy
import datetime
import functools
import heapq
import hashlib
import itertools
import json
import operator
import re
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
    Template
    https://adventofcode.com/2015/day/?
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

class Day15: # Science for Hungry People
    '''
    Science for Hungry People
    https://adventofcode.com/2015/day/15
    '''
    def get_recipes(self, raw_input_lines: List[str]):
        recipes = {}
        for raw_input_line in raw_input_lines:
            recipe_name, raw_recipe = raw_input_line.split(': ')
            recipe = {}
            raw_properties = raw_recipe.split(', ')
            for raw_property in raw_properties:
                property_name, a = raw_property.split(' ')
                recipe[property_name] = int(a)
            recipes[recipe_name] = recipe
        return recipes
    
    def solve(self, recipes, teaspoons: int):
        recipe_names = list(sorted(recipes))
        max_score = 0
        for a in range(teaspoons):
            for b in range(teaspoons - a):
                for c in range(teaspoons - a - b):
                    d = teaspoons - a - b - c
                    if d < 0:
                        break
                    amounts = [a, b, c, d]
                    capacity = 0
                    durability = 0
                    flavor = 0
                    texture = 0
                    for i in range(4):
                        recipe = recipes[recipe_names[i]]
                        capacity += recipe['capacity'] * amounts[i]
                        durability += recipe['durability'] * amounts[i]
                        flavor += recipe['flavor'] * amounts[i]
                        texture += recipe['texture'] * amounts[i]
                    capacity = 0 if capacity < 0 else capacity
                    durability = 0 if durability < 0 else durability
                    flavor = 0 if flavor < 0 else flavor
                    texture = 0 if texture < 0 else texture
                    score = capacity * durability * flavor * texture
                    max_score = max(max_score, score)
        result = max_score
        return result
    
    def solve2(self, recipes):
        result = len(recipes)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        recipes = self.get_recipes(raw_input_lines)
        solutions = (
            self.solve(recipes, 100),
            self.solve2(recipes),
            )
        result = solutions
        return result

class Day14: # Reindeer Olympics
    '''
    Reindeer Olympics
    https://adventofcode.com/2015/day/14
    '''
    def get_reindeer(self, raw_input_lines: List[str]):
        reindeer = {}
        for raw_input_line in raw_input_lines:
            parts = raw_input_line.split(' ')
            name = parts[0]
            speed = int(parts[3])
            fly_duration = int(parts[6])
            rest_duration = int(parts[13])
            reindeer[name] = (speed, fly_duration, rest_duration)
        result = reindeer
        return result
    
    def solve(self, reindeer):
        race = {}
        for name, info in reindeer.items():
            # distance, state, timer
            race[name] = [0, 'flying', info[1]]
        for _ in range(2503):
            for name in race:
                info = reindeer[name]
                state = race[name][1]
                if state == 'flying':
                    race[name][0] += info[0]
                race[name][2] -= 1
                if race[name][2] < 1:
                    if state == 'flying':
                        race[name][1] = 'resting'
                        race[name][2] = info[2]
                    else:
                        race[name][1] = 'flying'
                        race[name][2] = info[1]
        max_distance = 0
        for name in race:
            max_distance = max(max_distance, race[name][0])
        result = max_distance
        return result
    
    def solve2(self, reindeer):
        race = {}
        for name, info in reindeer.items():
            # distance, state, timer, points
            race[name] = [0, 'flying', info[1], 0]
        for _ in range(2503):
            for name in race:
                info = reindeer[name]
                state = race[name][1]
                if state == 'flying':
                    race[name][0] += info[0]
                race[name][2] -= 1
                if race[name][2] < 1:
                    if state == 'flying':
                        race[name][1] = 'resting'
                        race[name][2] = info[2]
                    else:
                        race[name][1] = 'flying'
                        race[name][2] = info[1]
            max_distance = max(info[0] for info in race.values())
            for name in race:
                if race[name][0] == max_distance:
                    race[name][3] += 1
        result = max(info[3] for info in race.values())
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        reindeer = self.get_reindeer(raw_input_lines)
        solutions = (
            self.solve(reindeer),
            self.solve2(reindeer),
            )
        result = solutions
        return result

class Day13: # Knights of the Dinner Table
    '''
    Knights of the Dinner Table
    https://adventofcode.com/2015/day/13
    '''
    def get_preferences(self, raw_input_lines: List[str]):
        preferences = collections.defaultdict(int)
        for raw_input_line in raw_input_lines:
            parts = raw_input_line.split(' ')
            person_a = parts[0]
            sign = -1 if parts[2] == 'lose' else 1
            value = sign * int(parts[3])
            person_b = parts[-1][:-1]
            pair = (min(person_a, person_b), max(person_a, person_b))
            preferences[pair] += value
        result = preferences
        return result
    
    def solve(self, preferences):
        max_happiness = float('-inf')
        guests = set(pair[0] for pair in preferences)
        guests |= set(pair[1] for pair in preferences)
        for seating in itertools.permutations(guests, len(guests)):
            happiness = 0
            for i in range(len(seating)):
                person_a = seating[i]
                person_b = seating[(i + 1) % len(seating)]
                pair = (min(person_a, person_b), max(person_a, person_b))
                happiness += preferences[pair]
            max_happiness = max(max_happiness, happiness)
        result = max_happiness
        return result
    
    def solve2(self, preferences):
        max_happiness = float('-inf')
        guests = set(pair[0] for pair in preferences)
        guests |= set(pair[1] for pair in preferences)
        guests.add('Me')
        for seating in itertools.permutations(guests, len(guests)):
            happiness = 0
            for i in range(len(seating)):
                person_a = seating[i]
                person_b = seating[(i + 1) % len(seating)]
                pair = (min(person_a, person_b), max(person_a, person_b))
                if pair in preferences:
                    happiness += preferences[pair]
            max_happiness = max(max_happiness, happiness)
        result = max_happiness
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        preferences = self.get_preferences(raw_input_lines)
        solutions = (
            self.solve(preferences),
            self.solve2(preferences),
            )
        result = solutions
        return result

class Day12: # JSAbacusFramework.io
    '''
    JSAbacusFramework.io
    https://adventofcode.com/2015/day/12
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = raw_input_lines[0]
        return result
    
    def solve(self, parsed_input):
        nums = []
        obj = json.loads(parsed_input)
        work = [obj]
        while len(work) > 0:
            element = work.pop()
            element_type = type(element)
            if element_type == list:
                for subelement in element:
                    work.append(subelement)
            elif element_type == dict:
                for subelement in element.values():
                    work.append(subelement)
            elif element_type == int:
                nums.append(element)
            elif element_type == str:
                pass
        result = sum(nums)
        return result
    
    def solve2(self, parsed_input):
        nums = []
        obj = json.loads(parsed_input)
        work = [obj]
        while len(work) > 0:
            element = work.pop()
            element_type = type(element)
            if element_type == list:
                for subelement in element:
                    work.append(subelement)
            elif element_type == dict:
                red_ind = False
                for value in element.values():
                    if value == 'red':
                        red_ind = True
                        break
                if not red_ind:
                    for subelement in element.values():
                        work.append(subelement)
            elif element_type == int:
                nums.append(element)
            elif element_type == str:
                pass
        result = sum(nums)
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

class Day11: # Corporate Policy
    '''
    Corporate Policy
    https://adventofcode.com/2015/day/11
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = raw_input_lines[0]
        return result
    
    def encode(self, password):
        encoded_password = 0
        for i, char in enumerate(reversed(password)):
            encoded_char = ord(char) - ord('a')
            encoded_password += (26 ** i) * encoded_char
        result = encoded_password
        return result

    def decode(self, encoded_password):
        password = collections.deque()
        while encoded_password > 0:
            char = chr(ord('a') + encoded_password % 26)
            encoded_password //= 26
            password.appendleft(char)
        result = password
        return result
    
    def check_validity(self, password):
        straight_ind = False
        banned_chars_ind = False
        paired_indices_found = set()
        for i, char in enumerate(password):
            if i >= 1:
                if password[i] == password[i - 1]:
                    paired_indices_found.add(i)
                    paired_indices_found.add(i - 1)
            if i >= 2:
                if (
                    ord(password[i]) == ord(password[i - 1]) + 1 and
                    ord(password[i - 1]) == ord(password[i - 2]) + 1
                ):
                    straight_ind = True
            if char in 'iol':
                banned_chars_ind = True
                break
        result = all([
            straight_ind,
            not banned_chars_ind,
            len(paired_indices_found) >= 4,
        ])
        return result
    
    def solve(self, prev_password):
        decoded_password = prev_password
        encoded_password = self.encode(prev_password)
        while True:
            encoded_password += 1
            decoded_password = self.decode(encoded_password)
            if self.check_validity(decoded_password):
                break
        result = ''.join(decoded_password)
        return result
    
    def solve2(self, prev_password):
        result = self.solve(self.solve(prev_password))
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        prev_password = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(prev_password),
            self.solve2(prev_password),
            )
        result = solutions
        return result

class Day10: # Elves Look, Elves Say
    '''
    Elves Look, Elves Say
    https://adventofcode.com/2015/day/10
    '''
    def get_sequence(self, raw_input_lines: List[str]):
        sequence = raw_input_lines[0]
        result = sequence
        return result
    
    def bruteforce(self, sequence, iteration_count):
        final_sequence = sequence
        for _ in range(iteration_count):
            counts = []
            for char in final_sequence:
                if len(counts) < 1 or counts[-1][1] != char:
                    counts.append([1, char])
                else:
                    counts[-1][0] += 1
            final_sequence = ''.join(
                ''.join(map(str, count)) for count in counts
            )
        result = final_sequence
        return result
    
    def solve(self, sequence, iteration_count):
        final_sequence = self.bruteforce(sequence, iteration_count)
        result = len(final_sequence)
        return result
    
    def main(self):
        assert self.bruteforce('1', 5) == '312211'
        raw_input_lines = get_raw_input_lines()
        sequence = self.get_sequence(raw_input_lines)
        solutions = (
            self.solve(sequence, 40),
            self.solve(sequence, 50),
            )
        result = solutions
        return result

class Day09: # All in a Single Night
    '''
    All in a Single Night
    https://adventofcode.com/2015/day/9
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        locations = set()
        distances = {}
        for row_data in raw_input_lines:
            source, _, destination, _, distance = row_data.split(' ')
            distance = int(distance)
            distances[(source, destination)] = distance
            distances[(destination, source)] = distance
            locations.add(source)
            locations.add(destination)
        result = (locations, distances)
        return result
    
    def bruteforce(self, locations, distances):
        min_distance = float('inf')
        for route in itertools.permutations(locations):
            distance = 0
            prev_stop = route[0]
            for stop in route[1:]:
                distance += distances[(prev_stop, stop)]
                prev_stop = stop
            min_distance = min(min_distance, distance)
        result = min_distance
        return result
    
    def bruteforce2(self, locations, distances):
        max_distance = float('-inf')
        for route in itertools.permutations(locations):
            distance = 0
            prev_stop = route[0]
            for stop in route[1:]:
                distance += distances[(prev_stop, stop)]
                prev_stop = stop
            max_distance = max(max_distance, distance)
        result = max_distance
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        locations, distances = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.bruteforce(locations, distances),
            self.bruteforce2(locations, distances),
            )
        result = solutions
        return result

class Day08: # Matchsticks
    '''
    Matchsticks
    https://adventofcode.com/2015/day/8
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, parsed_input):
        memory_char_count = 0
        literal_char_count = 0
        for row_data in parsed_input:
            prev_counts = (memory_char_count, literal_char_count)
            stack = []
            memory_char_count += 2
            for char in row_data[1:-1]:
                memory_char_count += 1
                stack.append(char)
                if (
                    len(stack) >= 1 and stack[0] != '\\' or
                    len(stack) >= 2 and stack[1] not in '\\"x' or
                    len(stack) >= 3 and stack[2] not in '0123456789abcdef' or
                    len(stack) >= 4 and stack[3] not in '0123456789abcdef'
                ):
                    literal_char_count += len(stack)
                    stack = []
                if (
                    (
                        len(stack) == 2 and
                        stack[0] == '\\' and
                        stack[1] in '\\"'
                    ) or
                    (
                        len(stack) == 4 and
                        stack[0] == '\\' and
                        stack[1] == 'x' and
                        stack[2] in '0123456789abcdef' and
                        stack[3] in '0123456789abcdef'
                    )
                ):
                    literal_char_count += 1
                    stack = []
        result = memory_char_count - literal_char_count
        return result
    
    def solve2(self, parsed_input):
        literal_char_count = 0
        encoded_char_count = 0
        for row_data in parsed_input:
            literal_char_count += len(row_data)
            encoded_char_count += 2
            for char in row_data:
                if char in ('"', '\\'):
                    encoded_char_count += 2
                else:
                    encoded_char_count += 1
        result = encoded_char_count - literal_char_count
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

class Day07: # Some Assembly Required
    '''
    Some Assembly Required
    https://adventofcode.com/2015/day/7
    '''
    def get_connections(self, raw_input_lines: List[str]):
        connections = {}
        for raw_input_line in raw_input_lines:
            a, wire = raw_input_line.split(' -> ')
            connection = list(a.split(' '))
            for i in range(len(connection)):
                try:
                    value = int(connection[i])
                    connection[i] = value
                except ValueError:
                    pass
            connections[wire] = tuple(connection)
        result = connections
        return result
    
    def solve(self, connections):
        OPS = {'NOT', 'AND', 'OR', 'LSHIFT', 'RSHIFT'}
        MODULO = 2 ** 16 # 0 to 65535
        wires = set(connections.keys())
        inputs = {}
        while len(wires) > 0:
            for wire in wires:
                needed = set()
                connection = connections[wire]
                for part in connection:
                    if (
                        part not in OPS and
                        type(part) == str and
                        part not in inputs.keys()
                    ):
                        needed.add(part)
                if len(needed) > 0:
                    continue
                if len(connection) == 1:
                    a = connection[0]
                    if type(a) is int:
                        inputs[wire] = a
                    else:
                        inputs[wire] = inputs[a]
                elif len(connection) == 2:
                    OP, a = connection
                    assert OP == 'NOT'
                    if type(a) is str:
                        a = inputs[a]
                    value = ~a
                    inputs[wire] = value
                elif len(connection) == 3:
                    a, OP, b = connection
                    value = 0
                    if type(a) is str:
                        a = inputs[a]
                    if type(b) is str:
                        b = inputs[b]
                    if OP == 'AND':
                        value = a & b
                    elif OP == 'OR':
                        value = a | b
                    elif OP == 'LSHIFT':
                        value = a << b
                    elif OP == 'RSHIFT':
                        value = a >> b
                    value = value % MODULO
                    inputs[wire] = value
                else:
                    raise AssertionError
            wires -= inputs.keys()
        result = inputs['a']
        return result
    
    def solve2(self, signal, connections):
        connections['b'] = (signal, )
        result = self.solve(connections)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        connections = self.get_connections(raw_input_lines)
        signal = self.solve(copy.deepcopy(connections))
        solutions = (
            signal,
            self.solve2(signal, copy.deepcopy(connections)),
            )
        result = solutions
        return result

class Day06: # Probably a Fire Hazard
    '''
    Probably a Fire Hazard
    https://adventofcode.com/2015/day/6
    '''
    def get_instructions(self, raw_input_lines: List[str]):
        instructions = []
        for raw_input_line in raw_input_lines:
            a, b = raw_input_line.split(' through ')
            mode = 'ERROR'
            if 'toggle' in a:
                a1, a2 = a.split(' ')
                left, top = tuple(map(int, a2.split(',')))
                mode = 'toggle'
            else:
                a1, a2, a3 = a.split(' ')
                left, top = tuple(map(int, a3.split(',')))
                if a2 == 'on':
                    mode = 'turn_on'
                elif a2 == 'off':
                    mode = 'turn_off'
            right, bottom = tuple(map(int, b.split(',')))
            instruction = (mode, left, top, right, bottom)
            instructions.append(instruction)
        result = instructions
        return result
    
    def solve(self, instructions):
        lights = [0] * (1000 * 1000)
        for mode, left, top, right, bottom in instructions:
            for row in range(top, bottom + 1):
                for col in range(left, right + 1):
                    index = 1000 * row + col
                    if mode == 'turn_on':
                        lights[index] = 1
                    elif mode == 'turn_off':
                        lights[index] = 0
                    elif mode == 'toggle':
                        lights[index] =  1 - lights[index]
        result = sum(lights)
        return result
    
    def solve2(self, instructions):
        lights = [0] * (1000 * 1000)
        for mode, left, top, right, bottom in instructions:
            for row in range(top, bottom + 1):
                for col in range(left, right + 1):
                    index = 1000 * row + col
                    if mode == 'turn_on':
                        lights[index] += 1
                    elif mode == 'turn_off':
                        lights[index] = max(0, lights[index] - 1)
                    elif mode == 'toggle':
                        lights[index] += 2
        result = sum(lights)
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

class Day05: # Doesn't He Have Intern-Elves For This?
    '''
    Doesn't He Have Intern-Elves For This?
    https://adventofcode.com/2015/day/5
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, parsed_input):
        '''
        Nice strings:
            Contain at least three vowels (aeiou)
            Contain at least one letter that appears twice in a row
            Do not contain the strings ab, cd, pq, or xy
        '''
        nice_string_count = 0
        for string in parsed_input:
            nice_ind = all([
                sum([
                    string.count('a'),
                    string.count('e'),
                    string.count('i'),
                    string.count('o'),
                    string.count('u'),
                ]) >= 3,
                any(
                    string[i] == string[i - 1] for
                    i in range(1, len(string))
                ),
                all([
                    'ab' not in string,
                    'cd' not in string,
                    'pq' not in string,
                    'xy' not in string,
                ]),
            ])
            if nice_ind:
                nice_string_count += 1
        result = nice_string_count
        return result
    
    def solve2(self, parsed_input):
        '''
        Nice strings:
            Contain a pair of any two letters that appear twice without overlapping
            Contain at least one letter which repeats with ONE letter between them
        '''
        nice_string_count = 0
        for string in parsed_input:
            pairs = {}
            duplicate_pair_found = False
            triplet_found = False
            for i in range(1, len(string)):
                pair = string[i - 1:i + 1]
                if pair in pairs and pairs[pair] < i - 1:
                    duplicate_pair_found = True
                if pair not in pairs:
                    pairs[pair] = i
                if i < len(string) - 1:
                    triplet = string[i - 1: i + 2]
                    if triplet[0] == triplet[2]:
                        triplet_found = True
            if duplicate_pair_found and triplet_found:
                nice_string_count += 1
        result = nice_string_count
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

class Day04: # The Ideal Stocking Stuffer
    '''
    The Ideal Stocking Stuffer
    https://adventofcode.com/2015/day/4
    '''
    def get_secret_key(self, raw_input_lines: List[str]):
        result = raw_input_lines[0]
        return result
    
    def solve(self, secret_key):
        num = 0
        while True:
            algorithm = hashlib.md5()
            key = secret_key + str(num)
            algorithm.update(key.encode('utf-8'))
            if algorithm.hexdigest()[:5] == '00000':
                break
            num += 1
        result = num
        return result
    
    def solve2(self, secret_key):
        num = 0
        while True:
            algorithm = hashlib.md5()
            key = secret_key + str(num)
            algorithm.update(key.encode('utf-8'))
            if algorithm.hexdigest()[:6] == '000000':
                break
            num += 1
        result = num
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        secret_key = self.get_secret_key(raw_input_lines)
        solutions = (
            self.solve(secret_key),
            self.solve2(secret_key),
            )
        result = solutions
        return result

class Day03: # Perfectly Spherical Houses in a Vacuum
    '''
    Perfectly Spherical Houses in a Vacuum
    https://adventofcode.com/2015/day/3
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = raw_input_lines[0]
        return result
    
    def solve(self, parsed_input):
        row = 0
        col = 0
        houses = set()
        for char in parsed_input:
            if char == '^':
                row -= 1
            elif char == '>':
                col += 1
            elif char == 'v':
                row += 1
            elif char == '<':
                col -= 1
            houses.add((row, col))
        result = len(houses)
        return result
    
    def solve2(self, parsed_input):
        santa_id = 0
        santas = [
            (0, 0),
            (0, 0),
            ]
        houses = set()
        for char in parsed_input:
            pos = santas[santa_id]
            if char == '^':
                santas[santa_id] = (pos[0] - 1, pos[1])
            elif char == 'v':
                santas[santa_id] = (pos[0] + 1, pos[1])
            elif char == '<':
                santas[santa_id] = (pos[0], pos[1] - 1)
            elif char == '>':
                santas[santa_id] = (pos[0], pos[1] + 1)
            houses.add(santas[santa_id])
            santa_id = 1 - santa_id
        result = len(houses)
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

class Day02: # I Was Told There Would Be No Math
    '''
    I Was Told There Would Be No Math
    https://adventofcode.com/2015/day/2
    '''
    def get_dimensions(self, raw_input_lines: List[str]):
        dimensions = []
        for raw_input_line in raw_input_lines:
            # Dimension = (Length, Width, Height)
            dimension = tuple(map(int, raw_input_line.split('x')))
            dimensions.append(dimension)
        result = dimensions
        return result
    
    def solve(self, dimensions: List[Tuple[int]]):
        wrapping_paper = []
        for dimension in dimensions:
            l, w, h = dimension
            smallest_side = min((l * w, w * h, h * l))
            needed = 2 * l * w + 2 * w * h + 2 * h * l + smallest_side
            wrapping_paper.append(needed)
        result = sum(wrapping_paper)
        return result
    
    def solve2(self, dimensions: List[Tuple[int]]):
        ribbon = []
        for dimension in dimensions:
            l, w, h = dimension
            perimeters = [
                2 * (l + w),
                2 * (w + h),
                2 * (h + l),
            ]
            volume = l * w * h
            needed = min(perimeters) + volume
            ribbon.append(needed)
        result = sum(ribbon)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        dimensions = self.get_dimensions(raw_input_lines)
        solutions = (
            self.solve(dimensions),
            self.solve2(dimensions),
            )
        result = solutions
        return result

class Day01: # Not Quite Lisp
    '''
    Not Quite Lisp
    https://adventofcode.com/2015/day/1
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = raw_input_lines[0]
        return result
    
    def solve(self, parsed_input):
        floor = 0
        for char in parsed_input:
            if char == '(':
                floor += 1
            elif char == ')':
                floor -= 1
        result = floor
        return result
    
    def solve2(self, parsed_input):
        floor = 0
        position = 0
        for position, char in enumerate(parsed_input, start=1):
            if char == '(':
                floor += 1
            elif char == ')':
                floor -= 1
            if floor == -1:
                result = position
                break
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
    python AdventOfCode2015.py 15 < inputs/2015day15.in
    '''
    solvers = {
        1: (Day01, 'Not Quite Lisp'),
        2: (Day02, 'I Was Told There Would Be No Math'),
        3: (Day03, 'Perfectly Spherical Houses in a Vacuum'),
        4: (Day04, 'The Ideal Stocking Stuffer'),
        5: (Day05, 'Doesn\'t He Have Intern-Elves For This?'),
        6: (Day06, 'Probably a Fire Hazard'),
        7: (Day07, 'Some Assembly Required'),
        8: (Day08, 'Matchsticks'),
        9: (Day09, 'All in a Single Night'),
       10: (Day10, 'Elves Look, Elves Say'),
       11: (Day11, 'Corporate Policy'),
       12: (Day12, 'JSAbacusFramework.io'),
       13: (Day13, 'Knights of the Dinner Table'),
       14: (Day14, 'Reindeer Olympics'),
       15: (Day15, 'Science for Hungry People'),
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
