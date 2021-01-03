'''
Created on Nov 14, 2020

@author: Sestren
'''
import argparse
import collections
from typing import List
    
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
    https://adventofcode.com/2019/day/?
    '''
    def get_parsed_input(self, raw_input_lines: List[str]) -> List[str]:
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, parsed_input: List[str]) -> int:
        result = 0
        return result
    
    def solve2(self, parsed_input: List[str]) -> str:
        result = 0
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

class Day06: # Universal Orbit Map
    '''
    Universal Orbit Map
    https://adventofcode.com/2019/day/6
    '''
    def get_orbits(self, raw_input_lines: List[str]) -> List[str]:
        orbits = {}
        for raw_input_line in raw_input_lines:
            target, source = raw_input_line.split(')')
            orbits[source] = target
        result = orbits
        return result
    
    def solve(self, orbits):
        depths = collections.defaultdict(int)
        depths['COM'] = 0
        work = collections.deque()
        work.append('COM')
        while len(work) > 0:
            current_source = work.pop()
            for source, target in orbits.items():
                if target == current_source:
                    work.appendleft(source)
                    depths[source] = depths[target] + 1
        result = sum(depths.values())
        return result
    
    def solve2(self, orbits):
        your_path = []
        current_orbit = 'YOU'
        while len(your_path) == 0 or current_orbit != 'COM':
            current_orbit = orbits[current_orbit]
            your_path.append(current_orbit)
        santas_path = []
        current_orbit = 'SAN'
        while len(santas_path) == 0 or current_orbit != 'COM':
            current_orbit = orbits[current_orbit]
            santas_path.append(current_orbit)
        result = -1
        for i in range(len(your_path)):
            current_orbit = your_path[i]
            try:
                j = santas_path.index(current_orbit)
                result = i + j
                break
            except (ValueError) as e:
                continue
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        orbits = self.get_orbits(raw_input_lines)
        solutions = (
            self.solve(orbits),
            self.solve2(orbits),
            )
        result = solutions
        return result

class Day04:
    '''
    Secure Container
    https://adventofcode.com/2019/day/4
    '''
    def get_parsed_input(self, raw_input_lines: 'List'):
        lower, upper = map(int, raw_input_lines[0].split('-'))
        return (lower, upper)
    
    def solve(self, lower: int, upper: int):
        valid_count = 0
        for password in range(lower, upper + 1):
            chars = str(password)
            n = len(chars)
            if n != 6:
                continue
            repeat_found = False
            valid = True
            for i in range(1, n):
                if int(chars[i]) < int(chars[i - 1]):
                    valid = False
                    break
                if chars[i] == chars[i - 1]:
                    repeat_found = True
            if not valid or not repeat_found:
                continue
            valid_count += 1
        result = valid_count
        return result
    
    def solve2(self, lower: int, upper: int):
        valid_count = 0
        for password in range(lower, upper + 1):
            chars = str(password)
            n = len(chars)
            if n != 6:
                continue
            valid = True
            streaks = [1]
            for i in range(1, n):
                if int(chars[i]) < int(chars[i - 1]):
                    valid = False
                    break
                if chars[i] == chars[i - 1]:
                    streaks[-1] += 1
                else:
                    streaks.append(1)
            if not valid or 2 not in streaks:
                continue
            valid_count += 1
        result = valid_count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        parsed_input = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(*parsed_input),
            self.solve2(*parsed_input),
            )
        result = solutions
        return result

class Day03:
    '''
    Crossed Wires
    https://adventofcode.com/2019/day/3
    '''
    def get_parsed_input(self, raw_input_lines: 'List'):
        wires = []
        for raw_input_line in raw_input_lines:
            wires.append(raw_input_line.split(','))
        return wires
    
    def solve(self, wires: 'List'):
        lines = [set(), set()]
        for i, wire in enumerate(wires):
            c = (0, 0)
            for move in wire:
                direction, distance = move[:1], int(move[1:])
                if direction == 'R':
                    d = (c[0] + distance, c[1])
                    lines[i].add((c, d))
                elif direction == 'D':
                    d = (c[0], c[1] + distance)
                    lines[i].add((c, d))
                elif direction == 'L':
                    d = (c[0] - distance, c[1])
                    lines[i].add((c, d))
                elif direction == 'U':
                    d = (c[0], c[1] - distance)
                    lines[i].add((c, d))
                c = d
        intersections = set()
        for a in lines[0]:
            for b in lines[1]:
                if a[0][0] == a[1][0] and b[0][1] == b[1][1]:
                    intersection = (a[0][0], b[0][1])
                    if (((b[0][0] <= intersection[0] <= b[1][0]) or 
                        (b[1][0] <= intersection[0] <= b[0][0])) and
                        ((a[0][1] <= intersection[1] <= a[1][1]) or
                         (a[1][1] <= intersection[1] <= a[0][1]))
                        ):
                        intersections.add(intersection)
                elif b[0][0] == b[1][0] and a[0][1] == a[1][1]:
                    intersection = (b[0][0], a[0][1])
                    if (((b[0][1] <= intersection[1] <= b[1][1]) or 
                        (b[1][1] <= intersection[1] <= b[0][1])) and
                        ((a[0][0] <= intersection[0] <= a[1][0]) or
                         (a[1][0] <= intersection[0] <= a[0][0]))
                        ):
                        intersections.add(intersection)
        closest_distance = None
        for intersection in intersections:
            distance = abs(intersection[0]) + abs(intersection[1])
            if closest_distance is None or distance < closest_distance:
                closest_distance = distance
        result = closest_distance
        return result
    
    def solve2(self, wires: 'List'):
        lines = [set(), set()]
        directions = {
            'R': (1, 0),
            'D': (0, 1),
            'L': (-1, 0),
            'U': (0, -1),
            }
        for i, wire in enumerate(wires):
            begin = (0, 0)
            total_distance = 0
            for move in wire:
                direction, distance = move[:1], int(move[1:])
                end = (
                    begin[0] + distance * directions[direction][0],
                    begin[1] + distance * directions[direction][1],
                    )
                lines[i].add((total_distance, begin, end))
                total_distance += distance
                begin = end
        intersections = set()
        for a in lines[0]:
            for b in lines[1]:
                if a[1][0] == a[2][0] and b[1][1] == b[2][1]:
                    intersection = ((a[1][0], b[1][1]), (a, b))
                    if (((b[1][0] <= intersection[0][0] <= b[2][0]) or 
                        (b[2][0] <= intersection[0][0] <= b[1][0])) and
                        ((a[1][1] <= intersection[0][1] <= a[2][1]) or
                         (a[2][1] <= intersection[0][1] <= a[1][1]))
                        ):
                        intersections.add(intersection)
                elif b[1][0] == b[2][0] and a[1][1] == a[2][1]:
                    intersection = ((b[1][0], a[1][1]), (a, b))
                    if (((b[1][1] <= intersection[0][1] <= b[2][1]) or 
                        (b[2][1] <= intersection[0][1] <= b[1][1])) and
                        ((a[1][0] <= intersection[0][0] <= a[2][0]) or
                         (a[2][0] <= intersection[0][0] <= a[1][0]))
                        ):
                        intersections.add(intersection)
        fewest_steps = None
        for intersection in intersections:
            point, (a, b) = intersection
            steps = a[0] + b[0]
            a_start, b_start = a[1], b[1]
            steps += abs(a_start[0] - point[0]) + abs(a_start[1] - point[1])
            steps += abs(b_start[0] - point[0]) + abs(b_start[1] - point[1])
            if fewest_steps is None or steps < fewest_steps:
                fewest_steps = steps
        result = fewest_steps
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        wires = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(wires),
            self.solve2(wires),
            )
        result = solutions
        return result

class Day01:
    '''
    The Tyranny of the Rocket Equation
    https://adventofcode.com/2019/day/1
    '''
    def get_parsed_input(self, raw_input_lines: 'List'):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(int(raw_input_line))
        return result
    
    def solve(self, parsed_input: 'List'):
        result = 0
        for mass in parsed_input:
            result += mass // 3 - 2
        return result
    
    def solve2(self, parsed_input: 'List'):
        result = 0
        for mass in parsed_input:
            total_fuel = 0
            current_mass = mass
            while True:
                fuel = max(0, current_mass // 3 - 2)
                total_fuel += fuel
                current_mass = fuel
                if fuel <= 0:
                    break
            result += total_fuel
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
    python AdventOfCode2019.py 6 < inputs/2019day06.in
    '''
    solvers = {
        1: (Day01, 'The Tyranny of the Rocket Equation'),
    #     2: (Day02, '1202 Program Alarm'),
        3: (Day03, 'Crossed Wires'),
        4: (Day04, 'Secure Container'),
    #     5: (Day05, 'Sunny with a Chance of Asteroids'),
        6: (Day06, 'Universal Orbit Map'),
    #     7: (Day07, 'Amplification Circuit'),
    #     8: (Day08, 'Space Image Format'),
    #     9: (Day09, 'Sensor Boost'),
    #    10: (Day10, 'Monitoring Station'),
    #    11: (Day11, 'Space Police'),
    #    12: (Day12, 'The N-Body Problem'),
    #    13: (Day13, 'Care Package'),
    #    14: (Day14, 'Space Stoichiometry'),
    #    15: (Day15, 'Oxygen System'),
    #    16: (Day16, 'Flawed Frequency Transmission'),
    #    17: (Day17, 'Set and Forget'),
    #    18: (Day18, 'Many-Worlds Interpretation'),
    #    19: (Day19, 'Tractor Beam'),
    #    20: (Day20, 'Donut Maze'),
    #    21: (Day21, 'Springdroid Adventure'),
    #    22: (Day22, 'Slam Shuffle'),
    #    23: (Day23, 'Category Six'),
    #    24: (Day24, 'Planet of Discord'),
    #    25: (Day25, 'Cryostasis'),
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
