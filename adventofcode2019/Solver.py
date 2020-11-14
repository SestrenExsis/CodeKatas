'''
Created on Nov 14, 2020

@author: Sestren
'''
import argparse
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
    python adventofcode2019\Solver.py 1 < adventofcode2019\day01.in
    '''
    solvers = {
        1: (Day01, 'The Tyranny of the Rocket Equation'),
    #     2: (Day02, '1202 Program Alarm'),
    #     3: (Day03, 'Crossed Wires'),
    #     4: (Day04, 'Secure Container'),
    #     5: (Day05, 'Sunny with a Chance of Asteroids'),
    #     6: (Day06, 'Universal Orbit Map'),
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
