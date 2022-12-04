'''
Created on 2022-11-30

@author: Sestren
'''
import argparse
import collections
import copy
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
        result = len(assignments)
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
    python AdventOfCode2022.py 4 < inputs/2022day04.in
    '''
    solvers = {
        1: (Day01, 'Calorie Counting'),
        2: (Day02, 'Rock Paper Scissors'),
        3: (Day03, 'Rucksack Reorganization'),
        4: (Day04, 'Camp Cleanup'),
    #     5: (Day05, 'Day05'),
    #     6: (Day06, 'Day06'),
    #     7: (Day07, 'Day07'),
    #     8: (Day08, 'Day08'),
    #     9: (Day09, 'Day09'),
    #    10: (Day10, 'Day10'),
    #    11: (Day11, 'Day11'),
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
