'''
Created on 2024-11-30

@author: Sestren
'''
import argparse

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
    https://adventofcode.com/2024/day/?
    '''
    def get_parsed_input(self, raw_input_lines: list[str]):
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

class Day02: # Red-Nosed Reports
    '''
    https://adventofcode.com/2024/day/2
    '''
    def get_reports(self, raw_input_lines: list[str]):
        reports = []
        for raw_input_line in raw_input_lines:
            report = tuple(map(int, raw_input_line.split()))
            reports.append(report)
        result = reports
        return result
    
    def solve(self, reports):
        safe_count = 0
        for report in reports:
            prev_diff = 0
            for i in range(1, len(report)):
                diff = report[i] - report[i - 1]
                if not(1 <= abs(diff) <= 3):
                    break
                if (
                    (prev_diff < 0 and diff > 0) or
                    (prev_diff > 0 and diff < 0)
                ):
                    break
                prev_diff = diff
            else:
                safe_count += 1
        result = safe_count
        return result
    
    def solve2(self, reports):
        result = len(reports)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        reports = self.get_reports(raw_input_lines)
        solutions = (
            self.solve(reports),
            self.solve2(reports),
            )
        result = solutions
        return result

class Day01: # Historian Hysteria
    '''
    https://adventofcode.com/2024/day/1
    '''
    def get_list_pairs(self, raw_input_lines: list[str]):
        left = []
        right = []
        for raw_input_line in raw_input_lines:
            (a, b) = map(int, raw_input_line.split())
            left.append(a)
            right.append(b)
        result = (left, right)
        return result
    
    def solve(self, left, right):
        sorted_left = sorted(left)
        sorted_right = sorted(right)
        assert len(sorted_left) == len(sorted_right)
        diffs = []
        for i in range(len(sorted_left)):
            diff = abs(sorted_left[i] - sorted_right[i])
            diffs.append(diff)
        result = sum(diffs)
        return result
    
    def solve2(self, left, right):
        right_counts = {}
        for num in right:
            if num not in right_counts:
                right_counts[num] = 0
            right_counts[num] += 1
        similarity_scores = []
        for num in left:
            if num not in right_counts:
                right_counts[num] = 0
            similarity_score = num * right_counts[num]
            similarity_scores.append(similarity_score)
        result = sum(similarity_scores)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        (left, right) = self.get_list_pairs(raw_input_lines)
        solutions = (
            self.solve(left, right),
            self.solve2(left, right),
            )
        result = solutions
        return result

if __name__ == '__main__':
    '''
    Usage
    python AdventOfCode2024.py 1 < inputs/2024day01.in
    '''
    solvers = {
        1: (Day01, 'Historian Hysteria'),
        2: (Day02, 'Red-Nosed Reports'),
    #     3: (Day03, 'XXX'),
    #     4: (Day04, 'XXX'),
    #     5: (Day05, 'XXX'),
    #     6: (Day06, 'XXX'),
    #     7: (Day07, 'XXX'),
    #     8: (Day08, 'XXX'),
    #     9: (Day09, 'XXX'),
    #    10: (Day10, 'XXX'),
    #    11: (Day11, 'XXX'),
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
