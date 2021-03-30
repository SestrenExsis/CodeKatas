'''
Created on Jan 4, 2021

@author: Sestren
'''
import argparse
import collections
import copy
from functools import lru_cache
import heapq
import itertools
import math
import random
import sys
from typing import Dict, List, Set, Tuple

class Vestigium: # 2020.Q.1
    '''
    2020.Q.1
    https://codingcompetitions.withgoogle.com/codejam/round/000000000019fd27/000000000020993c
    '''
    def solve(self, matrix):
        N = len(matrix)
        triangle_sum = (N * (N + 1)) // 2
        trace = sum(matrix[i][i] for i in range(N))
        row_repeat_count = sum(
            1 for
            row_data in matrix if
            sum(row_data) != triangle_sum or len(set(row_data)) != N
            )
        col_repeat_count = 0
        for col in range(N):
            col_sum = sum(matrix[row][col] for row in range(N))
            if any((
                col_sum != triangle_sum,
                len(set(matrix[i][col] for i in range(N))) != N,
                )):
                col_repeat_count += 1
        result = (trace, row_repeat_count, col_repeat_count)
        return result
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            matrix_size = int(input())
            matrix = [
                list(map(int, input().split(' '))) for
                _ in range(matrix_size)
                ]
            (trace, row_repeat_count, col_repeat_count) = self.solve(matrix)
            output_row = 'Case #{}: {} {} {}'.format(
                test_id,
                trace,
                row_repeat_count,
                col_repeat_count,
                )
            output.append(output_row)
            print(output_row)
        return output

class NestingDepth: # 2020.Q.2
    '''
    2020.Q.2
    https://codingcompetitions.withgoogle.com/codejam/round/000000000019fd27/0000000000209a9f
    '''
    def solve(self, raw_input):
        chars = []
        depth = 0
        for char in raw_input:
            target_depth = int(char)
            while depth < target_depth:
                chars.append('(')
                depth += 1
            while depth > target_depth:
                chars.append(')')
                depth -= 1
            chars.append(char)
        while depth > 0:
            chars.append(')')
            depth -= 1
        result = ''.join(chars)
        return result
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            raw_input = input()
            solution = self.solve(raw_input)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

class ParentingPartneringReturns: # 2020.Q.3
    '''
    2020.Q.3
    https://codingcompetitions.withgoogle.com/codejam/round/000000000019fd27/000000000020bdf9
    '''
    def solve(self, activities):
        schedule = []
        parents = {
            # key = parent_code: str
            # value = busy_til: int
            'C': 0,
            'J': 0,
        }
        for (start, end, activity_index) in sorted(activities):
            try:
                parent_code = next(iter(
                    parent_code for
                    parent_code, busy_til in parents.items() if
                    busy_til <= start
                    ))
                schedule.append((parent_code, activity_index))
                parents[parent_code] = end
            except StopIteration:
                break
        result = 'IMPOSSIBLE'
        if len(schedule) >= len(activities):
            rearranged_schedule = (
                parent_code for
                parent_code, activity_index in
                sorted(schedule, key=lambda x: x[1])
            )
            result = ''.join(rearranged_schedule)
        return result
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            activity_count = int(input())
            activities = set()
            for activity_index in range(activity_count):
                activity = tuple(map(int, input().split(' ')))
                activities.add((activity[0], activity[1], activity_index))
            solution = self.solve(activities)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

class Indicium__Incomplete: # 2020.Q.5
    '''
    2020.Q.5
    https://codingcompetitions.withgoogle.com/codejam/round/000000000019fd27/0000000000209aa0
    '''
    def get_main_diagonal(self, matrix_size: int, target_trace: int) -> Tuple[int]:
        assert 4 <= matrix_size <= 50
        assert matrix_size <= target_trace <= matrix_size ** 2
        N = matrix_size
        K = target_trace
        main_diagonal = None
        if N == 2:
            if K in (2, 4):
                main_diagonal = (K // N, K // N)
        elif N == 3:
            lookup = {
                (3, 3): (1, 1, 1),
                (3, 6): (2, 2, 2),
                (3, 9): (3, 3, 3),
            }
            main_diagonal = lookup[(N, K)]
        else:
            lookup = {
                (5, 16): (2, 2, 4, 4, 4),
                (5, 15): (1, 2, 4, 4, 4),
            }
            main_diagonal = lookup[(N, K)]
        result = main_diagonal
        return result

    def solve(self, matrix_size: int, target_trace: int):
        assert 2 <= matrix_size <= 50
        assert matrix_size <= target_trace <= matrix_size ** 2
        latin_square = None
        if matrix_size == 2:
            if target_trace == 2:
                latin_square = [[1, 2], [2, 1]]
            elif target_trace == 4:
                latin_square = [[2, 1], [1, 2]]
        elif matrix_size == 3:
            if target_trace == 3:
                latin_square = [[1, 2, 3], [3, 1, 2], [2, 3, 1]]
            elif target_trace == 6:
                latin_square = [[2, 3, 1], [1, 2, 3], [3, 1, 2]]
            elif target_trace == 9:
                latin_square = [[3, 1, 2], [2, 3, 1], [1, 2, 3]]
        elif target_trace not in (matrix_size + 1, matrix_size ** 2 - 1):
            latin_square = list([0] * matrix_size for _ in range(matrix_size))
            rownums = list(
                set(range(1, matrix_size + 1)) for _
                in range(matrix_size)
                )
            colnums = list(
                set(range(1, matrix_size + 1)) for _
                in range(matrix_size)
                )
            main_diagonal = self.get_main_diagonal(matrix_size, target_trace)
            if (
                main_diagonal[0] != main_diagonal[2] and
                main_diagonal[1] != main_diagonal[2]
                ):
                rownums[0].remove(main_diagonal[2])
                colnums[0].remove(main_diagonal[2])
                latin_square[0][1] = main_diagonal[2]
                rownums[1].remove(main_diagonal[2])
                colnums[1].remove(main_diagonal[2])
                latin_square[1][0] = main_diagonal[2]
            for i in range(len(main_diagonal)):
                rownums[i].remove(main_diagonal[i])
                colnums[i].remove(main_diagonal[i])
                latin_square[i][i] = main_diagonal[i]
            # TODO: Allegedly, you can greedily pick as long as you start with
            # the B and C rows filled in with A where missing
            # 14352
            # 42135
            # 51423
            # 35241
            # 23514
            # if main_diagonal[0] != main_diagonal[1]:
            #   Fill top-left 2x2 with itself rotated
            # elif main_diagonal[0] == main_diagonal[1] != main_diagonal[2]:
            #   Fill top-left 2x2 with main_diagonal[2]
            # elif main_diagonal[0] == main_diagonal[1]:
            #   Pick an arbitrary number to fill the top-left 2x2
            for row in range(matrix_size):
                for col in range(matrix_size):
                    if latin_square[row][col] > 0:
                        continue
                    valid_nums = rownums[row] & colnums[col]
                    if len(valid_nums) == 0:
                        print('ERROR')
                        break
                    num = valid_nums.pop()
                    # TODO: Figure out why valid numbers are empty
                    rownums[row].remove(num)
                    colnums[col].remove(num)
                    latin_square[row][col] = num
        result = 'IMPOSSIBLE'
        if latin_square is not None:
            result = 'POSSIBLE\n' + '\n'.join((
                ' '.join(map(str, row_data)) for row_data in latin_square
                ))
        return result
    
    def main(self):
        # Tests
        # assert self.get_main_diagonal(2, 2) == (1, 1)
        # assert self.get_main_diagonal(2, 4) == (2, 2)
        # assert self.get_main_diagonal(3, 3) == (1, 1, 1)
        # assert self.get_main_diagonal(3, 6) == (2, 2, 2)
        # assert self.get_main_diagonal(3, 9) == (3, 3, 3)
        # assert self.get_main_diagonal(5, 16) == (2, 2, 4, 4, 4)
        # assert self.get_main_diagonal(5, 15) == (1, 2, 4, 4, 4)
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            matrix_size, target_trace = tuple(map(int, input().split(' ')))
            solution = self.solve(matrix_size, target_trace)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

class PatternMatching: # 2020.1A.1
    '''
    2020.1A.1
    https://codingcompetitions.withgoogle.com/codejam/round/000000000019fd74/00000000002b3034
    '''
    def solve(self, patterns):
        # Find the longest prefix
        prefix = []
        for pattern in patterns:
            for i, char in enumerate(pattern):
                if char == '*':
                    break
                elif i < len(prefix):
                    if prefix[i] != char:
                        return '*'
                else:
                    prefix.append(char)
        # Find the longest suffix
        suffix = []
        for pattern in patterns:
            for i, char in enumerate(reversed(pattern)):
                if char == '*':
                    break
                elif i < len(suffix):
                    if suffix[i] != char:
                        return '*'
                else:
                    suffix.append(char)
        # Gather up all middle portions and ignore any internal wild cards
        middles = []
        for pattern in patterns:
            left = 0
            right = len(pattern) - 1
            while pattern[left] != '*':
                left += 1
            while pattern[right] != '*':
                right -= 1
            middle = []
            for i in range(left, right):
                char = pattern[i]
                if char != '*':
                    middle.append(char)
            middles.append(''.join(middle))
        result = ''.join(prefix) + ''.join(middles) + ''.join(reversed(suffix))
        return result
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            pattern_count = int(input())
            patterns = []
            for _ in range(pattern_count):
                pattern = input()
                patterns.append(pattern)
            solution = self.solve(patterns)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

class Solver: # 2020.1A.2
    '''
    2020.1A.2 (Pascal Walk)
    https://codingcompetitions.withgoogle.com/codejam/round/000000000019fd74/00000000002b1353
    '''
    
    def get_next_row(self, pascal_row: list) -> list:
        # This:         [1, 4, 6, 4, 1]
        # Becomes this: [1, 5, 10, 10, 5, 1]
        next_row = [1] * (len(pascal_row) + 1)
        for i in range(1, len(next_row) - 1):
            next_row[i] = pascal_row[i - 1] + pascal_row[i]
        result = next_row
        return result
    
    def get_prev_row(self, pascal_row: list) -> list:
        # This:         [1, 5, 10, 10, 5, 1]
        # Becomes this: [1, 4, 6, 4, 1]
        if len(pascal_row) < 2:
            return pascal_row
        prev_row = [1] * (len(pascal_row) - 1)
        for i in range(1, len(pascal_row) - 1):
            prev_row[i] = pascal_row[i] - prev_row[i - 1]
        result = prev_row
        return result

    def solve(self, target):
        # Go down the lefthand-center column of the triangle until you are halfway there
        # Then proceed up the righthand-center of the column, zip to the righthand edge
        # if you need to
        result = target
        return result
    
    def run_tests(self):
        pascal_row = [1]
        # Descend
        pascal_row = self.get_next_row(pascal_row)
        assert pascal_row == [1, 1]
        pascal_row = self.get_next_row(pascal_row)
        assert pascal_row == [1, 2, 1]
        pascal_row = self.get_next_row(pascal_row)
        assert pascal_row == [1, 3, 3, 1]
        pascal_row = self.get_next_row(pascal_row)
        assert pascal_row == [1, 4, 6, 4, 1]
        pascal_row = self.get_next_row(pascal_row)
        assert pascal_row == [1, 5, 10, 10, 5, 1]
        pascal_row = self.get_next_row(pascal_row)
        assert pascal_row == [1, 6, 15, 20, 15, 6, 1]
        pascal_row = self.get_next_row(pascal_row)
        assert pascal_row == [1, 7, 21, 35, 35, 21, 7, 1]
        pascal_row = self.get_next_row(pascal_row)
        assert pascal_row == [1, 8, 28, 56, 70, 56, 28, 8, 1]
        # Ascend
        pascal_row = self.get_prev_row(pascal_row)
        print(pascal_row)
        assert pascal_row == [1, 7, 21, 35, 35, 21, 7, 1]
        pascal_row = self.get_prev_row(pascal_row)
        assert pascal_row == [1, 6, 15, 20, 15, 6, 1]
        pascal_row = self.get_prev_row(pascal_row)
        assert pascal_row == [1, 5, 10, 10, 5, 1]
        pascal_row = self.get_prev_row(pascal_row)
        assert pascal_row == [1, 4, 6, 4, 1]
        pascal_row = self.get_prev_row(pascal_row)
        assert pascal_row == [1, 3, 3, 1]
        pascal_row = self.get_prev_row(pascal_row)
        assert pascal_row == [1, 2, 1]
        pascal_row = self.get_prev_row(pascal_row)
        assert pascal_row == [1, 1]
        pascal_row = self.get_prev_row(pascal_row)
        assert pascal_row == [1]
        pascal_row = self.get_prev_row(pascal_row)
        assert pascal_row == [1]
        print('All tests pass')
    
    def main(self):
        # Max number of steps in the walk is 500
        # Max target in test set 3 is 1_000_000_000
        self.run_tests()
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            target = int(input())
            solution = self.solve(target)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

class SolverA:
    def solve(self, raw_input):
        result = len(raw_input)
        return result
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            raw_input = input()
            solution = self.solve(raw_input)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

class SolverB:
    def solve(self, raw_input):
        result = len(raw_input)
        return result
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            raw_input = input()
            solution = self.solve(raw_input)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

class SolverC:
    def solve(self, raw_input):
        result = len(raw_input)
        return result
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            raw_input = input()
            solution = self.solve(raw_input)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

class Template:
    '''
    2020.Q.2
    https://codingcompetitions.withgoogle.com/codejam/round/000000000019fd27/0000000000209a9f
    '''
    def solve(self, raw_input):
        result = len(raw_input)
        return result
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            raw_input = input()
            solution = self.solve(raw_input)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

if __name__ == '__main__':
    '''
    Usage
    python GoogleCodeJam2020.py 2020.1A.1 < inputs/PatternMatching.in
    python GoogleCodeJam2020.py Solver < inputs/Solver.in
    '''
    solvers = {
        '2020.Q.1': (Vestigium, 'Vestigium'),
        '2020.Q.2': (NestingDepth, 'Nesting Depth'),
        '2020.Q.3': (ParentingPartneringReturns, 'Parenting Partnering Returns'),
        # '2020.Q.4': (ESAbATAd, 'ESAbATAd'),
        '2020.Q.5': (Indicium__Incomplete, 'Indicium'),
        '2020.1A.1': (PatternMatching, 'Pattern Matching'),
        '2020.1A.2': (Solver, 'PascalWalk'),
        # '2020.1A.3': (SolverC, 'Problem2020_1A_3'),
        # '2020.1A.4': (Problem2020_1A_4, 'Problem2020_1A_4'),
        # '2020.1B.1': (Expogo, 'Expogo'),
        # '2020.1B.2': (BlindfoldedBullseye, 'Blindfolded Bullseye'),
        # '2020.1B.3': (JoinTheRanks, 'Join the Ranks'),
        # '2020.1C.1': (OverexcitedFan, 'Overexcited Fan'),
        # '2020.1C.2': (Overrandomized, 'Overrandomized'),
        # '2020.1C.3': (OversizedPancakeChoppers, 'Oversized Pancake Choppers'),
        # '2020.2.1': (IncrementalHouseOfPancakes, 'Incremental House of Pancakes'),
        # '2020.2.2': (SecurityUpdate, 'Security Update'),
        # '2020.2.3': (WormholeInOne, 'Wormhole in One'),
        # '2020.2.4': (EmacsPlusPlus, 'Emacs++'),
        'Solver': (Solver, 'Solver'),
        }
    parser = argparse.ArgumentParser()
    parser.add_argument('problem', help='Solve for a given problem', type=str)
    args = parser.parse_args()
    problem = args.problem
    solver = solvers[problem][0]()
    print(f'Solution for "{problem}" ({solvers[problem][1]})')
    solution = solver.main()
    #print(f'  Answer:', solution)

# Template for Submission page
'''
import collections

class Solver:
    def solve(self, raw_input):
        result = len(raw_input)
        return result
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            raw_input = input()
            solution = self.solve(raw_input)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

if __name__ == '__main__':
    solver = Solver()
    solver.main()
'''

# Usage for Interactive Problems
'''
-- python judges/DatBae.py 0 python solvers/DatBae.py

import os
os.system('python filename.py')
'''