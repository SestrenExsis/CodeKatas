'''
Created on April 1, 2022

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

class SolverPunchedCards: # Punched Cards
    '''
    Punched Cards
    https://codingcompetitions.withgoogle.com/codejam/round/0000000000876ff1/0000000000a4621b    '''
    def solve(self, rows: int, cols: int):
        punchcard = [
            '+' + '-+' * cols,
        ]
        for _ in range(rows):
            punchcard.append('|' + '.|' * cols)
            punchcard.append('+' + '-+' * cols)
        punchcard[0] = '..' + punchcard[0][2:]
        punchcard[1] = '..' + punchcard[1][2:]
        result = punchcard
        return result
    
    def main(self):
        '''
        For Test Set 1:
            1 <= T <= 81
            1 <= R <= 10
            1 <= C <= 10
        '''
        T = int(input())
        output = []
        for test_id in range(1, T + 1):
            R, C = tuple(map(int, input().split(' ')))
            solution = self.solve(R, C)
            print('Case #{}:'.format(test_id))
            for row_data in solution:
                print(row_data)
        return output

class Solver3DPrinting: # 3D Printing
    '''
    3D Printing
    https://codingcompetitions.withgoogle.com/codejam/round/0000000000876ff1/0000000000a4672b
    '''
    def solve(self, printer1, printer2, printer3):
        C = min(printer1[0], printer2[0], printer3[0])
        M = min(printer1[1], printer2[1], printer3[1])
        Y = min(printer1[2], printer2[2], printer3[2])
        K = min(printer1[3], printer2[3], printer3[3])
        inks_used = [0, 0, 0, 0]
        inks_available = [C, M, Y, K]
        ink_needed = 10 ** 6
        for ink_id in range(len(inks_available)):
            used = min(ink_needed, inks_available[ink_id])
            ink_needed -= used
            inks_used[ink_id] = used
        ink_needed = 10 ** 6
        result = 'IMPOSSIBLE'
        if sum(inks_used) == ink_needed:
            result = ' '.join(map(str, inks_used))
        return result
    
    def main(self):
        '''
        10 ** 6 units to print a letter
        Need 3 letters, all the same color
        1 <= T <= 100
        For Test Set 1:
        0 <= Ci <= 10 ** 6, for all i
        0 <= Mi <= 10 ** 6, for all i
        0 <= Yi <= 10 ** 6, for all i
        0 <= Ki <= 10 ** 6, for all i
        '''
        T = int(input())
        output = []
        for test_id in range(1, T + 1):
            P1 = tuple(map(int, input().split(' ')))
            P2 = tuple(map(int, input().split(' ')))
            P3 = tuple(map(int, input().split(' ')))
            solution = self.solve(P1, P2, P3)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

class Solverd1000000: # d1000000
    '''
    d1000000
    https://codingcompetitions.withgoogle.com/codejam/round/0000000000876ff1/0000000000a46471
    '''
    def solve(self, dice: list):
        '''
        Start with the smallest dice while forming your straight
        Once the smallest die is too small to continue the straight,
        pick the smallest die that is big enough to continue.
        4 4 4 5 6 7 7 7 10 10
        1 2 3 4 5 6 7 x 8  9
        '''
        straight = 1
        for sides in sorted(dice):
            if sides >= straight:
                straight += 1
        result = straight - 1
        return result
    
    def main(self):
        '''
        1 <= T <= 100
        For Test Set 1:
            1 <= N <= 10
            1 <= Si <= 20, for all i
        For Test Set 2:
            1 <= N <= 10 ** 5
            1 <= Si <= 10 ** 6, for all i
        '''
        T = int(input())
        output = []
        for test_id in range(1, T + 1):
            N = int(input())
            S = map(int, input().split(' '))
            solution = self.solve(S)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output
    
class SolverChainReactions: # Chain Reactions
    '''
    Chain Reactions
    https://codingcompetitions.withgoogle.com/codejam/round/0000000000876ff1/0000000000a45ef7
    '''
    def solve(self, N: int, fun_factors: tuple, connections: tuple):
        initiators = set(range(1, N + 1)) - set(connections)
        max_total_fun = 0
        for orderings in itertools.permutations(initiators):
            F = list(fun_factors)
            total_fun = 0
            for initiator in orderings:
                fun = 0
                id = initiator
                while id > 0:
                    fun = max(fun, F[id - 1])
                    F[id - 1] = 0
                    id = connections[id - 1]
                total_fun += fun
            max_total_fun = max(max_total_fun, total_fun)
        result = max_total_fun
        return result
    
    def main(self):
        '''
        Limits
        Memory limit: 1 GB
        1 ≤ T ≤ 100
        1 ≤ Fi ≤ 109
        0 ≤ Pi ≤ i - 1, for all i

        Test Set 1 (Visible Verdict)
        Time limit: 5 seconds
        1 ≤ N ≤10

        Test Set 2 (Visible Verdict)
        Time limit: 5 seconds
        1 ≤ N ≤ 1000

        Test Set 3 (Hidden Verdict)
        Time limit: 10 seconds
        1 ≤ N ≤ 100000
        '''
        T = int(input())
        output = []
        for test_id in range(1, T + 1):
            N = int(input())
            F = tuple(map(int, input().split(' ')))
            P = tuple(map(int, input().split(' ')))
            solution = self.solve(N, F, P)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

class SolverDoubleOrOneThing: # Double or One Thing
    '''
    Solver 1A.1
    https://codingcompetitions.withgoogle.com/codejam/round/
    '''
    def solve(self, seed_word):
        min_word = seed_word
        words = [(seed_word, 0, 0)]
        while len(words) > 0:
            word, progress, index = heapq.heappop(words)
            if word < min_word:
                min_word = word
            if progress < len(seed_word):
                word2 = word[:index] + seed_word[progress] + word[index:]
                if word <= min_word:
                    heapq.heappush(words, (word, progress + 1, index + 1))
                if word2 <= min_word:
                    heapq.heappush(words, (word2, progress + 1, index + 2))
        result = min_word
        return result
    
    def main(self):
        T = int(input())
        output = []
        for test_id in range(1, T + 1):
            word = input()
            solution = self.solve(word)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

class SolverWeightlifting: # Weightlifting
    '''
    Solver 1A.3
    https://codingcompetitions.withgoogle.com/codejam/round/
    '''
    def solve_slowly(self, exercises):
        # assert 1 <= len(exercises[0]) <= 3
        min_cost = float('inf')
        W = len(exercises[0])
        seen = set()
        work = collections.deque() # (cost, progress, stack)
        work.appendleft((0, 0, tuple()))
        while len(work) > 0:
            N = len(work)
            for _ in range(N):
                cost, progress, stack = work.pop()
                # print(N, cost, progress, stack)
                exercise = exercises[progress]
                needed = list(exercise)
                for weight in stack:
                    needed[weight] -= 1
                if len(set(needed)) == 1 and sum(needed) == 0:
                    progress += 1
                if progress == len(exercises):
                    min_cost = cost + len(stack)
                    while len(work) > 0:
                        work.pop()
                    break
                if (progress, stack) in seen:
                    continue
                seen.add((progress, stack))
                if len(stack) > 0:
                    next_stack = list(stack)
                    next_stack.pop()
                    work.appendleft((cost + 1, progress, tuple(next_stack)))
                exercise = exercises[progress]
                needed = list(exercise)
                for weight in stack:
                    needed[weight] -= 1
                for weight in range(W):
                    if needed[weight] < 1:
                        continue
                    next_stack = list(stack)
                    next_stack.append(weight)
                    work.appendleft((cost + 1, progress, tuple(next_stack)))
        result = min_cost
        return result
    
    def main(self):
        T = int(input())
        output = []
        for test_id in range(1, T + 1):
            E, W = tuple(map(int, input().split(' ')))
            exercises = []
            for _ in range(E):
                exercise = tuple(map(int, input().split(' ')))
                exercises.append(exercise)
            solution = self.solve_slowly(exercises)
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
    python GoogleCodeJam2022.py A < inputs/SolverA.in
    '''
    solvers = {
        'Q.1': (SolverPunchedCards, 'Punched Cards'),
        'Q.2': (Solver3DPrinting, '3D Printing'),
        'Q.3': (Solverd1000000, 'd1000000'),
        'Q.4': (SolverChainReactions, 'Chain Reactions'),
        '1A.1': (SolverDoubleOrOneThing, 'Double or One Thing'),
        '1A.3': (SolverWeightlifting, 'Weightlifting'),
        }
    parser = argparse.ArgumentParser()
    parser.add_argument('problem', help='Solve for a given problem', type=str)
    args = parser.parse_args()
    problem = args.problem
    solver = solvers[problem][0]()
    print(f'Solution for "{problem}" ({solvers[problem][1]})')
    solution = solver.main()

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