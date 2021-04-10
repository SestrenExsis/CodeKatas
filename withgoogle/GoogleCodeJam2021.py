'''
Created on March 26, 2021

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

class AppendSort: # 2021.Q.A
    '''
    2021.Q.A
    https://codingcompetitions.withgoogle.com/codejam/round/000000000043585d/00000000007549e5
    When we append something, we might as well append the smallest number it takes to make it bigger
    Either the current number is already big enough, in which case we do nothing, or ...
    it needs to be made
    N[i] = smallest number bigger than N[i - 1] that starts with the digits in N[i]

    '''
    def solve(self, nums):
        modded_nums = nums[::]
        prev_num = ''
        for i in range(1, len(modded_nums)):
            str_num = str(nums[i])
            start = max(nums[i], modded_nums[i - 1])
            offset = 0
            while True:
                if start + offset > modded_nums[i - 1] and str(start + offset).startswith(str_num):
                    break
                offset += 1
            modded_nums[i] = start + offset
        result = sum(len(str(modded_nums[i])) - len(str(nums[i])) for i in range(len(nums)))
        return result
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            N = int(input())
            nums = list(map(int, input().split(' ')))
            solution = self.solve(nums)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

class SolverB: # 2021.Q.B
    '''
    2021.Q.B
    https://codingcompetitions.withgoogle.com/codejam/round/000000000043585d/00000000007543d8
    Test set 1 max sum is 10
    Test set 2 max sum is 100
    Test set 3 max sum is 10 ** 15

    Start with the largest product that is <= the maximum sum,
    work our way down until we find a match
    '''
    def solve(self, deck):
        sanity_check = self.bruteforce(deck)
        max_score = 0
        total_deck_value = sum(prime * count for prime, count in deck.items())
        for product in range(total_deck_value, -1, -1):
            remaining_product = product
            primes = []
            valid_ind = True
            for candidate in range(2, 500):
                if remaining_product == 1:
                    break
                if remaining_product % candidate == 0:
                    while remaining_product % candidate == 0:
                        if candidate not in deck or deck[candidate] < 1:
                            valid_ind = False
                            break
                        deck[candidate] -= 1
                        primes.append(candidate)
                        remaining_product /= candidate
            if valid_ind:
                if sum(prime * count for prime, count in deck.items()) == product:
                    max_score = product
                    break
            for prime in primes:
                deck[prime] += 1
        result = max_score
        assert result == sanity_check
        return result
    
    def bruteforce(self, deck):
        cards = []
        for prime, count in deck.items():
            for _ in range(count):
                cards.append(prime)
        total_deck_value = sum(cards)
        max_score = 0
        for N in range(1, len(cards)):
            for arrangement in itertools.combinations(cards, N):
                total_sum = total_deck_value - sum(arrangement)
                product = 1
                for card in arrangement:
                    product *= card
                if total_sum == product:
                    score = total_sum
                    if score > max_score:
                        max_score = score
        result = max_score
        return result
    
    def random_test(self):
        max_deck_value = 10
        primes = [2, 3, 5, 7]
        max_cards = max_deck_value // 2
        # primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        random_deck = {}
        sanity_check = 0
        zero_check = random.random() < 0.5
        while True:
            N = random.randint(1, max_cards)
            random_deck = {}
            total_deck_value = 0
            for _ in range(N):
                prime = random.choice(primes)
                total_deck_value += prime
                if prime not in random_deck:
                    random_deck[prime] = 0
                random_deck[prime] += 1
            sanity_check = self.bruteforce(random_deck)
            if total_deck_value <= max_deck_value:
                if zero_check is True and sanity_check == 0:
                    break
                elif zero_check is not True and sanity_check != 0:
                    break
        print(sanity_check, random_deck)
        solution = self.solve(random_deck)
        assert solution == sanity_check
    
    def main(self):
        # for _ in range(100_000):
        #     self.random_test()
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            M = int(input())
            deck = {}
            for _ in range(M):
                prime, count = tuple(map(int, input().split(' ')))
                deck[prime] = count
            solution = self.solve(deck)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

class SolverC: # 2021.Q.C
    '''
    2021.Q.C
    https://codingcompetitions.withgoogle.com/codejam/round/000000000043585d/0000000000754750
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

class SolverD: # 2021.Q.D
    '''
    2021.Q.D
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

class SolverE: # 2021.Q.E
    '''
    2021.Q.E
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

class Reversort: # 2021.Q.A
    '''
    2021.Q.A
    https://codingcompetitions.withgoogle.com/codejam/round/000000000043580a/00000000006d0a5c
    '''
    def solve(self, arr):
        total_cost = 0
        for left in range(len(arr) - 1):
            min_num = arr[left]
            right = left
            for i in range(left, len(arr)):
                if arr[i] < min_num:
                    min_num = arr[i]
                    right = i
            arr = arr[:left] + arr[left:right + 1][::-1] + arr[right + 1:]
            total_cost += right - left + 1
        result = total_cost
        return result
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            element_count = int(input())
            elements = list(map(int, input().split(' ')))
            solution = self.solve(elements)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

class MoonsAndUmbrellas: # 2021.Q.B
    '''
    2021.Q.B
    https://codingcompetitions.withgoogle.com/codejam/round/000000000043580a/00000000006d1145
    '''
    def solve(self, costs: dict, mural: str) -> int:
        cost_c = 0
        cost_j = 0
        for i in range(len(mural)):
            next_cost_c = min(cost_c, cost_j + costs['JC'])
            next_cost_j = min(cost_j, cost_c + costs['CJ'])
            char = mural[i]
            if char in ('C', '?'):
                cost_c = next_cost_c
            else:
                cost_c = float('inf')
            if char in ('J', '?'):
                cost_j = next_cost_j
            else:
                cost_j = float('inf')
        result = min(cost_c, cost_j)
        return result
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            cost1, cost2, mural = input().split(' ')
            costs = {
                'CJ': int(cost1),
                'JC': int(cost2),
            }
            solution = self.solve(costs, mural)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

class ReversortEngineering: # 2021.Q.C
    '''
    2021.Q.C
    https://codingcompetitions.withgoogle.com/codejam/round/000000000043580a/00000000006d12d7
    '''
    arrays = {}

    def get_cost(self, arr):
        cost = 0
        for left in range(len(arr) - 1):
            min_num = arr[left]
            right = left
            for i in range(left, len(arr)):
                if arr[i] < min_num:
                    min_num = arr[i]
                    right = i
            arr = arr[:left] + arr[left:right + 1][::-1] + arr[right + 1:]
            cost += right - left + 1
        result = cost
        return result
    
    def populate_arrays(self, size: int=7):
        elements = range(1, size + 1)
        for array in itertools.permutations(elements):
            cost = self.get_cost(array)
            if (size, cost) not in self.arrays:
                self.arrays[(size, cost)] = array

    def solve(self, size: int, target_cost: int):
        if (size, size - 1) not in self.arrays:
            self.populate_arrays(size)
        result = None
        if (size, target_cost) in self.arrays:
            result = self.arrays[(size, target_cost)]
        return result
    
    def main(self):
        MAX_ARRAY_SIZE = 7
        for size in range(1, MAX_ARRAY_SIZE + 1):
            self.populate_arrays(size)
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            size, target_cost = tuple(map(int, input().split(' ')))
            solution = 'IMPOSSIBLE'
            array = self.solve(size, target_cost)
            if array is not None:
                solution = ' '.join(map(str, array))
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

class CheatingDetection:
    '''
    2021.Q.E
    https://codingcompetitions.withgoogle.com/codejam/round/000000000043580a/00000000006d1155
    
    100 players
    10_000 questions
    question difficulties and player skills are random with range [-3.00, 3.00]
    f(x) = 1 / (1 + e ^ -x), where x = skill - difficulty

    The cheater is the person whose odds of getting a hard question right 
    is least influenced by increases in a question's difficulty
    '''
    player_count = 100
    question_count = 10_000

    def solve(self, player_answers: list, partition_threshold: float):
        difficulties = [0] * self.question_count
        for player_id, answers in enumerate(player_answers):
            for question_id, answer in enumerate(answers):
                if answer == '0':
                    difficulties[question_id] += 1
        # Partition the questions into two groups: easy and hard
        questions = []
        max_partition_size = int(partition_threshold * self.question_count)
        easy_questions = []
        hard_questions = []
        for question_id, difficulty in enumerate(difficulties):
            if len(easy_questions) > max_partition_size:
                heapq.heappushpop(easy_questions, (-difficulty, question_id))
            else:
                heapq.heappush(easy_questions, (-difficulty, question_id))
            if len(hard_questions) > max_partition_size:
                heapq.heappushpop(hard_questions, (difficulty, question_id))
            else:
                heapq.heappush(hard_questions, (difficulty, question_id))
        quiz = []
        for player_id in range(self.player_count):
            easy_questions_correct = 0
            for _, question_id in easy_questions:
                if player_answers[player_id][question_id] == '1':
                    easy_questions_correct += 1
            hard_questions_correct = 0
            for _, question_id in hard_questions:
                if player_answers[player_id][question_id] == '1':
                    hard_questions_correct += 1
            quiz.append((easy_questions_correct, hard_questions_correct))
        cheater = None
        lowest_diff = float('inf')
        for player_id in range(self.player_count):
            easy_questions_correct, hard_questions_correct = quiz[player_id]
            diff = easy_questions_correct - hard_questions_correct
            if diff < lowest_diff:
                cheater = player_id
                lowest_diff = diff
        result = cheater + 1
        return result
    
    def main(self):
        test_count = int(input())
        threshold = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            player_answers = []
            for i in range(self.player_count):
                player_answers.append(input())
            solution = self.solve(player_answers, 0.10)
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
    python GoogleCodeJam2021.py 2021.1A.A < inputs/SolverA.in
    '''
    solvers = {
        '2021.Q.A': (Reversort, 'Reversort'),
        '2021.Q.B': (MoonsAndUmbrellas, 'Moons and Umbrellas'),
        '2021.Q.C': (ReversortEngineering, 'Reversort Engineering'),
        # '2021.Q.D': (MedianSort, 'Median Sort'),
        '2021.Q.E': (CheatingDetection, 'Cheating Detection'),
        '2021.1A.A': (AppendSort, 'Append Sort'),
        '2021.1A.B': (SolverB, 'Prime Time'),
        '2021.1A.C': (SolverC, 'Hacked Exam'),
        # '2021.1A.D': (SolverD, '???'),
        # '2021.1A.E': (SolverE, '???'),
        # '2021.1B.A': (SolverA, '???'),
        # '2021.1B.B': (SolverB, '???'),
        # '2021.1B.C': (SolverC, '???'),
        # '2021.1B.D': (SolverD, '???'),
        # '2021.1B.E': (SolverE, '???'),
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

# Usage for Interactive Problems
'''
-- python judges/DatBae.py 0 python solvers/DatBae.py

import os
os.system('python filename.py')
'''