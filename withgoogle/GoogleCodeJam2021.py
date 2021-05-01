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

class SolverA: # 2021.1C.A
    '''
    2021.1C.A
    https://codingcompetitions.withgoogle.com/codejam/???
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

class SolverB: # 2021.1C.B
    '''
    2021.1C.C
    https://codingcompetitions.withgoogle.com/codejam/???
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

class SolverC: # 2021.1C.C
    '''
    2021.1C.C
    https://codingcompetitions.withgoogle.com/codejam/???
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

class SolverD: # 2021.1C.D
    '''
    2021.1C.D
    https://codingcompetitions.withgoogle.com/codejam/???
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

class SolverE: # 2021.1C.E
    '''
    2021.1C.E
    https://codingcompetitions.withgoogle.com/codejam/???
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

class BrokenClock: # 2021.1B.A
    '''
    2021.1B.A
    https://codingcompetitions.withgoogle.com/codejam/round/0000000000435baf/00000000007ae694
    1 tick is equal to (1 / 12) * (10 ** −10) degrees
    Hours hand rotates exactly 1 tick each nanosecond
    Minutes hand rotates exactly 12 ticks each nanosecond
    Seconds hand rotates exactly 720 ticks each nanosecond
    There are 10 ** 9 nanoseconds in a second
    OUTPUT: "hours minutes seconds nanoseconds" since midnight
    '''
    HOUR_HAND_TICKS_PER_NANOSEC = H_NANO = 1
    MINUTE_HAND_TICKS_PER_NANOSEC = M_NANO = 12
    SECOND_HAND_TICKS_PER_NANOSEC = S_NANO = 720
    NANOSECS_PER_SECOND = 10 ** 9
    SECONDS_PER_MINUTE = 60
    NANOSECS_PER_MINUTE = NANOSECS_PER_SECOND * SECONDS_PER_MINUTE
    MINUTES_PER_HOUR = 60
    NANOSECS_PER_HOUR = NANOSECS_PER_MINUTE * MINUTES_PER_HOUR
    HOURS_PER_CYCLE = 12
    NANOSECS_PER_CYCLE = NANOSECS_PER_HOUR * HOURS_PER_CYCLE
    
    def canonicalize(self, A, B, C):
        canonical_form = None
        for a, b, c in (
            (A - A, B - A, C - A),
            (A - B, B - B, C - B),
            (A - C, B - C, C - C),
        ):
            a %= self.NANOSECS_PER_CYCLE
            b %= self.NANOSECS_PER_CYCLE
            c %= self.NANOSECS_PER_CYCLE
            candidate = tuple(sorted([a, b, c]))
            if canonical_form is None or candidate < canonical_form:
                canonical_form = candidate
        result = canonical_form
        return result
    
    def solve(self, A, B, C):
        clocks = {}
        for ticks in range(0, self.NANOSECS_PER_CYCLE, self.NANOSECS_PER_SECOND):
            h = (ticks // self.NANOSECS_PER_HOUR) % self.HOURS_PER_CYCLE
            m = (ticks // self.NANOSECS_PER_MINUTE) % self.MINUTES_PER_HOUR
            s = (ticks // self.NANOSECS_PER_SECOND) % self.SECONDS_PER_MINUTE
            clock = (h, m, s, ticks % self.NANOSECS_PER_SECOND)
            h_ticks = (self.H_NANO * ticks) % self.NANOSECS_PER_CYCLE
            m_ticks = (self.M_NANO * ticks) % self.NANOSECS_PER_CYCLE
            s_ticks = (self.S_NANO * ticks) % self.NANOSECS_PER_CYCLE
            canonical_form = self.canonicalize(h_ticks, m_ticks, s_ticks)
            clocks[canonical_form] = clock
        canonical_form = self.canonicalize(A, B, C)
        clock = clocks[canonical_form]
        result = ' '.join(map(str, clock))
        return result
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            A, B, C = tuple(map(int, input().split(' ')))
            solution = self.solve(A, B, C)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

class Subtransmutation: # 2021.1B.B
    '''
    2021.1B.B
    https://codingcompetitions.withgoogle.com/codejam/round/0000000000435baf/00000000007ae4aa
    For some fixed numbers A and B, with A < B, you can take one unit of metal i
    and destroy it to create one unit of metal (i − A) and one unit of metal (i−B)
    If either of those integers is not positive, that specific unit is not created
    In particular, if i ≤ A, the spell simply destroys the unit and creates nothing.
    If A < i ≤ B the spell destroys the unit and creates only a single unit of metal (i−A)
    A single unit of metal represented by the smallest possible integer that is 
    sufficient to complete your task, or say that there is no such metal.
    4 -> (3, 2)
    4 -> (3, 1)
    4 -> (2, 1)
    A is 1 and B is 2 for Test Set 1
    A and B are between 1 and 20 for Test Set 2

    1 2 3 4
    -------
    1 2 0 0
    0 1 1 0
    0 0 0 1

    1 2 3 4 5 6
    -----------
    0 0 0 0 0 1
    0 0 0 1 1 0
    0 1 1 0 1 0
    1 2 0 0 1 0
    2 1 0 0 1 0

    2 x x x 1 x
    2 1 x x 1 x
    1 0 1 x 1 x
    
    '''
    def solve(self, metals, A, B):
        result = len(metals)
        return result
    
    def main(self):
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            _, A, B = tuple(map(int, input().split(' ')))
            metal_counts = tuple(map(int, input().split(' ')))
            metals = {}
            for metal_id, count in enumerate(metal_counts, start=1):
                if count > 0:
                    metals[metal_id] = count
            solution = self.solve(metals, A, B)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

class AppendSort: # 2021.1A.A
    '''
    2021.1A.A
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

class PrimeTime: # 2021.1A.B
    '''
    2021.1A.B
    https://codingcompetitions.withgoogle.com/codejam/round/000000000043585d/00000000007543d8
    Test set 1 max card count is 10
    Test set 2 max card count is 100
    Test set 3 max card count is 10 ^ 15

    Start with the largest product that is <= the maximum sum,
    work our way down until we find a match
    '''
    def solve(self, deck: list, primes: list):
        max_score = 0
        total_deck_value = sum(prime * count for prime, count in deck.items())
        for product in range(total_deck_value, -1, -1):
            remaining_product = product
            chosen_primes = []
            valid_ind = True
            i = 0
            while remaining_product > 1:
                if i >= len(primes):
                  valid_ind = False
                  break
                candidate = primes[i]
                while remaining_product % candidate == 0:
                    if candidate not in deck or deck[candidate] < 1:
                        valid_ind = False
                        break
                    deck[candidate] -= 1
                    chosen_primes.append(candidate)
                    remaining_product //= candidate
                i += 1
            if valid_ind:
                if sum(prime * count for prime, count in deck.items()) == product:
                    max_score = product
                    break
            for prime in chosen_primes:
                deck[prime] += 1
        result = max_score
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
    
    def get_primes(self, max_prime: int) -> List[int]:
        prime_inds = [True] * (max_prime + 1)
        prime_inds[0] = False
        prime_inds[1] = False
        for num in range(2, int(math.sqrt(max_prime)) + 1):
            if prime_inds[num]:
                for i in range(num + num, max_prime + 1, num):
                    prime_inds[i] = False
        primes = []
        for num, prime_ind in enumerate(prime_inds):
            if prime_ind:
                primes.append(num)
        result = primes
        return result
    
    def random_test(self, primes: list):
        max_cards = 10
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
            if zero_check is True and sanity_check == 0:
                break
            elif zero_check is not True and sanity_check != 0:
                break
        solution = self.solve(random_deck, primes)
        try:
            assert solution == sanity_check
        except AssertionError:
            print('Expected:', sanity_check)
            print('Observed:', solution)
            print('Deck:', random_deck)
            raise AssertionError
    
    def main(self):
        primes = self.get_primes(500)
        self.solve({5: 1, 431: 2, 401: 1, 293: 1, 89: 1, 137: 1}, primes)
        for _ in range(100_000):
            self.random_test(primes)
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            M = int(input())
            deck = {}
            for _ in range(M):
                prime, count = tuple(map(int, input().split(' ')))
                deck[prime] = count
            solution = self.solve(deck, primes)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

class HackedExamNotStarted: # 2021.1A.C
    '''
    2021.1A.C
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

class CheatingDetection: # 2021.Q.E
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
    python GoogleCodeJam2021.py 2021.1C.A < inputs/SolverA.in
    '''
    solvers = {
        '2021.Q.A': (Reversort, 'Reversort'),
        '2021.Q.B': (MoonsAndUmbrellas, 'Moons and Umbrellas'),
        '2021.Q.C': (ReversortEngineering, 'Reversort Engineering'),
        # '2021.Q.D': (MedianSort, 'Median Sort'),
        '2021.Q.E': (CheatingDetection, 'Cheating Detection'),
        '2021.1A.A': (AppendSort, 'Append Sort'),
        '2021.1A.B': (PrimeTime, 'Prime Time'),
        '2021.1A.C': (HackedExamNotStarted, 'Hacked Exam'),
        '2021.1B.A': (BrokenClock, 'Broken Clock'),
        '2021.1B.B': (Subtransmutation, 'Subtransmutation'),
        # '2021.1B.C': (DigitBlocks, 'Digit Blocks'),
        '2021.1C.A': (SolverA, 'SolverA'),
        '2021.1C.B': (SolverB, 'SolverB'),
        '2021.1C.C': (SolverC, 'SolverC'),
        '2021.1C.D': (SolverD, 'SolverD'),
        '2021.1C.E': (SolverE, 'SolverE'),
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