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

class ClosestPick: # 2021.1C.A
    '''
    2021.1C.A
    https://codingcompetitions.withgoogle.com/codejam/round/00000000004362d7/00000000007c0f00
    '''
    def get_hits(self, unsold_tickets, ticket_count, a, b):
        hits = 0
        for start, end in unsold_tickets:
            if any((
                start == 1 and end in (a, b),
                end == ticket_count and start in (a, b),
                start in (a, b) and end in (a, b),
            )):
                hits += end - start + 1
            elif start in (a, b) or end in (a, b):
                hits += 1 + (end - start) // 2
        result = hits
        return result
    
    def solve(self, ticket_count, tickets_purchased):
        # Generate intervals to represent unsold tickets
        unsold_tickets = []
        prev_ticket = 0
        tickets = list(sorted(tickets_purchased))
        for ticket in tickets:
            if ticket != prev_ticket + 1:
                unsold_tickets.append([prev_ticket + 1, ticket - 1])
            prev_ticket = ticket
        if prev_ticket < ticket_count:
            unsold_tickets.append([prev_ticket + 1, ticket_count])
        # The best tickets to buy are adjacent to purchased tickets
        candidates = set()
        for ticket in tickets_purchased:
            if (
                ticket > 1 and
                ticket - 1 not in tickets_purchased
            ):
                candidates.add(ticket - 1)
            if (
                ticket < ticket_count and
                ticket + 1 not in tickets_purchased
            ):
                candidates.add(ticket + 1)
        # Try all combinations from the candidate tickets
        max_odds = 0
        for a in candidates:
            for b in candidates:
                hits = self.get_hits(unsold_tickets, ticket_count, a, b)
                odds = hits / ticket_count
                max_odds = max(max_odds, odds)
        result = max_odds
        return result
    
    def main(self):
        '''
        For Test Set 1:
            1 <= len(tickets_purchased) <= 30
            1 <= ticket_count <= 30
        For Test Set 2:
            1 <= len(tickets_purchased) <= 30
            1 <= ticket_count <= 10 ** 9
        '''
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            _, ticket_count = tuple(map(int, input().split(' ')))
            tickets_purchased = set(tuple(map(int, input().split(' '))))
            solution = self.solve(ticket_count, tickets_purchased)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

class RoaringYears: # 2021.1C.B
    '''
    2021.1C.B
    https://codingcompetitions.withgoogle.com/codejam/round/00000000004362d7/00000000007c0f01
    Given the current year (which may or may not be roaring),
    find what the next roaring year is going to be.
    '''
    roaring_years = []

    def is_roaring_year(self, year):
        string = str(year)
        valid_ind = False
        for i in range(1, len(string) // 2 + 1):
            start = int(string[:i])
            chunks = []
            for j in range(len(string) // i):
                chunks.append(str(start + j))
            if ''.join(chunks) == string:
                valid_ind = True
                break
        result = valid_ind
        return result

    def populate_roaring_years(self, max_digits: int=6):
        roaring_years = []
        for start in range(1, 10 ** (max_digits // 2) + 1):
            if start % 1000000 == 0:
                print(start)
            max_repeat = (max_digits // len(str(start))) + 1
            for repeat in range(2, max_repeat + 1):
                if len(str(start)) * repeat > 2 * max_digits:
                    break
                year = ''.join(
                    map(str, range(start, start + repeat))
                )
                roaring_years.append(int(year))
        self.roaring_years = sorted(roaring_years)
    
    def get_next_roaring_year(self, year):
        '''
        This assumption is wrong, but what's missing?
        192020
        ------
        192 193
        19 20 21
        2 3 4 5 6 7

        99999 --> 100101

         9999 --> 12345
        Given 192020, either:
            1 increasing for LENGTH + 1 if LENGTH is even
            10 ** ((LENGTH + 1) // 2) increasing for 2 if LENGTH is odd
            First digit increasing for LENGTH
            First digit + 1 increasing for LENGTH
            First two digits increasing for LENGTH // 2
            First two digits + 1 increasing for LENGTH // 2
            ...

        Get the smallest of these that are larger than the number
        Guaranteed at least one will succeed (???)

        FIRST HALF IS GREATER OR
        FIRST HALF IS EQUAL AND SECOND HALF IS GREATER
        '''
        STR = str(year)
        LEN = len(STR)
        HALFLEN = (LEN + 1) // 2
        starts = {
            1,
            10 ** HALFLEN,
        }
        for size in range(1, HALFLEN + 1):
            first = int(STR[:size])
            for start in range(first, 10 ** size + 1):
                starts.add(start)
        roaring_years = set()
        for start in starts:
            num = start
            sequence = str(num)
            size = 1
            while size < 2 or len(sequence) < LEN:
                num += 1
                sequence += str(num)
                size += 1
            roaring_years.add(int(sequence))
            sequence += str(num + 1)
            roaring_years.add(int(sequence))
        result = min(
            roaring_year for
            roaring_year in roaring_years if
            roaring_year > year
        )
        return result

    def solve(self, year):
        left = 0
        right = len(self.roaring_years) - 1
        while left < right:
            mid = left + (right - left) // 2
            roaring_year = self.roaring_years[mid]
            if roaring_year <= year:
                left = mid + 1
            else:
                right = mid
        result = self.roaring_years[right]
        return result
    
    def main(self):
        self.populate_roaring_years(6)
        # for year in range(1, 10 ** 5 + 1):
        #     yr1 = min(y for y in self.roaring_years if y > year)
        #     yr2 = self.get_next_roaring_year(year)
        #     if year % 10000 == 0:
        #         print(year)
        #     try:
        #         assert yr1 == yr2
        #     except AssertionError:
        #         print(year, yr1, yr2)
        #         exit()
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            start_year = int(input())
            solution = self.solve(start_year)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

class DoubleOrNOTing: # 2021.1C.C
    '''
    Double or NOTing
    https://codingcompetitions.withgoogle.com/codejam/round/00000000004362d7/00000000007c1139
    '''
    def __init__(self):
        self.masks = [
            (0, 0, 1), # (power, lower, upper)
        ]
    
    def add_mask(self, num: int):
        while self.masks[-1][2] < num:
            power, _, upper = self.masks[-1]
            self.masks.append((
                power + 1,
                upper + 1,
                2 * (upper + 1) - 1,
            ))

    def bnot(self, num: int) -> int:
        self.add_mask(num)
        assert 0 <= num <= self.masks[-1][2]
        mask = 0
        left = 0
        right = len(self.masks)
        while left < right:
            mid = left + (right - left) // 2
            _, lower, upper = self.masks[mid]
            if lower <= num <= upper:
                mask = upper
                break
            elif num < lower:
                right = mid
            elif num > upper:
                left = mid + 1
        notted_num = mask - num
        result = notted_num
        return result

    def solve(self, source: int, target: int, max_tries: int = 1000):
        visits = set()
        work = collections.deque()
        work.append((0, source))
        min_step_count = None
        while len(work) > 0 and min_step_count is None:
            N = len(work)
            for _ in range(N):
                step_count, num = work.pop()
                if (step_count) > max_tries:
                    break
                if num == target:
                    min_step_count = step_count
                    break
                visits.add(num)
                doubled_num = 2 * num
                if doubled_num not in visits:
                    work.appendleft((step_count + 1, doubled_num))
                notted_num = self.bnot(num)
                if notted_num not in visits:
                    work.appendleft((step_count + 1, notted_num))
        result = min_step_count
        return result
    
    def main(self):
        '''
        For Test Set 1:
            1 <= len(S) <= 8
            1 <= len(E) <= 8
        For Test Set 2:
            1 <= len(S) <= 100
            1 <= len(E) <= 100
        '''
        test_count = int(input())
        output = []
        for test_id in range(1, test_count + 1):
            S, E = tuple(input().split(' '))
            source = int('0b' + S, 2)
            target = int('0b' + E, 2)
            solution = self.solve(source, target, 2 * (len(S) + len(E)))
            output_row = 'Case #{}: {}'.format(
                test_id,
                'IMPOSSIBLE' if solution is None else solution,
                )
            output.append(output_row)
            print(output_row)
        return output

class SolverA: # ???
    '''
    ???
    https://codingcompetitions.withgoogle.com/codejam/???
    '''
    def solve(self, raw_input):
        result = len(raw_input)
        return result
    
    def main(self):
        '''
        For Test Set 1:
            1 <= A <= 10
            1 <= B <= 10
        For Test Set 2:
            1 <= A <= 10 ** 9
            1 <= B <= 10 ** 9
        '''
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

class SolverB: # ???
    '''
    ???
    https://codingcompetitions.withgoogle.com/codejam/???
    '''
    def solve(self, raw_input):
        result = len(raw_input)
        return result
    
    def main(self):
        '''
        For Test Set 1:
            1 <= A <= 10
            1 <= B <= 10
        For Test Set 2:
            1 <= A <= 1000
            1 <= B <= 1000
        '''
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

class SolverC: # ???
    '''
    ???
    https://codingcompetitions.withgoogle.com/codejam/???
    '''
    def solve(self, raw_input):
        result = len(raw_input)
        return result
    
    def main(self):
        '''
        For Test Set 1:
            1 <= A <= 10
            1 <= B <= 10
        For Test Set 2:
            1 <= A <= 1000
            1 <= B <= 1000
        '''
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
            _ = int(input()) # num_count
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
            _ = int(input()) # element_count
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

    def solve(self, CJ: int, JC: int, S: str):
        C = float('inf')
        J = float('inf')
        if S[0] in ('C', '?'):
            C = 0
        if S[0] in ('J', '?'):
            J = 0
        F = [(C, J)]
        for i in range(1, len(S)):
            C = float('inf')
            J = float('inf')
            if S[i] in ('C', '?'):
                C = min(
                    C,
                    F[i - 1][0],
                    F[i - 1][1] + JC,
                )
            if S[i] in ('J', '?'):
                J = min(
                    J,
                    F[i - 1][0] + CJ,
                    F[i - 1][1],
                )
            F.append((C, J))
        result = min(F[len(S) - 1])
        return result
    
    def main(self):
        T = int(input())
        output = []
        for test_id in range(1, T + 1):
            raw_input = input()
            parts = raw_input.split(' ')
            X = int(parts[0])
            Y = int(parts[1])
            S = parts[2]
            solution = self.solve(X, Y, S)
            output_row = 'Case #{}: {}'.format(
                test_id,
                solution,
                )
            output.append(output_row)
            print(output_row)
        return output

if __name__ == '__main__':
    solver = MoonsAndUmbrellas()
    solver.main()

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
        _ = int(input()) # threshold
        output = []
        for test_id in range(1, test_count + 1):
            player_answers = []
            for _ in range(self.player_count):
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
        '2021.1C.A': (ClosestPick, 'Closest Pick'),
        '2021.1C.B': (RoaringYears, 'Roaring Years'),
        '2021.1C.C': (DoubleOrNOTing, 'Double or NOTing'),
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