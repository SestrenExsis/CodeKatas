'''
Created on Nov 24, 2020

@author: Sestren
'''
import argparse
import collections
from typing import List, Tuple
    
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
    https://adventofcode.com/2020/day/?
    '''
    def get_parsed_input(self, raw_input_lines: List[str]) -> List[str]:
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, parsed_input: List[str]) -> int:
        result = len(parsed_input)
        return result
    
    def solve2(self, parsed_input: List[str]) -> int:
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

class Day09: # Encoding Error
    '''
    Encoding Error
    https://adventofcode.com/2020/day/9
    '''
    def get_numbers(self, raw_input_lines: List[str]) -> List[int]:
        numbers = []
        for raw_input_line in raw_input_lines:
            numbers.append(int(raw_input_line))
        result = numbers
        return result
    
    def solve(self, numbers: List[int], span: int=25) -> int:
        invalid_num = None
        for i in range(span, len(numbers)):
            complements = set()
            for offset in range(1, span + 1):
                complement = numbers[i] - numbers[i - offset]
                if numbers[i - offset] in complements:
                    break
                complements.add(complement)
            else:
                invalid_num = numbers[i]
                break
        result = invalid_num
        return result
    
    def solve2(self, numbers: List[int], target: int) -> int:
        queue = collections.deque()
        left = 0
        right = 0
        queue.append(numbers[0])
        while True:
            if sum(queue) == target:
                break
            while right < len(numbers) and sum(queue) < target:
                right += 1
                queue.append(numbers[right])
            while left < right and sum(queue) > target:
                left += 1
                queue.popleft()
        result = min(queue) + max(queue)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        numbers = self.get_numbers(raw_input_lines)
        target = self.solve(numbers, 25)
        solutions = (
            target,
            self.solve2(numbers, target),
            )
        result = solutions
        return result

class Day08: # Handheld Halting
    '''
    Handheld Halting
    https://adventofcode.com/2020/day/8
    '''
    def get_instructions(self, raw_input_lines: List[str]) -> List[str]:
        instructions = []
        for raw_input_line in raw_input_lines:
            operation, raw_argument = raw_input_line.split(' ')
            instruction = (operation, int(raw_argument))
            instructions.append(instruction)
        result = instructions
        return result
    
    def solve(self, instructions: List[Tuple[str, int]]) -> int:
        acc = 0
        pc = 0
        seen = set()
        while True:
            if pc in seen or pc >= len(instructions):
                break
            seen.add(pc)
            operation, argument = instructions[pc]
            if operation == 'nop':
                pc += 1
            elif operation == 'acc':
                acc += argument
                pc += 1
            elif operation == 'jmp':
                pc += argument
        result = acc
        return result
    
    def solve2(self, instructions: List[Tuple[str, int]]) -> int:
        swapped = {
            'acc': 'acc',
            'nop': 'jmp',
            'jmp': 'nop',
        }
        acc = 0
        for i in range(len(instructions)):
            if instructions[i][0] not in ('nop', 'jmp'):
                continue
            acc = 0
            pc = 0
            seen = set()
            halted = False
            while True:
                if pc >= len(instructions):
                    halted = True
                    break
                if pc in seen:
                    break
                seen.add(pc)
                operation, argument = instructions[pc]
                if pc == i:
                    operation = swapped[operation]
                if operation == 'nop':
                    pc += 1
                elif operation == 'acc':
                    acc += argument
                    pc += 1
                elif operation == 'jmp':
                    pc += argument
            if halted:
                break
        result = acc
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

class Day07: # Handy Haversacks
    '''
    Handy Haversacks
    https://adventofcode.com/2020/day/7
    '''
    def get_bags(self, raw_input_lines: List[str]) -> List[str]:
        bags = {}
        for raw_input_line in raw_input_lines:
            bag, raw_contents = raw_input_line.split(' bags contain ')
            contents = {}
            if raw_contents != 'no other bags.':
                raw_contents = raw_contents.split(', ')
                for raw_content in raw_contents:
                    (raw_content)
                    words = raw_content.split(' ')
                    contents[' '.join(words[1:3])] = int(words[0])
            bags[bag] = contents
        result = bags
        return result

    def solve(self, bags: List[str]) -> int:
        containing_bags = set()
        work = ['shiny gold']
        while len(work) > 0:
            curr_bag = work.pop()
            for bag, contents in bags.items():
                for content in contents:
                    if content == curr_bag:
                        if bag not in containing_bags:
                            work.append(bag)
                        containing_bags.add(bag)
        result = len(containing_bags)
        return result
    
    def solve2(self, bags: List[str]) -> int:
        bag_count = 0
        work = [(1, 'shiny gold')]
        while len(work) > 0:
            curr_count, curr_bag = work.pop()
            contents = bags[curr_bag]
            for next_bag, next_count in contents.items():
                work.append((curr_count * next_count, next_bag))
            bag_count += curr_count
        result = bag_count - 1
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        bags = self.get_bags(raw_input_lines)
        solutions = (
            self.solve(bags),
            self.solve2(bags),
            )
        result = solutions
        return result

class Day06: # Custom Customs
    '''
    Custom Customs
    https://adventofcode.com/2020/day/6
    '''
    def get_groups(self, raw_input_lines: List[str]) -> List[List[str]]:
        groups = []
        group = []
        for raw_input_line in raw_input_lines:
            if len(raw_input_line) < 1:
                groups.append(group)
                group = []
            else:
                group.append(raw_input_line)
        groups.append(group)
        result = groups
        return result
    
    def solve(self, groups: List[List[str]]) -> int:
        answer_count = 0
        for group in groups:
            answers = set()
            for person in group:
                for answer in person:
                    answers.add(answer)
            answer_count += len(answers)
        result = answer_count
        return result
    
    def solve2(self, groups: List[List[str]]) -> int:
        answer_count = 0
        for group in groups:
            person_count = len(group)
            answers = collections.defaultdict(int)
            for person in group:
                for answer in person:
                    answers[answer] += 1
            for answer, count in answers.items():
                if count == person_count:
                    answer_count += 1
        result = answer_count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        groups = self.get_groups(raw_input_lines)
        solutions = (
            self.solve(groups),
            self.solve2(groups),
            )
        result = solutions
        return result

class Day05: # Binary Boarding
    '''
    Binary Boarding
    https://adventofcode.com/2020/day/5
    '''
    def get_parsed_input(self, raw_input_lines: List[str]) -> List[str]:
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, parsed_input: List[str]) -> int:
        seat_ids = []
        for code in parsed_input:
            row = 0
            row_val = 64
            for char in code[:7]:
                if char == 'B':
                    row += row_val
                row_val //= 2
            col = 0
            col_val = 4
            for char in code[7:]:
                if char == 'R':
                    col += col_val
                col_val //= 2
            seat_id = 8 * row + col
            seat_ids.append(seat_id)
        result = max(seat_ids)
        return result
    
    def solve2(self, parsed_input: List[str]) -> int:
        seat_ids = set()
        for code in parsed_input:
            row = 0
            row_val = 64
            for char in code[:7]:
                if char == 'B':
                    row += row_val
                row_val //= 2
            col = 0
            col_val = 4
            for char in code[7:]:
                if char == 'R':
                    col += col_val
                col_val //= 2
            seat_id = 8 * row + col
            seat_ids.add(seat_id)
        for middle_seat in range(min(seat_ids), max(seat_ids)):
            if all([
                middle_seat - 1 in seat_ids,
                middle_seat not in seat_ids,
                middle_seat + 1 in seat_ids,
            ]):
                result = middle_seat
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

class Day04: # Passport Processing
    '''
    Passport Processing
    https://adventofcode.com/2020/day/4
    '''
    def get_passports(self, raw_input_lines: List[str]) -> List[str]:
        passports = []
        passport = {}
        for raw_input_line in raw_input_lines:
            if len(raw_input_line) < 1:
                passports.append(passport)
                passport = {}
            else:
                pairs = raw_input_line.split(' ')
                for pair in pairs:
                    key, val = pair.split(':')
                    passport[key] = val
        passports.append(passport)
        result = passports
        return result
    
    def solve(self, passports: List[str]) -> int:
        valid_passport_count = 0
        for passport in passports:
            if all(key in passport for key in (
                'byr',
                'iyr',
                'eyr',
                'hgt',
                'hcl',
                'ecl',
                'pid',
                )):
                valid_passport_count += 1
        result = valid_passport_count
        return result
    
    def solve2(self, passports: List[str]) -> int:
        valid_passport_count = 0
        for passport in passports:
            try:
                # Validate birth year
                birth_year = int(passport['byr'])
                assert 1920 <= birth_year <= 2002
                # Validate issue year
                issue_year = int(passport['iyr'])
                assert 2010 <= issue_year <= 2020
                # Validate expiration year
                expiration_year = int(passport['eyr'])
                assert 2020 <= expiration_year <= 2030
                # Validate height
                height_unit = passport['hgt'][-2:]
                height_amt = int(passport['hgt'][:-2])
                assert (
                    height_unit == 'cm' and 150 <= height_amt <= 193 or
                    height_unit == 'in' and 59 <= height_amt <= 76
                    )
                # Validate hair color
                hair_color = passport['hcl']
                assert hair_color[0] == '#'
                assert all(
                    digit in '0123456789abcdef' for 
                    digit in hair_color[1:]
                    )
                # Validate eye color
                eye_color = passport['ecl']
                assert eye_color in (
                    'amb',
                    'blu',
                    'brn',
                    'gry',
                    'grn',
                    'hzl',
                    'oth',
                    )
                # Validate passport ID
                passport_id = passport['pid']
                assert len(passport_id) == 9
                assert all(
                    digit in '0123456789' for 
                    digit in passport_id
                    )
                # All validations passed
                valid_passport_count += 1
            except (AssertionError, KeyError, ValueError):
                continue
        result = valid_passport_count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        passports = self.get_passports(raw_input_lines)
        solutions = (
            self.solve(passports),
            self.solve2(passports),
            )
        result = solutions
        return result

class Day03: # Toboggan Trajectory
    '''
    Toboggan Trajectory
    https://adventofcode.com/2020/day/3
    '''
    def get_parsed_input(self, raw_input_lines: List[str]) -> List[str]:
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def get_hit_count(self, trees: List[str], right: int, down: int) -> int:
        row, col = 0, 0
        hit_count = 0
        rows = len(trees)
        cols = len(trees[0])
        while row < rows:
            if trees[row][col % cols] == '#':
                hit_count += 1
            col += right
            row += down
        result = hit_count
        return result
    
    def solve(self, parsed_input: List[str]) -> int:
        result = self.get_hit_count(parsed_input, 3, 1)
        return result
    
    def solve2(self, parsed_input: List[str]) -> int:
        result = self.get_hit_count(parsed_input, 1, 1)
        result *= self.get_hit_count(parsed_input, 3, 1)
        result *= self.get_hit_count(parsed_input, 5, 1)
        result *= self.get_hit_count(parsed_input, 7, 1)
        result *= self.get_hit_count(parsed_input, 1, 2)
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

class Day02: # Password Philosophy
    '''
    Password Philosophy
    https://adventofcode.com/2020/day/2
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            a, b, c = raw_input_line.split(' ')
            num_a, num_b = map(int, a.split('-'))
            char = b[0]
            password = c
            result.append((num_a, num_b, char, password))
        return result
    
    def solve(self, parsed_input: List[str]) -> int:
        valid_password_count = 0
        for min_count, max_count, char, password in parsed_input:
            char_count = password.count(char)
            if min_count <= char_count <= max_count:
                valid_password_count += 1
        result = valid_password_count
        return result
    
    def solve2(self, parsed_input: List[str]) -> int:
        valid_password_count = 0
        for i, j, char, password in parsed_input:
            check = 0
            if password[i - 1] == char:
                check += 1
            if password[j - 1] == char:
                check += 1
            if check == 1:
                valid_password_count += 1
        result = valid_password_count
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

class Day01: # Report Repair
    '''
    Report Repair
    https://adventofcode.com/2020/day/1
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(int(raw_input_line))
        return result
    
    def solve(self, parsed_input: List[str]) -> int:
        result = -1
        seen = set()
        for num in parsed_input:
            target = 2020 - num
            if target in seen:
                result = num * target
            else:
                seen.add(num)
        return result
    
    def solve2(self, parsed_input: List[str]) -> int:
        nums = sorted(parsed_input)
        N = len(nums)
        for i in range(N):
            target = 2020 - nums[i]
            j = i + 1
            k = N - 1
            while j < k:
                total = nums[j] + nums[k]
                if total == target:
                    return nums[i] * nums[j] * nums[k]
                elif total < target:
                    j += 1
                elif total > target:
                    k -= 1
    
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
    python Solver.py 9 < day09.in
    '''
    solvers = {
        1: (Day01, 'Report Repair'),
        2: (Day02, 'Password Philosophy'),
        3: (Day03, 'Toboggan Trajectory'),
        4: (Day04, 'Passport Processing'),
        5: (Day05, 'Binary Boarding'),
        6: (Day06, 'Custom Customs'),
        7: (Day07, 'Handy Haversacks'),
        8: (Day08, 'Handheld Halting'),
        9: (Day09, 'Encoding Error'),
    #    10: (Day10, '???'),
    #    11: (Day11, '???'),
    #    12: (Day12, '???'),
    #    13: (Day13, '???'),
    #    14: (Day14, '???'),
    #    15: (Day15, '???'),
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
