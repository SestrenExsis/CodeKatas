'''
Created on 2021-11-29

@author: Sestren
'''
import argparse
import collections
import copy
import functools
import heapq
import operator
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
    https://adventofcode.com/2021/day/?
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

class Day25: # Sea Cucumber
    '''
    https://adventofcode.com/2021/day/25
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        east = set()
        south = set()
        rows = len(raw_input_lines)
        cols = len(raw_input_lines[0])
        for row in range(rows):
            for col in range(cols):
                if raw_input_lines[row][col] == '>':
                    east.add((row, col))
                elif raw_input_lines[row][col] == 'v':
                    south.add((row, col))
        result = rows, cols, east, south
        return result
    
    def solve(self, rows, cols, east, south):
        step_id = 0
        while True:
            step_id += 1
            move_count = 0
            next_east = set()
            for (row, col) in east:
                col2 = (col + 1) % cols
                if (row, col2) not in east and (row, col2) not in south:
                    next_east.add((row, col2))
                    move_count += 1
                else:
                    next_east.add((row, col))
            east = next_east
            next_south = set()
            for (row, col) in south:
                row2 = (row + 1) % rows
                if (row2, col) not in east and (row2, col) not in south:
                    next_south.add((row2, col))
                    move_count += 1
                else:
                    next_south.add((row, col))
            south = next_south
            if move_count < 1:
                break
        result = step_id
        return result
    
    def solve2(self):
        result = 'Happy holidays!'
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        rows, cols, east, south = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(rows, cols, set(east), set(south)),
            self.solve2(),
            )
        result = solutions
        return result

class Day24: # Arithmetic Logic Unit
    '''
    https://adventofcode.com/2021/day/24
    '''
    def get_program(self, raw_input_lines: List[str]):
        program = []
        for raw_input_line in raw_input_lines:
            parts = raw_input_line.split(' ')
            instruction = []
            for part in parts:
                try:
                    num = int(part)
                    instruction.append(num)
                except ValueError:
                    instruction.append(part)
            program.append(tuple(instruction))
        result = program
        return result
    
    class ALU:
        def __init__(self, program, input):
            self.program = program
            self.pc = 0
            self.registers = {
                'w': 0,
                'x': 0,
                'y': 0,
                'z': 0,
            }
            self.input = list(reversed(list(map(int, str(input)))))
            self.legal = True
        
        def step(self):
            if not self.legal:
                return
            instruction = self.program[self.pc]
            op = instruction[0]
            if op == 'inp':
                a = instruction[1]
                value = self.input.pop()
                self.registers[a] = value
            elif op == 'add':
                a, b = instruction[1], instruction[2]
                operand = b
                if type(b) is str:
                    operand = self.registers[b]
                value = self.registers[a] + operand
                self.registers[a] = value
            elif op == 'mul':
                a, b = instruction[1], instruction[2]
                operand = b
                if type(b) is str:
                    operand = self.registers[b]
                value = self.registers[a] * operand
                self.registers[a] = value
            elif op == 'div':
                a, b = instruction[1], instruction[2]
                operand = b
                if type(b) is str:
                    operand = self.registers[b]
                if operand == 0:
                    self.legal = False
                    return
                value = self.registers[a] // operand
                self.registers[a] = value
            elif op == 'mod':
                a, b = instruction[1], instruction[2]
                operand = b
                if type(b) is str:
                    operand = self.registers[b]
                if self.registers[a] < 0 or operand <= 0:
                    self.legal = False
                    return
                value = self.registers[a] % operand
                self.registers[a] = value
            elif op == 'eql':
                a, b = instruction[1], instruction[2]
                operand = b
                if type(b) is str:
                    operand = self.registers[b]
                value = 1 if self.registers[a] == operand else 0
                self.registers[a] = value
            self.pc += 1
        
        def spin(self, spin_count: int=1):
            for _ in range(spin_count):
                for _ in range(18):
                    self.step()
        
        def run(self):
            while self.pc < len(self.program) and self.legal:
                self.step()
    
    def F(self, w, z, param_a, param_b, param_c):
        '''
        mul x 0
        add x z
        mod x 26
        div z param_a
        add x param_b
        eql x w
        eql x 0
        mul y 0
        add y 25
        mul y x
        add y 1
        mul z y
        mul y 0
        add y w
        add y param_c
        mul y x
        add z y
        -------
        x = z % 26
        z = z // param_a
        x = x + param_b
        x = 0 if x == w else 1
        y = 25 * x + 1
        z = z * y
        y = x * (w + param_c)
        result = z + y
        -------
        x = 0 if w == (z % 26 + param_b) else 1
        result = (z // param_a) * (25 * x + 1) + x * (w + param_c)
        '''
        result = (z // 26) if param_a else z
        if w != (z % 26 + param_b):
            result = 26 * result + w + param_c
        return result
    
    def G(self, inputs):
        z = 0
        for i, (param_a, param_b, param_c) in enumerate((
            (0, 15, 15),
            (0, 15, 10),
            (0, 12, 2),
            (0, 13, 16),
            (1, -12, 12),
            (0, 10, 11),
            (1, -9, 5),
            (0, 14, 16),
            (0, 13, 6),
            (1, -14, 15),
            (1, -11, 3),
            (1, -2, 12),
            (1, -16, 10),
            (1, -14, 13),
        )):
            z = self.F(inputs[i], z, param_a, param_b, param_c)
        result = z
        return result
    
    def solve_slowly(self, program):
        '''
        Model numbers are 14-digit numbers without zeroes, passed
        left-to-right to the input.
        Programs are valid if z has a value of 0 at the end.
        '''
        max_valid_model_number = float('-inf')
        for model_number in range(
            99_999_999_999_999,
            11_111_111_111_111 - 1,
            -1,
        ):
            if '0' in str(model_number):
                continue
            alu = self.ALU(program, model_number)
            if alu.input[:4] == [9, 9, 9, 9]:
                print(model_number)
            alu.run()
            if alu.legal and alu.registers['z'] == 0:
                max_valid_model_number = model_number
                break
            inputs = list(reversed(list(map(int, str(model_number)))))
            assert self.G(inputs) == alu.registers['z']
        result = max_valid_model_number
        return result
    
    def interactive_solver(self, program):
        while True:
            user_input = input()
            alu = self.ALU(program, user_input)
            for _ in range(18 * len(str(user_input))):
                alu.step()
            z = alu.registers['z']
            stack = []
            while z > 0:
                stack.append(z % 26)
                z //= 26
            print(' ' * 16, stack)
    
    def main(self):
        # raw_input_lines = get_raw_input_lines()
        with open('inputs/2021day24.in') as open_file:
            raw_input_lines = open_file.read().splitlines()
            program = self.get_program(raw_input_lines)
            self.interactive_solver(program)

class Day23: # Amphipod
    '''
    https://adventofcode.com/2021/day/23
    '''
    amphipods = {
        'A': (1, 1),
        'B': (2, 10),
        'C': (3, 100),
        'D': (4, 1000),
    }
    halls =  {0, 1, 3, 5, 7, 9, 10}
    entries = {2, 4, 6, 8}

    def get_initial_states(self, raw_input_lines: List[str]):
        state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for raw_input_line in reversed(raw_input_lines[2:4]):
            state[2] = 10 * state[2] + self.amphipods[raw_input_line[3]][0]
            state[4] = 10 * state[4] + self.amphipods[raw_input_line[5]][0]
            state[6] = 10 * state[6] + self.amphipods[raw_input_line[7]][0]
            state[8] = 10 * state[8] + self.amphipods[raw_input_line[9]][0]
        state2 = state[:]
        state2[2] = 1000 * (state2[2] // 10) + 440 + state2[2] % 10
        state2[4] = 1000 * (state2[4] // 10) + 230 + state2[4] % 10
        state2[6] = 1000 * (state2[6] // 10) + 120 + state2[6] % 10
        state2[8] = 1000 * (state2[8] // 10) + 310 + state2[8] % 10
        result = (tuple(state), tuple(state2))
        return result
    
    def edit_move(self, cost, state, capacity):
        while True:
            prev_cost = cost
            for hall_id in self.halls:
                if state[hall_id] < 1:
                    continue
                # The only valid choice for an entry is determined by its type
                amphipod = state[hall_id] % 10
                assert amphipod in (1, 2, 3, 4)
                entry_id = 2 * amphipod
                # Check if entryway is ready for amphipods to move in
                if not (state[entry_id] in (0, amphipod)):
                    continue
                next_state = list(state)
                # Check if hallway is blocked
                start = min(hall_id, entry_id)
                end = max(hall_id, entry_id)
                for i in range(start, end + 1):
                    if i == hall_id or i not in self.halls:
                        continue
                    if next_state[i] > 0:
                        break
                else:
                    # Cost is determined by manhattan distance and amphipod type
                    stack = next_state[entry_id]
                    height = capacity if stack == 0 else capacity - len(str(stack))
                    steps = abs(hall_id - entry_id) + height
                    cost += steps * (10 ** (amphipod - 1))
                    # Move the amphipod
                    next_state[hall_id] = 0
                    next_state[entry_id] = 10 * next_state[entry_id] + amphipod
                    state = tuple(next_state)
            if cost == prev_cost:
                break
        result = (cost, state)
        return result
    
    def solve(self, initial_state, goal_state):
        min_cost = float('inf')
        capacity = len(str(goal_state[2]))
        assert capacity in (2, 4)
        work = [
            (0, tuple(initial_state)),
        ]
        debug = 0
        while len(work) > 0:
            cost, state = heapq.heappop(work)
            if debug % 10_000 == 0:
                print(len(work), cost, state)
            debug += 1
            # Check if we have already reached the goal state
            if state == goal_state:
                min_cost = cost
                break
            next_moves = []
            # Consider all possible entryway-to-hallway moves
            for entry_id in self.entries:
                # Check if entryway is empty or already at its goal
                if (
                    state[entry_id] < 1 or
                    state[entry_id] == goal_state[entry_id]
                ):
                    continue
                amphipod = state[entry_id] % 10
                # Check if entryway has only amphipods in the correct position
                if (
                    len(set(str(state[entry_id]))) == 1 and
                    amphipod == 1 + ((entry_id - 2) // 2)
                ):
                    continue
                for hall_id in self.halls:
                    next_state = list(state)
                    next_state[entry_id] //= 10
                    # Check if hallway is blocked
                    start = min(hall_id, entry_id)
                    end = max(hall_id, entry_id)
                    for i in range(start, end + 1):
                        if i not in self.halls:
                            continue
                        if next_state[i] > 0:
                            break
                    else:
                        next_state[hall_id] = amphipod
                        # Cost is determined by manhattan distance and amphipod type
                        stack = next_state[entry_id]
                        height = capacity if stack == 0 else capacity - len(str(stack))
                        steps = abs(hall_id - entry_id) + height
                        next_cost = cost + steps * (10 ** (amphipod - 1))
                        next_move = (next_cost, tuple(next_state))
                        next_moves.append(next_move)
            # Force any amphipods that can reach their goal state to do so
            for cost, state in next_moves:
                edited_cost, edited_state = self.edit_move(cost, state, capacity)
                heapq.heappush(work, (edited_cost, edited_state))
        result = min_cost
        return result
    
    def run_tests(self):
        for (cost, state, expected_cost, expected_state) in (
            (
                0, (1, 1,  0, 0, 22, 0, 33, 0, 44, 0, 0),
                6, (0, 0, 11, 0, 22, 0, 33, 0, 44, 0, 0),
            ),
            (
                0, (1, 1,  0, 0, 22, 0, 33, 0,  0, 4, 4),
                6006, (0, 0, 11, 0, 22, 0, 33, 0, 44, 0, 0),
            ),
            (
                0, (4, 4,  0, 0, 22, 0, 33, 0,  0, 1, 1),
                18018, (0, 0, 11, 0, 22, 0, 33, 0, 44, 0, 0),
            ),
            (
                0, (1, 4,  0, 0, 22, 0, 33, 0,  0, 1, 4),
                12012, (0, 0, 11, 0, 22, 0, 33, 0, 44, 0, 0),
            ),
        ):
            observed_cost, observed_state = self.edit_move(cost, state, 2)
            assert observed_cost == expected_cost
            assert observed_state == expected_state
    
    def main(self):
        self.run_tests()
        raw_input_lines = get_raw_input_lines()
        states = self.get_initial_states(raw_input_lines)
        solutions = (
            self.solve(states[0], (0, 0, 11, 0, 22, 0, 33, 0, 44, 0, 0)),
            self.solve(states[1], (0, 0, 1111, 0, 2222, 0, 3333, 0, 4444, 0, 0)),
            )
        result = solutions
        return result

class Day22: # Reactor Reboot
    '''
    https://adventofcode.com/2021/day/22
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        steps = []
        for raw_input_line in raw_input_lines:
            action, parts = raw_input_line.split(' ')
            xx, yy, zz = parts.split(',')
            x1, x2 = tuple(map(int, xx[2:].split('..')))
            y1, y2 = tuple(map(int, yy[2:].split('..')))
            z1, z2 = tuple(map(int, zz[2:].split('..')))
            steps.append((action, x1, x2, y1, y2, z1, z2))
        result = steps
        return result
    
    def solve(self, steps):
        activity = set()
        for action, x1, x2, y1, y2, z1, z2 in steps:
            for x in range(x1, x2 + 1):
                if not (-50 <= x <= 50):
                    break
                for y in range(y1, y2 + 1):
                    if not (-50 <= y <= 50):
                        break
                    for z in range(z1, z2 + 1):
                        if not (-50 <= z <= 50):
                            break
                        if action == 'on':
                            activity.add((x, y, z))
                        elif action == 'off':
                            activity.discard((x, y, z))
        result = len(activity)
        return result
    
    def solve2(self, steps):
        x_cuts = set()
        y_cuts = set()
        z_cuts = set()
        for _, x1, x2, y1, y2, z1, z2 in steps:
            x_cuts.add(x1)
            x_cuts.add(x2 + 1)
            y_cuts.add(y1)
            y_cuts.add(y2 + 1)
            z_cuts.add(z1)
            z_cuts.add(z2 + 1)
        x_segments = list(sorted(x_cuts))
        y_segments = list(sorted(y_cuts))
        z_segments = list(sorted(z_cuts))
        compressed_steps = []
        for action, x1, x2, y1, y2, z1, z2 in steps:
            print(action, x1, x2, y1, y2, z1, z2)
            # Calculate x segments covered
            x_start = 0
            while x_segments[x_start] < x1:
                x_start += 1
            x_end = len(x_segments) - 1
            while x_segments[x_end] > x2 + 1:
                x_end -= 1
            # Calculate y segments covered
            y_start = 0
            while y_segments[y_start] < y1:
                y_start += 1
            y_end = len(y_segments) - 1
            while y_segments[y_end] > y2 + 1:
                y_end -= 1
            # Calculate z segments covered
            z_start = 0
            while z_segments[z_start] < z1:
                z_start += 1
            z_end = len(z_segments) - 1
            while z_segments[z_end] > z2 + 1:
                z_end -= 1
            # Compress steps
            compressed_steps.append((
                action == 'on',
                x_start, x_end,
                y_start, y_end,
                z_start, z_end,
            ))
        total_area = 0
        for xi in range(len(x_segments)):
            print(xi, total_area)
            for yi in range(len(y_segments)):
                for zi in range(len(z_segments)):
                    active = False
                    for action, x1, x2, y1, y2, z1, z2 in reversed(compressed_steps):
                        if x1 <= xi < x2 and y1 <= yi < y2 and z1 <= zi < z2:
                            active = action
                            break
                    if active:
                        area_x = x_segments[xi + 1] - x_segments[xi]
                        area_y = y_segments[yi + 1] - y_segments[yi]
                        area_z = z_segments[zi + 1] - z_segments[zi]
                        total_area += area_x * area_y * area_z
        result = total_area
        return result
    
    def solve2_slowly(self, steps):
        x_cuts = set()
        y_cuts = set()
        z_cuts = set()
        for _, x1, x2, y1, y2, z1, z2 in steps:
            x_cuts.add(x1)
            x_cuts.add(x2 + 1)
            y_cuts.add(y1)
            y_cuts.add(y2 + 1)
            z_cuts.add(z1)
            z_cuts.add(z2 + 1)
        x_segments = list(sorted(x_cuts))
        y_segments = list(sorted(y_cuts))
        z_segments = list(sorted(z_cuts))
        activity = set()
        for i, (action, x1, x2, y1, y2, z1, z2) in enumerate(steps):
            print(i, len(activity))
            # Calculate x segments covered
            x_start = 0
            while x_segments[x_start] < x1:
                x_start += 1
            x_end = len(x_segments) - 1
            while x_segments[x_end] > x2 + 1:
                x_end -= 1
            # Calculate y segments covered
            y_start = 0
            while y_segments[y_start] < y1:
                y_start += 1
            y_end = len(y_segments) - 1
            while y_segments[y_end] > y2 + 1:
                y_end -= 1
            # Calculate z segments covered
            z_start = 0
            while z_segments[z_start] < z1:
                z_start += 1
            z_end = len(z_segments) - 1
            while z_segments[z_end] > z2 + 1:
                z_end -= 1
            # Turn on or off all segments
            for xx in range(x_start, x_end):
                for yy in range(y_start, y_end):
                    for zz in range(z_start, z_end):
                        if action == 'on':
                            activity.add((xx, yy, zz))
                        elif action == 'off':
                            activity.discard((xx, yy, zz))
        total_area = 0
        for x, y, z in activity:
            area_x = x_segments[x + 1] - x_segments[x]
            area_y = y_segments[y + 1] - y_segments[y]
            area_z = z_segments[z + 1] - z_segments[z]
            total_area += area_x * area_y * area_z
        result = total_area
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        steps = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(steps),
            self.solve2(steps),
            )
        result = solutions
        return result

class Day21: # Dirac Dice
    '''
    https://adventofcode.com/2021/day/21
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        pos1 = int(raw_input_lines[0].split(' ')[-1]) - 1
        pos2 = int(raw_input_lines[1].split(' ')[-1]) - 1
        result = (pos1, pos2)
        return result
    
    def solve(self, starts):
        scores = [0, 0]
        positions = [starts[0], starts[1]]
        roll_count = 0
        curr_player_id = 0
        while max(scores) < 1000:
            a = 1 + (roll_count) % 100
            b = 1 + (roll_count + 1) % 100
            c = 1 + (roll_count + 2) % 100
            new_pos = (positions[curr_player_id] + a + b + c) % 10
            positions[curr_player_id] = new_pos
            score = 1 + new_pos
            scores[curr_player_id] += score
            roll_count += 3
            curr_player_id = 1 - curr_player_id
        result = min(scores) * roll_count
        return result
    
    def solve2(self, starts):
        # player_id: count
        wins = collections.defaultdict(int)
        # (max_score, score1, score2, pos1, pos2, turn, count)
        work = [(0, 0, 0, starts[0], starts[1], 1, 1)]
        while len(work) > 0:
            element = heapq.heappop(work)
            max_score, score1, score2, pos1, pos2, turn, count = element
            max_score *= -1
            score1 *= -1
            score2 *= -1
            if max_score >= 21:
                winning_player = 1 if score1 > score2 else 2
                wins[winning_player] += count
                continue
            for roll, occurrences in (
                (3, 1),
                (4, 3),
                (5, 6),
                (6, 7),
                (7, 6),
                (8, 3),
                (9, 1),
            ):
                npos1 = pos1
                npos2 = pos2
                nscore1 = score1
                nscore2 = score2
                nturn = turn
                ncount = occurrences * count
                if turn == 1:
                    nturn = 2
                    npos1 = (npos1 + roll) % 10
                    nscore1 += 1 + npos1
                elif turn == 2:
                    nturn = 1
                    npos2 = (npos2 + roll) % 10
                    nscore2 += 1 + npos2
                nmax_score = max(nscore1, nscore2)
                heapq.heappush(work, (
                    -nmax_score,
                    -nscore1,
                    -nscore2,
                    npos1,
                    npos2,
                    nturn,
                    ncount,
                ))
        result = max(wins.values())
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

class Day20: # Trench Map
    '''
    https://adventofcode.com/2021/day/20
    '''
    def get_image_info(self, raw_input_lines: List[str]):
        image_enhancement_algorithm = set()
        for index, cell in enumerate(raw_input_lines[0]):
            if cell == '#':
                image_enhancement_algorithm.add(index)
        image = set()
        for row, raw_input_line in enumerate(raw_input_lines[2:]):
            for col, cell in enumerate(raw_input_line):
                if cell == '#':
                    image.add((row, col))
        result = image_enhancement_algorithm, image
        return result
    
    def visualize(self, image):
        min_row = min(row for row, _ in image)
        max_row = max(row for row, _ in image)
        min_col = min(col for _, col in image)
        max_col = max(col for _, col in image)
        for row in range(min_row, max_row + 1):
            row_data = []
            for col in range(min_col, max_col + 1):
                cell = '.'
                if (row, col) in image:
                    cell = '#'
                row_data.append(cell)
            print(''.join(row_data))
    
    def solve(self, image_enhancement_algorithm, input_image, enhancement_count):
        image = set(input_image)
        for enhancement_id in range(enhancement_count):
            min_row = min(row for row, _ in image)
            max_row = max(row for row, _ in image)
            min_col = min(col for _, col in image)
            max_col = max(col for _, col in image)
            next_image = set()
            for row in range(min_row - 3, max_row + 4):
                for col in range(min_col - 3, max_col + 4):
                    index = 0
                    for (r, c, value) in (
                        (row - 1, col - 1, 256),
                        (row - 1, col + 0, 128),
                        (row - 1, col + 1, 64),
                        (row + 0, col - 1, 32),
                        (row + 0, col + 0, 16),
                        (row + 0, col + 1, 8),
                        (row + 1, col - 1, 4),
                        (row + 1, col + 0, 2),
                        (row + 1, col + 1, 1),
                    ):
                        if (
                            (r, c) in image or
                            enhancement_id % 2 == 1 and
                            0 in image_enhancement_algorithm and
                            (
                                row <= min_row or
                                row >= max_row or
                                col <= min_col or
                                col >= max_col
                            )
                        ):
                            index += value
                    if index in image_enhancement_algorithm:
                        next_image.add((row, col))
            image = next_image
        result = len(image)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        image_enhancement_algorithm, input_image = self.get_image_info(raw_input_lines)
        solutions = (
            self.solve(image_enhancement_algorithm, input_image, 2),
            self.solve(image_enhancement_algorithm, input_image, 50),
            )
        result = solutions
        return result

class Day19: # Beacon Scanner
    '''
    https://adventofcode.com/2021/day/19
    '''
    def get_scanners(self, raw_input_lines: List[str]):
        scanners = collections.defaultdict(list)
        scanner_id = -1
        for raw_input_line in raw_input_lines:
            if 'scanner' in raw_input_line:
                scanner_id = int(raw_input_line.split(' ')[2])
            elif ',' in raw_input_line:
                scan = tuple(map(int, raw_input_line.split(',')))
                scanners[scanner_id].append(scan)
        result = scanners
        return result
    
    def get_master_scan(self, scanners):
        # Start with one scan as the reference point for all others
        _, scan = scanners.popitem()
        scanner_coords = {
            (0, 0, 0),
        }
        master_scan = collections.defaultdict(int)
        for coordinate in scan:
            master_scan[coordinate] += 1
        # Merge all scans into the master scan
        while len(scanners) > 0:
            merged_keys = set()
            for scan_key, scan in scanners.items():
                # Try all 24 orientations for an incoming scan
                for (a, b, c, d, e, f, g, h, i) in (
                    ( 1,  0,  0,    0,  1,  0,    0,  0,  1),
                    (-1,  0,  0,    0, -1,  0,    0,  0,  1),
                    (-1,  0,  0,    0,  1,  0,    0,  0, -1),
                    ( 1,  0,  0,    0, -1,  0,    0,  0, -1),
                    (-1,  0,  0,    0,  0,  1,    0,  1,  0),
                    ( 1,  0,  0,    0,  0, -1,    0,  1,  0),
                    ( 1,  0,  0,    0,  0,  1,    0, -1,  0),
                    (-1,  0,  0,    0,  0, -1,    0, -1,  0),
                    ( 0, -1,  0,    1,  0,  0,    0,  0,  1),
                    ( 0,  1,  0,   -1,  0,  0,    0,  0,  1),
                    ( 0,  1,  0,    1,  0,  0,    0,  0, -1),
                    ( 0, -1,  0,   -1,  0,  0,    0,  0, -1),
                    ( 0,  1,  0,    0,  0,  1,    1,  0,  0),
                    ( 0, -1,  0,    0,  0, -1,    1,  0,  0),
                    ( 0, -1,  0,    0,  0,  1,   -1,  0,  0),
                    ( 0,  1,  0,    0,  0, -1,   -1,  0,  0),
                    ( 0,  0,  1,    1,  0,  0,    0,  1,  0),
                    ( 0,  0, -1,   -1,  0,  0,    0,  1,  0),
                    ( 0,  0, -1,    1,  0,  0,    0, -1,  0),
                    ( 0,  0,  1,   -1,  0,  0,    0, -1,  0),
                    ( 0,  0, -1,    0,  1,  0,    1,  0,  0),
                    ( 0,  0,  1,    0, -1,  0,    1,  0,  0),
                    ( 0,  0,  1,    0,  1,  0,   -1,  0,  0),
                    ( 0,  0, -1,    0, -1,  0,   -1,  0,  0),
                ):
                    # Try every pair of beacons as an offset
                    match_found = False
                    for (x0, y0, z0) in scan:
                        x1 = a * x0 + b * y0 + c * z0
                        y1 = d * x0 + e * y0 + f * z0
                        z1 = g * x0 + h * y0 + i * z0
                        # A pair is (x1, y1, z1) and (x2, y2, z2)
                        for (x2, y2, z2) in master_scan:
                            dx = x1 - x2
                            dy = y1 - y2
                            dz = z1 - z2
                            # Look for matching beacons
                            match_count = 0
                            coords = collections.defaultdict(int)
                            for x3, y3, z3 in scan:
                                x = (a * x3 + b * y3 + c * z3) - dx
                                y = (d * x3 + e * y3 + f * z3) - dy
                                z = (g * x3 + h * y3 + i * z3) - dz
                                coords[(x, y, z)] += 1
                                if (x, y, z) in master_scan:
                                    match_count += 1
                            # If 12 or more match, then merge into master
                            if match_count >= 12:
                                scanner_coords.add((dx, dy, dz))
                                merged_keys.add(scan_key)
                                for coord in coords:
                                    master_scan[coord] += coords[coord]
                                match_found = True
                                break
                        if match_found:
                            break
                    if match_found:
                        break
            # Delete scans that have been merged into the master
            for scan_key in merged_keys:
                del scanners[scan_key]
        result = (master_scan, scanner_coords)
        return result
    
    def solve(self, master_scan):
        result = len(master_scan)
        return result
    
    def solve2(self, scanner_coords):
        max_manhattan_distance = 0
        for x1, y1, z1 in scanner_coords:
            for x2, y2, z2 in scanner_coords:
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                dz = abs(z2 - z1)
                manhattan_distance = dx + dy + dz
                max_manhattan_distance = max(
                    max_manhattan_distance,
                    manhattan_distance,
                )
        result = max_manhattan_distance
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        scanners = self.get_scanners(raw_input_lines)
        master_scan, scanner_coords = self.get_master_scan(scanners)
        solutions = (
            self.solve(master_scan),
            self.solve2(scanner_coords),
            )
        result = solutions
        return result

class Day18: # Snailfish
    '''
    https://adventofcode.com/2021/day/18
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def explode(self, num: str, index: int) -> str:
        left = index
        while left > 0 and num[left - 1] in ',0123456789':
            left -= 1
        right = index
        while right < len(num) - 1 and num[right + 1] in ',0123456789':
            right += 1
        left_side = num[:left - 1]
        right_side = num[right + 2:]
        pair = tuple(map(int, num[left:right + 1].split(',')))
        L2 = len(left_side) - 1
        while L2 > 0:
            if left_side[L2] in '0123456789':
                L1 = L2
                while L1 > 0 and left_side[L1 - 1] in '0123456789':
                    L1 -= 1
                newNum = str(int(left_side[L1:L2 + 1]) + pair[0])
                left_side = left_side[:L1] + newNum + left_side[L2 + 1:]
                break
            L2 -= 1
        R1 = 0
        while R1 < len(right_side):
            if right_side[R1] in '0123456789':
                R2 = R1
                while R2 < len(right_side) and right_side[R2 + 1] in '0123456789':
                    R2 += 1
                newNum = str(int(right_side[R1:R2 + 1]) + pair[1])
                right_side = right_side[:R1] + newNum + right_side[R2 + 1:]
                break
            R1 += 1
        num = left_side + '0' + right_side
        result = num
        return result
    
    def split(self, num: str, index: int) -> str:
        left = index
        while left > 0 and num[left - 1] in '0123456789':
            left -= 1
        right = index
        while right < len(num) - 1 and num[right + 1] in '0123456789':
            right += 1
        value = int(num[left:right + 1])
        a = value // 2
        b = value - a
        num = num[:left] + '[' + str(a) + ',' + str(b) + ']' + num[right + 1:]
        result = num
        return result
    
    def reduce(self, num: str) -> str:
        '''
        Explode the first pair that is 4 layers deep
        If no explosions, split the first number that is 10 or greater
        '''
        while True:
            # Check for explosions
            explosion_ind = False
            depth = 0
            for index, char in enumerate(num):
                if char == '[':
                    depth += 1
                elif char == ']':
                    depth -= 1
                if depth >= 4 and char == ']':
                    num = self.explode(num, index - 1)
                    explosion_ind = True
                    break
            # If no explosions, check for splits
            if explosion_ind:
                continue
            split_ind = False
            value = 0
            for index, char in enumerate(num):
                if char in '0123456789':
                    value = 10 * value + int(char)
                else:
                    value = 0
                if value >= 10:
                    num = self.split(num, index)
                    split_ind = True
                    break
            if not explosion_ind and not split_ind:
                break
        result = num
        return result

    def add(self, numA: str, numB: str) -> str:
        result = self.reduce('[' + numA + ',' + numB + ']')
        return result
    
    def magnitude(self, num: str) -> int:
        magnitude = 0
        while True:
            repeat_ind = False
            left = 0
            right = 0
            for char in num:
                if char == '[':
                    left = right
                elif char == ']':
                    repeat_ind = True
                    break
                right += 1
            if repeat_ind:
                pair = tuple(map(int, num[left + 1:right].split(',')))
                magnitude = 3 * pair[0] + 2 * pair[1]
                num = num[:left] + str(magnitude) + num[right + 1:]
            else:
                break
        result = magnitude
        return result
    
    def solve(self, nums):
        numA = nums[0]
        for numB in nums[1:]:
            numA = self.add(numA, numB)
        result = self.magnitude(numA)
        return result
    
    def solve2(self, nums):
        max_magnitude = 0
        for i, numA in enumerate(nums):
            for j, numB in enumerate(nums):
                if i == j:
                    continue
                magnitude = self.magnitude(self.add(numA, numB))
                max_magnitude = max(max_magnitude, magnitude)
        result = max_magnitude
        return result
    
    def test(self):
        print('Explosion tests ...', end='')
        assert self.explode('[[[[[9,8],1],2],3],4]', 7) == '[[[[0,9],2],3],4]'
        assert self.explode('[[[[[4,3],4],4],[7,[[8,4],9]]],[1,1]]', 7) == '[[[[0,7],4],[7,[[8,4],9]]],[1,1]]'
        print('... PASSED!!')
        print('Split tests ...', end='')
        assert self.split('[[[[0,7],4],[15,[0,13]]],[1,1]]', 14) == '[[[[0,7],4],[[7,8],[0,13]]],[1,1]]'
        assert self.split('[[[[0,7],4],[[7,8],[0,13]]],[1,1]]', 23) == '[[[[0,7],4],[[7,8],[0,[6,7]]]],[1,1]]'
        print('... PASSED!!')
        print('Magnitude tests ...', end='')
        assert self.magnitude('[[1,2],[[3,4],5]]') == 143
        assert self.magnitude('[[[[0,7],4],[[7,8],[6,0]]],[8,1]]') == 1384
        assert self.magnitude('[[[[1,1],[2,2]],[3,3]],[4,4]]') == 445
        assert self.magnitude('[[[[3,0],[5,3]],[4,4]],[5,5]]') == 791
        assert self.magnitude('[[[[5,0],[7,4]],[5,5]],[6,6]]') == 1137
        assert self.magnitude('[[[[8,7],[7,7]],[[8,6],[7,7]]],[[[0,7],[6,6]],[8,7]]]') == 3488
        print('... PASSED!!')
        print('Add tests ...', end='')
        assert self.add('[[[1,1],[2,2]],[3,3]]', '[4,4]') == '[[[[1,1],[2,2]],[3,3]],[4,4]]'
        assert self.add('[[[[1,1],[2,2]],[3,3]],[4,4]]', '[5,5]') == '[[[[3,0],[5,3]],[4,4]],[5,5]]'
        assert self.add('[[[[3,0],[5,3]],[4,4]],[5,5]]', '[6,6]') == '[[[[5,0],[7,4]],[5,5]],[6,6]]'
        assert self.add('[[[[4,3],4],4],[7,[[8,4],9]]]','[1,1]') == '[[[[0,7],4],[[7,8],[6,0]]],[8,1]]'
        assert self.add('[[[0,[4,5]],[0,0]],[[[4,5],[2,6]],[9,5]]]','[7,[[[3,7],[4,3]],[[6,3],[8,8]]]]') == '[[[[4,0],[5,4]],[[7,7],[6,0]]],[[8,[7,7]],[[7,9],[5,0]]]]'
        print('... PASSED!!')
        print('Reduce tests ...', end='')
        assert self.reduce('[[[[[4,3],4],4],[7,[[8,4],9]]],[1,1]]') == '[[[[0,7],4],[[7,8],[6,0]]],[8,1]]'
        print('... PASSED!!')
    
    def main(self):
        # self.test()
        raw_input_lines = get_raw_input_lines()
        nums = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(nums),
            self.solve2(nums),
            )
        result = solutions
        return result

class Day17: # Trick Shot
    '''
    https://adventofcode.com/2021/day/17
    '''
    class Simulation:
        def __init__(self, x_vel: int, y_vel: int):
            self.x = 0
            self.y = 0
            self.x_vel = x_vel
            self.y_vel = y_vel
        
        def step(self):
            self.x += self.x_vel
            self.y += self.y_vel
            if self.x_vel > 0:
                self.x_vel -= 1
            elif self.x_vel < 0:
                self.x_vel += 1
            self.y_vel -= 1

    def get_area(self, raw_input_lines: List[str]):
        a, b, c, d = raw_input_lines[0].split()
        c1, c2 = c.split('..')
        min_x = int(c1.split('=')[1])
        max_x = int(c2[:-1])
        d1, d2 = d.split('..')
        min_y = int(d1.split('=')[1])
        max_y = int(d2)
        result = (min_x, max_x, min_y, max_y)
        return result
    
    def find_valid_x_vels(self, lower_bound: int, upper_bound: int) -> set:
        valid_x_vels = set()
        for initial_x_vel in range(upper_bound + 1, -1, -1):
            valid_ind = False
            x = 0
            x_vel = initial_x_vel
            while x_vel > 0:
                x += x_vel
                x_vel -= 1
                if lower_bound <= x <= upper_bound:
                    valid_ind = True
                    break
            if valid_ind:
                valid_x_vels.add(x_vel)
        result = valid_x_vels
        return result
    
    def find_valid_y_vels(self, lower_bound: int, upper_bound: int) -> set:
        valid_y_vels = set()
        for initial_y_vel in range(upper_bound + 1, -1, -1):
            valid_ind = False
            y = 0
            y_vel = initial_y_vel
            while y_vel > 0:
                y += y_vel
                y_vel -= 1
                if lower_bound <= y <= upper_bound:
                    valid_ind = True
                    break
            if valid_ind:
                valid_y_vels.add(y_vel)
        result = valid_y_vels
        return result
    
    def solve(self, x_min, x_max, y_min, y_max):
        valid_x_vels = set()
        for x_vel in range(-100, 100 + 1):
            sim = self.Simulation(x_vel, 0)
            valid_ind = False
            for _ in range(x_vel + 1):
                sim.step()
                if x_min <= sim.x <= x_max:
                    valid_ind = True
                    break
            if valid_ind:
                valid_x_vels.add(x_vel)
        valid_y_vels = set()
        for y_vel in range(-100, 100 + 1):
            sim = self.Simulation(0, y_vel)
            valid_ind = False
            for _ in range(10_000):
                sim.step()
                if y_min <= sim.y <= y_max:
                    valid_ind = True
                    break
            if valid_ind:
                valid_y_vels.add(y_vel)
        highest_valid_y = float('-inf')
        for x_vel in valid_x_vels:
            for y_vel in valid_y_vels:
                sim = self.Simulation(x_vel, y_vel)
                valid_ind = False
                highest_y = float('-inf')
                for _ in range(10_000):
                    sim.step()
                    highest_y = max(highest_y, sim.y)
                    if x_min <= sim.x <= x_max and y_min <= sim.y <= y_max:
                        valid_ind = True
                    if sim.y < y_min and sim.y_vel < 0:
                        break
                if valid_ind:
                    highest_valid_y = max(highest_valid_y, highest_y)
        result = highest_valid_y
        return result
    
    def solve2(self, x_min, x_max, y_min, y_max):
        valid_x_vels = set()
        for x_vel in range(x_max + 1):
            sim = self.Simulation(x_vel, 0)
            valid_ind = False
            for _ in range(x_vel + 1):
                sim.step()
                if x_min <= sim.x <= x_max:
                    valid_ind = True
                    break
            if valid_ind:
                valid_x_vels.add(x_vel)
        valid_y_vels = set()
        for y_vel in range(-100, 500 + 1):
            sim = self.Simulation(0, y_vel)
            valid_ind = False
            for _ in range(10_000):
                sim.step()
                if y_min <= sim.y <= y_max:
                    valid_ind = True
                    break
            if valid_ind:
                valid_y_vels.add(y_vel)
        valid_velocities = set()
        for x_vel in valid_x_vels:
            for y_vel in valid_y_vels:
                sim = self.Simulation(x_vel, y_vel)
                valid_ind = False
                for _ in range(10_000):
                    sim.step()
                    if x_min <= sim.x <= x_max and y_min <= sim.y <= y_max:
                        valid_ind = True
                    if sim.y < y_min and sim.y_vel < 0:
                        break
                if valid_ind:
                    valid_velocities.add((x_vel, y_vel))
        result = len(valid_velocities)
        return result
    
    def solve_slowly(self, x_min, x_max, y_min, y_max):
        highest_valid_y = float('-inf')
        for x_vel in range(-100, 100 + 1):
            for y_vel in range(-100, 100 + 1):
                sim = self.Simulation(x_vel, y_vel)
                valid_ind = False
                highest_y = float('-inf')
                for _ in range(10_000):
                    sim.step()
                    highest_y = max(highest_y, sim.y)
                    if x_min <= sim.x <= x_max and y_min <= sim.y <= y_max:
                        valid_ind = True
                    if sim.y < y_min and sim.y_vel < 0:
                        break
                if valid_ind:
                    highest_valid_y = max(highest_valid_y, highest_y)
        result = highest_valid_y
        return result
    
    def solve2_slowly(self, x_min, x_max, y_min, y_max):
        valid_velocities = set()
        for x_vel in range(-100, 500 + 1):
            for y_vel in range(-500, 500 + 1):
                sim = self.Simulation(x_vel, y_vel)
                valid_ind = False
                for _ in range(1_000):
                    sim.step()
                    if x_min <= sim.x <= x_max and y_min <= sim.y <= y_max:
                        valid_ind = True
                    if sim.y < y_min and sim.y_vel < 0:
                        break
                if valid_ind:
                    valid_velocities.add((x_vel, y_vel))
        result = len(valid_velocities)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        area = self.get_area(raw_input_lines)
        solutions = (
            self.solve(*area),
            self.solve2(*area),
            )
        result = solutions
        return result

class Day16: # Packet Decoder
    '''
    https://adventofcode.com/2021/day/16
    '''
    def get_transmission(self, raw_input_lines: List[str]):
        transmission = raw_input_lines[0]
        result = transmission
        return result
    
    class BitStream:
        def __init__(self, hexchars):
            self.versions = []
            self.bits = collections.deque()
            self.chars = hexchars
            for char in self.chars:
                num = int(char, 16)
                for power in (8, 4, 2, 1):
                    bit = (num & power) // power
                    self.bits.append(bit)
            self.cursor = 0
        
        def read_bits(self, size) -> int:
            num = 0
            for _ in range(size):
                bit = self.bits[self.cursor]
                num = 2 * num + bit
                self.cursor += 1
            result = num
            return result
        
        def read_packet(self) -> int:
            value = 0
            version = self.read_bits(3)
            self.versions.append(version)
            packet_type_id = self.read_bits(3)
            if packet_type_id == 4: # literal
                literal = 0
                while True:
                    group = self.read_bits(5)
                    literal = 16 * literal + (group & 15)
                    if group < 16:
                        break
                value = literal
            else: # operator
                sub_packets = []
                length_type_id = self.read_bits(1)
                if length_type_id == 0:
                    packet_length = self.read_bits(15)
                    end = self.cursor + packet_length
                    while self.cursor < end:
                        sub_packet = self.read_packet()
                        sub_packets.append(sub_packet)
                elif length_type_id == 1:
                    sub_packet_count = self.read_bits(11)
                    for _ in range(sub_packet_count):
                        sub_packet = self.read_packet()
                        sub_packets.append(sub_packet)
                else:
                    raise ValueError('Length type ID is invalid')
                if packet_type_id == 0:
                    value = sum(sub_packets)
                elif packet_type_id == 1:
                    value = 1
                    for sub_packet in sub_packets:
                        value *= sub_packet
                elif packet_type_id == 2:
                    value = min(sub_packets)
                elif packet_type_id == 3:
                    value = max(sub_packets)
                elif packet_type_id == 5:
                    assert len(sub_packets) == 2
                    value = 1 if sub_packets[0] > sub_packets[1] else 0
                elif packet_type_id == 6:
                    assert len(sub_packets) == 2
                    value = 1 if sub_packets[0] < sub_packets[1] else 0
                elif packet_type_id == 7:
                    assert len(sub_packets) == 2
                    value = 1 if sub_packets[0] == sub_packets[1] else 0
            result = value
            return result

    def solve(self, transmission):
        stream = self.BitStream(transmission)
        stream.read_packet()
        result = sum(stream.versions)
        return result
    
    def solve2(self, transmission):
        stream = self.BitStream(transmission)
        result = stream.read_packet()
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        transmission = self.get_transmission(raw_input_lines)
        solutions = (
            self.solve(transmission),
            self.solve2(transmission),
            )
        result = solutions
        return result

class Day15: # Chiton
    '''
    https://adventofcode.com/2021/day/15
    '''
    def get_grid(self, raw_input_lines: List[str]):
        grid = []
        for raw_input_line in raw_input_lines:
            grid.append(raw_input_line)
        result = grid
        return result
    
    def solve(self, grid):
        rows = len(grid)
        cols = len(grid[0])
        visits = {}
        work = [(0, 0, 0)] # (cost, row, col)
        min_cost = float('inf')
        while len(work) > 0:
            cost, row, col = heapq.heappop(work)
            if (row, col) == (rows - 1, cols - 1):
                min_cost = cost
                break
            if (row, col) in visits and visits[(row, col)] <= cost:
                continue
            visits[(row, col)] = cost
            for (nrow, ncol) in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if not (0 <= nrow < rows and 0 <= ncol < cols):
                    continue
                ncost = cost + int(grid[nrow][ncol])
                heapq.heappush(work, (ncost, nrow, ncol))
        result = min_cost
        return result
    
    def solve2(self, grid, size: int=5):
        subrows = len(grid)
        subcols = len(grid[0])
        rows = size * subrows
        cols = size * subcols
        visits = {}
        work = [(0, 0, 0)] # (cost, row, col)
        min_cost = float('inf')
        while len(work) > 0:
            cost, row, col = heapq.heappop(work)
            if (row, col) == (rows - 1, cols - 1):
                min_cost = cost
                break
            if (row, col) in visits and visits[(row, col)] <= cost:
                continue
            visits[(row, col)] = cost
            for (nrow, ncol) in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if not (0 <= nrow < rows and 0 <= ncol < cols):
                    continue
                r1, r2 = divmod(nrow, subrows)
                c1, c2 = divmod(ncol, subcols)
                cost2 = 1 + (int(grid[r2][c2]) + r1 + c1 - 1) % 9
                ncost = cost + cost2
                heapq.heappush(work, (ncost, nrow, ncol))
        result = min_cost
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        grid = self.get_grid(raw_input_lines)
        solutions = (
            self.solve(grid),
            self.solve2(grid),
            )
        result = solutions
        return result

class Day14: # Extended Polymerization
    '''
    https://adventofcode.com/2021/day/14
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        template = raw_input_lines[0]
        insertions = {}
        for raw_input_line in raw_input_lines[2:]:
            a, b = raw_input_line.split(' -> ')
            insertions[a] = b
        result = template, insertions
        return result
    
    def solve(self, template, insertions):
        for _ in range(10):
            next_template = []
            for i in range(1, len(template)):
                pair = template[i - 1:i + 1]
                if template[i - 1:i + 1] in insertions:
                    next_template.append(pair[0] + insertions[pair])
                else:
                    next_template.append(pair[0])
            next_template.append(template[-1])
            template = ''.join(next_template)
        counts = collections.Counter(template)
        max_count = float('-inf')
        min_count = float('inf')
        for count in counts.values():
            max_count = max(max_count, count)
            min_count = min(min_count, count)
        result = max_count - min_count
        return result
    
    def solve2_slowly(self, template, insertions):
        for _ in range(40):
            next_template = []
            for i in range(1, len(template)):
                pair = template[i - 1:i + 1]
                if template[i - 1:i + 1] in insertions:
                    next_template.append(pair[0] + insertions[pair])
                else:
                    next_template.append(pair[0])
            next_template.append(template[-1])
            template = ''.join(next_template)
        counts = collections.Counter(template)
        max_count = float('-inf')
        min_count = float('inf')
        for char, count in counts.items():
            max_count = max(max_count, count)
            min_count = min(min_count, count)
        result = max_count - min_count
        return result
    
    def solve2(self, template, insertions):
        pairs = collections.defaultdict(int)
        for i in range(1, len(template)):
            pair = template[i - 1:i + 1]
            pairs[pair] += 1
        for _ in range(40):
            next_pairs = collections.defaultdict(int)
            for pair, count in pairs.items():
                if pair in insertions:
                    a, b, c = pair[0], insertions[pair], pair[1]
                    next_pairs[a + b] += count
                    next_pairs[b + c] += count
                else:
                    next_pairs[pair] += count
            pairs = next_pairs
        counts = collections.defaultdict(int)
        for pair, count in pairs.items():
            counts[pair[0]] += count
        counts[template[-1]] += 1
        max_count = float('-inf')
        min_count = float('inf')
        for _, count in counts.items():
            max_count = max(max_count, count)
            min_count = min(min_count, count)
        result = max_count - min_count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        template, insertions = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(template, insertions),
            self.solve2(template, insertions),
            )
        result = solutions
        return result

class Day13: # Transparent Origami
    '''
    https://adventofcode.com/2021/day/13
    '''
    def get_dots_and_folds(self, raw_input_lines: List[str]):
        dots = set()
        folds = []
        result = []
        for raw_input_line in raw_input_lines:
            if len(raw_input_line) < 1:
                pass
            elif raw_input_line[0] == 'f':
                a, b = raw_input_line.split('=')
                fold = (a[-1], int(b))
                folds.append(fold)
            else:
                dot = tuple(map(int, raw_input_line.split(',')))
                dots.add(dot)
        result = (dots, folds)
        return result
    
    def solve(self, dots, folds):
        fold_axis, fold_line = folds[0]
        next_dots = set()
        for col, row in dots:
            next_dot = (col, row)
            if fold_axis == 'x' and col > fold_line:
                next_col = 2 * fold_line - col
                next_dot = (next_col, row)
            elif fold_axis == 'y' and row > fold_line:
                next_row = 2 * fold_line - row
                next_dot = (col, next_row)
            next_dots.add(next_dot)
        result = len(next_dots)
        return result
    
    def solve2(self, dots, folds):
        for fold_axis, fold_line in folds:
            next_dots = set()
            for col, row in dots:
                next_dot = (col, row)
                if fold_axis == 'x' and col > fold_line:
                    next_col = 2 * fold_line - col
                    next_dot = (next_col, row)
                elif fold_axis == 'y' and row > fold_line:
                    next_row = 2 * fold_line - row
                    next_dot = (col, next_row)
                next_dots.add(next_dot)
            dots = next_dots
        max_row = max(row for col, row in dots)
        max_col = max(col for col, row in dots)
        code = ['']
        for row in range(max_row + 1):
            line = []
            for col in range(max_col + 1):
                cell = '.'
                if (col, row) in dots:
                    cell = '#'
                line.append(cell)
            code.append(''.join(line))
        result = '\n'.join(code)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        dots, folds = self.get_dots_and_folds(raw_input_lines)
        solutions = (
            self.solve(copy.deepcopy(dots), folds),
            self.solve2(copy.deepcopy(dots), folds),
            )
        result = solutions
        return result

class Day12: # Passage Pathing
    '''
    https://adventofcode.com/2021/day/12
    '''
    def get_passages(self, raw_input_lines: List[str]):
        passages = collections.defaultdict(set)
        for raw_input_line in raw_input_lines:
            a, b = raw_input_line.split('-')
            passages[a].add(b)
            passages[b].add(a)
        result = passages
        return result
    
    def solve(self, passages):
        small_caves = set()
        for passage in passages:
            if passage[0] in 'abcdefghijklmnopqrstuvwxyz':
                small_caves.add(passage)
        paths = set()
        work = [(['start'], set())]
        while len(work) > 0:
            path, visited = work.pop()
            passage = path[-1]
            if passage == 'end':
                paths.add(tuple(path))
            visited.add(passage)
            for next_passage in passages[passage]:
                if next_passage in visited and next_passage in small_caves:
                    continue
                work.append((path + [next_passage], set(visited)))
        result = len(paths)
        return result
    
    def solve2(self, passages):
        small_caves = set()
        for passage in passages:
            if passage[0] in 'abcdefghijklmnopqrstuvwxyz':
                small_caves.add(passage)
        paths = set()
        work = [(['start'], set(), True)]
        while len(work) > 0:
            path, visited, spare_visit = work.pop()
            passage = path[-1]
            if passage == 'end':
                paths.add(tuple(path))
                continue
            visited.add(passage)
            for next_passage in passages[passage]:
                next_spare_visit = spare_visit
                if next_passage in visited and next_passage in small_caves:
                    if spare_visit and next_passage not in ('start', 'end'):
                        next_spare_visit = False
                    else:
                        continue
                work.append((
                    path + [next_passage],
                    set(visited),
                    next_spare_visit,
                ))
        result = len(paths)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        passages = self.get_passages(raw_input_lines)
        solutions = (
            self.solve(passages),
            self.solve2(passages),
            )
        result = solutions
        return result

class Day11: # Dumbo Octopus
    '''
    https://adventofcode.com/2021/day/11
    '''
    def get_octopuses(self, raw_input_lines: List[str]):
        octopuses = {}
        for row, raw_input_line in enumerate(raw_input_lines):
            for col, cell in enumerate(raw_input_line):
                octopuses[(row, col)] = int(cell)
        result = octopuses
        return result
    
    class Simulation:
        def __init__(self, octopuses):
            self.octopuses = copy.deepcopy(octopuses)
            self.flashes = set()

        def step(self):
            self.flashes = set()
            # Increase energies by 1
            for octopus in self.octopuses.keys():
                self.octopuses[octopus] += 1
                if self.octopuses[octopus] > 9:
                    self.flashes.add(octopus)
            # Propogate flashing of octopuses at 10+ energy
            flashes_left = set(self.flashes)
            while len(flashes_left) > 0:
                new_flashes = set()
                for (row, col) in flashes_left:
                    for (nrow, ncol) in (
                        (row - 1, col - 1),
                        (row - 1, col + 0),
                        (row - 1, col + 1),
                        (row + 0, col - 1),
                        (row + 0, col + 1),
                        (row + 1, col - 1),
                        (row + 1, col + 0),
                        (row + 1, col + 1),
                    ):
                        if (
                            (nrow, ncol) in self.octopuses and
                            (nrow, ncol) not in self.flashes and
                            (nrow, ncol) not in new_flashes
                        ):
                            self.octopuses[(nrow, ncol)] += 1
                            if self.octopuses[(nrow, ncol)] > 9:
                                new_flashes.add((nrow, ncol))
                self.flashes.update(new_flashes)
                flashes_left = new_flashes
            # Reset flashed octopuses to 0 energy
            for flash in self.flashes:
                self.octopuses[flash] = 0
    
    def solve(self, octopuses):
        flash_count = 0
        sim = self.Simulation(octopuses)
        for _ in range(100):
            sim.step()
            flash_count += len(sim.flashes)
        result = flash_count
        return result
    
    def solve2(self, octopuses):
        target_turn_id = None
        turn_id = 1
        sim = self.Simulation(octopuses)
        while True:
            sim.step()
            if len(sim.flashes) == len(octopuses):
                target_turn_id = turn_id
                break
            turn_id += 1
        result = target_turn_id
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        octopuses = self.get_octopuses(raw_input_lines)
        solutions = (
            self.solve(copy.deepcopy(octopuses)),
            self.solve2(copy.deepcopy(octopuses)),
            )
        result = solutions
        return result

class Day10: # Syntax Scoring
    '''
    https://adventofcode.com/2021/day/10
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, lines):
        values = {
            ')': 3,
            ']': 57,
            '}': 1197,
            '>': 25137,
        }
        pairs = {
            ')': '(',
            ']': '[',
            '}': '{',
            '>': '<',
        }
        score = 0
        for line in lines:
            stack = []
            for char in line:
                if char in '([{<':
                    stack.append(char)
                elif char in ')]}>':
                    if len(stack) < 1 or stack[-1] != pairs[char]:
                        score += values[char]
                        break
                    stack.pop()
        result = score
        return result
    
    def solve2(self, lines):
        values = {
            ')': 1,
            ']': 2,
            '}': 3,
            '>': 4,
        }
        opens = {
            ')': '(',
            ']': '[',
            '}': '{',
            '>': '<',
        }
        closes = {
            '(': ')',
            '[': ']',
            '{': '}',
            '<': '>',
        }
        scores = []
        for line in lines:
            corrupt_ind = False
            stack = []
            for char in line:
                if char in '([{<':
                    stack.append(char)
                elif char in ')]}>':
                    if len(stack) < 1 or stack[-1] != opens[char]:
                        corrupt_ind = True
                        break
                    stack.pop()
            if corrupt_ind:
                continue
            print(''.join(stack))
            score = 0
            while len(stack) > 0:
                char = closes[stack.pop()]
                score = 5 * score + values[char]
            scores.append(score)
        result = sorted(scores)[len(scores) // 2]
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        lines = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(lines),
            self.solve2(lines),
            )
        result = solutions
        return result

class Day09: # Smoke Basin
    '''
    https://adventofcode.com/2021/day/9
    '''
    def get_height_map(self, raw_input_lines: List[str]):
        height_map = []
        for raw_input_line in raw_input_lines:
            row_data = []
            for cell in raw_input_line:
                row_data.append(int(cell))
            height_map.append(row_data)
        result = height_map
        return result
    
    def solve(self, height_map):
        rows = len(height_map)
        cols = len(height_map[0])
        low_point_risk_levels = []
        for row in range(rows):
            for col in range(cols):
                low_point = height_map[row][col]
                for (nrow, ncol) in (
                    (row - 1, col    ),
                    (row + 1, col    ),
                    (row    , col - 1),
                    (row    , col + 1),
                ):
                    if not (0 <= nrow < rows and 0 <= ncol < cols):
                        continue
                    if height_map[nrow][ncol] <= low_point:
                        break
                else:
                    low_point_risk_levels.append(1 + low_point)
        result = sum(low_point_risk_levels)
        return result
    
    def solve2(self, height_map):
        rows = len(height_map)
        cols = len(height_map[0])
        unvisited = set()
        for row in range(rows):
            for col in range(cols):
                unvisited.add((row, col))
        basins = []
        while len(unvisited) > 0:
            row, col = unvisited.pop()
            if height_map[row][col] == 9:
                continue
            basin = 0
            work = [(row, col)]
            while len(work) > 0:
                row, col = work.pop()
                basin += 1
                for (nrow, ncol) in (
                    (row - 1, col    ),
                    (row + 1, col    ),
                    (row    , col - 1),
                    (row    , col + 1),
                ):
                    if (
                        0 <= nrow < rows and
                        0 <= ncol < cols and
                        (nrow, ncol) in unvisited and
                        height_map[nrow][ncol] != 9
                    ):
                        unvisited.remove((nrow, ncol))
                        work.append((nrow, ncol))
            basins.append(basin)
        basins.sort()
        result = basins[-1] * basins[-2] * basins[-3]
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        height_map = self.get_height_map(raw_input_lines)
        solutions = (
            self.solve(height_map),
            self.solve2(height_map),
            )
        result = solutions
        return result

class Day08: # Seven Segment Search
    '''
    https://adventofcode.com/2021/day/8

    2 of 7 segments (1):    1
    3 of 7 segments (1):                7
    4 of 7 segments (1):          4
    5 of 7 segments (3):      2 3   5
    6 of 7 segments (3):  0           6     9
    7 of 7 segments (1):  8

    Has canonical A:  0   2 3   5 6 7 8 9
    Has canonical B:  0       4 5 6   8 9
    Has canonical C:  0 1 2 3 4     7 8 9
    Has canonical D:      2 3 4 5 6   8 9
    Has canonical E:  0   2       6   8
    Has canonical F:  0 1   3 4 5 6 7 8 9
    Has canonical G:  0   2 3   5 6   8 9
    '''
    def get_entries(self, raw_input_lines: List[str]):
        entries = []
        for raw_input_line in raw_input_lines:
            parts = raw_input_line.split(' ')
            unique_signal_patterns = parts[:10]
            output_value = parts[-4:]
            entry = (unique_signal_patterns, output_value)
            entries.append(entry)
        result = entries
        return result
    
    def solve(self, entries):
        easy_digits = []
        for _, output_value in entries:
            for element in output_value:
                if len(element) in (2, 3, 4, 7):
                    easy_digits.append(element)
        result = len(easy_digits)
        return result
    
    def decode_output(self, patterns, output) -> int:
        digits_in_segment_count = {
            2: {'1', },
            3: {'7', },
            4: {'4', },
            5: {'2', '3', '5', },
            6: {'0', '6', '9', },
            7: {'8', },
        }
        # Let S be the non-canonical segments found in a given string
        # A string can be one of the following:
        #   a segment ('a' through 'f')
        #   a digit ('0' through '9')
        #   the intersection of multiple digits (e.g., '025')
        S = {}
        for segment_count, digits in digits_in_segment_count.items():
            code = ''.join(sorted(digits))
            if code not in S:
                # Start with all possible segments
                S[code] = set('abcdefg')
            for pattern in patterns:
                # Take the intersection of each pattern with the same length
                if len(pattern) == segment_count:
                    S[code] &= set(pattern)
        S['3'] = S['235'] | S['1']
        S['d'] = S['235'] & S['4']
        S['0'] = S['8'] - S['d']
        S['f'] = S['069'] & S['1']
        S['c'] = S['1'] - S['f']
        S['6'] = S['8'] - S['c']
        S['9'] = S['069'] | S['4']
        S['5'] = S['9'] - S['c']
        S['e'] = S['8'] - S['9']
        S['2'] = S['235'] | S['c'] | S['e']
        # Let D be the inverse of S for single digit lookups
        # i.e., the keys will be the segment keys that produce a digit value
        D = {}
        for digits, segments in S.items():
            if len(digits) == 1:
                key = ''.join(sorted(segments))
                D[key] = next(iter(digits))
        decoded_output = 0
        for segment_code in output:
            key = ''.join(sorted(set(segment_code)))
            decoded_output = 10 * decoded_output + int(D[key])
        result = decoded_output
        return result
    
    def solve2(self, entries):
        decoded_outputs = []
        for patterns, output in entries:
            decoded_output = self.decode_output(patterns, output)
            decoded_outputs.append(decoded_output)
        result = sum(decoded_outputs)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        entries = self.get_entries(raw_input_lines)
        solutions = (
            self.solve(entries),
            self.solve2(entries),
            )
        result = solutions
        return result

class Day07: # The Treachery of Whales
    '''
    https://adventofcode.com/2021/day/7
    '''
    def get_crabs(self, raw_input_lines: List[str]):
        result = list(map(int, raw_input_lines[0].split(',')))
        return result
    
    def solve(self, crabs):
        min_col = min(crabs)
        max_col = max(crabs)
        min_cost = float('inf')
        for target_col in range(min_col, max_col + 1):
            cost = 0
            for col in crabs:
                cost += abs(col - target_col)
            min_cost = min(min_cost, cost)
        result = min_cost
        return result
    
    def solve2(self, crabs):
        min_col = min(crabs)
        max_col = max(crabs)
        triangle_sums = [0]
        for num in range(1, abs(max_col - min_col + 1)):
            triangle_sums.append(triangle_sums[-1] + num)
        min_cost = float('inf')
        for target_col in range(min_col, max_col + 1):
            cost = 0
            for col in crabs:
                steps = abs(col - target_col)
                cost += triangle_sums[steps]
            min_cost = min(min_cost, cost)
        result = min_cost
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        crabs = self.get_crabs(raw_input_lines)
        solutions = (
            self.solve(crabs),
            self.solve2(crabs),
            )
        result = solutions
        return result

class Day06: # Lanternfish
    '''
    https://adventofcode.com/2021/day/6
    '''
    def get_starting_fish(self, raw_input_lines: List[str]):
        starting_fish = list(map(int, raw_input_lines[0].split(',')))
        result = starting_fish
        return result
    
    def solve(self, starting_fish, day_count):
        fish_timers = collections.defaultdict(int)
        for fish in starting_fish:
            fish_timers[fish] += 1
        for _ in range(day_count):
            next_fish_timers = collections.defaultdict(int)
            for timer, count in fish_timers.items():
                if timer == 0:
                    next_fish_timers[8] += count
                    next_fish_timers[6] += count
                else:
                    next_fish_timers[timer - 1] += count
            fish_timers = next_fish_timers
        result = sum(fish_timers.values())
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        starting_fish = self.get_starting_fish(raw_input_lines)
        solutions = (
            self.solve(starting_fish, 80),
            self.solve(starting_fish, 256),
            )
        result = solutions
        return result

class Day05: # Hydrothermal Venture
    '''
    https://adventofcode.com/2021/day/5
    '''
    def get_vents(self, raw_input_lines: List[str]):
        vents = []
        for raw_input_line in raw_input_lines:
            a, b = raw_input_line.split(' -> ')
            start = tuple(reversed(tuple(map(int, a.split(',')))))
            end = tuple(reversed(tuple(map(int, b.split(',')))))
            vents.append(tuple(sorted((start, end))))
        result = vents
        return result
    
    def solve(self, vents):
        orthogonal_vents = []
        for start, end in vents:
            if start[0] == end[0] or start[1] == end[1]:
                orthogonal_vents.append((start, end))
        points = collections.defaultdict(int)
        for start, end in orthogonal_vents:
            for row in range(start[0], end[0] + 1):
                for col in range(start[1], end[1] + 1):
                    points[(row, col)] += 1
        result = sum(1 for count in points.values() if count > 1)
        return result
    
    def solve2(self, vents):
        points = collections.defaultdict(int)
        for start, end in vents:
            row_delta = 0
            if end[0] > start[0]:
                row_delta = 1
            elif end[0] < start[0]:
                row_delta = -1
            col_delta = 0
            if end[1] > start[1]:
                col_delta = 1
            elif end[1] < start[1]:
                col_delta = -1
            distance = 1 + max(abs(end[0] - start[0]), abs(end[1] - start[1]))
            for step in range(distance):
                row = start[0] + step * row_delta
                col = start[1] + step * col_delta
                points[(row, col)] += 1
        result = sum(1 for count in points.values() if count > 1)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        vents = self.get_vents(raw_input_lines)
        solutions = (
            self.solve(vents),
            self.solve2(vents),
            )
        result = solutions
        return result

class Day04: # Giant Squid
    '''
    https://adventofcode.com/2021/day/4
    '''
    def get_bingo_subsystem(self, raw_input_lines: List[str]):
        numbers = []
        for num_str in raw_input_lines[0].split(','):
            number = int(num_str)
            numbers.append(number)
        boards = []
        board = []
        for raw_input_line in raw_input_lines[1:]:
            if len(raw_input_line) == 0:
                if len(board) > 0:
                    boards.append(board)
                board = []
            else:
                board_row = list(map(int, raw_input_line.split()))
                board.append(board_row)
        if len(board) > 0:
            boards.append(board)
        result = (boards, numbers)
        return result
    
    def solve(self, boards, numbers):
        rows = len(boards[0])
        cols = len(boards[0][0])
        called_numbers = set()
        winning_board_id = -1
        latest_number = -1
        for latest_number in numbers:
            called_numbers.add(latest_number)
            for board_id in range(len(boards)):
                for row in range(rows):
                    nums = set(boards[board_id][row])
                    if len(nums & called_numbers) == rows:
                        winning_board_id = board_id
                        break
                for col in range(cols):
                    nums = set()
                    for row in range(rows):
                        nums.add(boards[board_id][row][col])
                    if len(nums & called_numbers) == rows:
                        winning_board_id = board_id
                        break
            if winning_board_id >= 0:
                break
        winning_board_score = 0
        for row in range(rows):
            for col in range(cols):
                if boards[winning_board_id][row][col] not in called_numbers:
                    winning_board_score += boards[winning_board_id][row][col]
        result = latest_number * winning_board_score
        return result
    
    def solve2(self, boards, numbers):
        rows = len(boards[0])
        cols = len(boards[0][0])
        called_numbers = set()
        winning_board_ids = set()
        latest_win = {
            'board_id': -1,
            'round_id': -1,
            'number': -1,
        }
        for round_id, number in enumerate(numbers, start=1):
            called_numbers.add(number)
            for board_id in range(len(boards)):
                if board_id in winning_board_ids:
                    continue
                for row in range(rows):
                    nums = set(boards[board_id][row])
                    if len(nums & called_numbers) == rows:
                        latest_win = {
                            'board_id': board_id,
                            'round_id': round_id,
                            'number': number,
                        }
                        winning_board_ids.add(board_id)
                for col in range(cols):
                    nums = set()
                    for row in range(rows):
                        nums.add(boards[board_id][row][col])
                    if len(nums & called_numbers) == rows:
                        latest_win = {
                            'board_id': board_id,
                            'round_id': round_id,
                            'number': number,
                        }
                        winning_board_ids.add(board_id)
        called_numbers = set(numbers[:latest_win['round_id']])
        latest_winning_board_score = 0
        for row in range(rows):
            for col in range(cols):
                number = boards[latest_win['board_id']][row][col]
                if number not in called_numbers:
                    latest_winning_board_score += number
        result = latest_win['number'] * latest_winning_board_score
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        boards, numbers = self.get_bingo_subsystem(raw_input_lines)
        solutions = (
            self.solve(boards, numbers),
            self.solve2(boards, numbers),
            )
        result = solutions
        return result

class Day03: # Binary Diagnostic
    '''
    https://adventofcode.com/2021/day/3
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, parsed_input):
        N = len(parsed_input)
        counts = [0] * len(parsed_input[0])
        for line in parsed_input:
            for i, char in enumerate(line):
                if char == '1':
                    counts[i] += 1
        gamma_chars = []
        epsilon_chars = []
        for i, count in enumerate(counts):
            if count >= N // 2:
                gamma_chars.append('1')
                epsilon_chars.append('0')
            else:
                gamma_chars.append('0')
                epsilon_chars.append('1')
        gamma = int(''.join(gamma_chars), 2)
        epsilon = int(''.join(epsilon_chars), 2)
        power = gamma * epsilon
        result = power
        return result
    
    def solve2(self, parsed_input):
        N = len(parsed_input[0])
        nums = sorted(parsed_input)
        # Calculate oxygen generator rating
        # majority per bit, 1s win in ties
        left = 0
        right = len(nums)
        for i in range(N):
            counts = [0, 0]
            for num in nums[left:right]:
                digit = int(num[i])
                counts[digit] += 1
            assert sum(counts) == right - left
            if counts[1] >= counts[0]:
                left += counts[0]
            else:
                right -= counts[1]
        oxy = int(nums[left], 2)
        assert left == right - 1
        # Calculate C02 scrubber rating
        # minority per bit, 0s win in ties
        left = 0
        right = len(nums)
        for i in range(N):
            if left == right - 1:
                break
            counts = [0, 0]
            for num in nums[left:right]:
                digit = int(num[i])
                counts[digit] += 1
            assert sum(counts) == right - left
            if counts[0] <= counts[1]:
                right -= counts[1]
            else:
                left += counts[0]
        co2 = int(nums[left], 2)
        assert left == right - 1
        life_support = oxy * co2
        result = life_support
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

class Day02: # Dive!
    '''
    https://adventofcode.com/2021/day/2
    '''
    def get_commands(self, raw_input_lines: List[str]):
        commands = []
        for raw_input_line in raw_input_lines:
            command, raw_amount = raw_input_line.split(' ')
            amount = int(raw_amount)
            commands.append((command, amount))
        result = commands
        return result
    
    def solve(self, commands):
        x_pos = 0
        depth = 0
        for command, amount in commands:
            if command == 'forward':
                x_pos += amount
            elif command == 'down':
                depth += amount
            elif command == 'up':
                depth -= amount
        result = x_pos * depth
        return result
    
    def solve2(self, commands):
        x_pos = 0
        depth = 0
        aim = 0
        for command, amount in commands:
            if command == 'forward':
                x_pos += amount
                depth += aim * amount
            elif command == 'down':
                aim += amount
            elif command == 'up':
                aim -= amount
        result = x_pos * depth
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        commands = self.get_commands(raw_input_lines)
        solutions = (
            self.solve(commands),
            self.solve2(commands),
            )
        result = solutions
        return result

class Day01: # Sonar Sweep
    '''
    https://adventofcode.com/2021/day/1
    '''
    def get_sonar_sweeps(self, raw_input_lines: List[str]):
        sonar_sweeps = []
        for raw_input_line in raw_input_lines:
            sonar_sweeps.append(int(raw_input_line))
        result = sonar_sweeps
        return result
    
    def solve(self, sonar_sweeps):
        count = 0
        for i in range(1, len(sonar_sweeps)):
            if sonar_sweeps[i] > sonar_sweeps[i - 1]:
                count += 1
        result = count
        return result
    
    def solve2(self, sonar_sweeps):
        count = 0
        a, b, c = sonar_sweeps[0], sonar_sweeps[1], sonar_sweeps[2]
        for i in range(3, len(sonar_sweeps)):
            d = sonar_sweeps[i]
            if d > a:
                count += 1
            a, b, c = b, c, d
        result = count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        sonar_sweeps = self.get_sonar_sweeps(raw_input_lines)
        solutions = (
            self.solve(sonar_sweeps),
            self.solve2(sonar_sweeps),
            )
        result = solutions
        return result

if __name__ == '__main__':
    '''
    Usage
    python AdventOfCode2021.py 1 < inputs/2021day01.in
    '''
    solvers = {
        1: (Day01, 'Sonar Sweep'),
        2: (Day02, 'Dive!'),
        3: (Day03, 'Binary Diagnostic'),
        4: (Day04, 'Giant Squid'),
        5: (Day05, 'Hydrothermal Venture'),
        6: (Day06, 'Lanternfish'),
        7: (Day07, 'The Treachery of Whales'),
        8: (Day08, 'Seven Segment Search'),
        9: (Day09, 'Smoke Basin'),
       10: (Day10, 'Syntax Scoring'),
       11: (Day11, 'Dumbo Octopus'),
       12: (Day12, 'Passage Pathing'),
       13: (Day13, 'Transparent Origami'),
       14: (Day14, 'Extended Polymerization'),
       15: (Day15, 'Chiton'),
       16: (Day16, 'Packet Decoder'),
       17: (Day17, 'Trick Shot'),
       18: (Day18, 'Snailfish'),
       19: (Day19, 'Beacon Scanner'),
       20: (Day20, 'Trench Map'),
       21: (Day21, 'Dirac Dice'),
       22: (Day22, 'Reactor Reboot'),
       23: (Day23, 'Amphipod'),
       24: (Day24, 'Arithmetic Logic Unit'),
       25: (Day25, 'Sea Cucumber'),
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
