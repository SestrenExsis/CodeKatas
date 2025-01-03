'''
Created on 2024-11-30

@author: Sestren
'''
import argparse
import copy
import functools
import heapq

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

class ChronospatialComputer:
    def __init__(self, registers, program):
        self.registers = registers
        self.program = program
        self.pc = 0
        self.outputs = []
        self.debug = []

    def __str__(self):
        pc = self.pc
        r = tuple(map(str, (self.registers['A'], self.registers['B'], self.registers['C'])))
        registers = 'A: ' + r[0] + ', B: ' + r[1] + ', C: ' + r[2]
        outputs = ','.join(map(str, self.outputs))
        result = str((pc, registers, outputs))
        return result
    
    def clone(self):
        clone = ChronospatialComputer(self.registers, self.program)
        clone.pc = self.pc
        result = clone
        return result
    
    def combo(self, operand) -> int:
        assert 0 <= operand <= 6
        values = [
            0,
            1,
            2,
            3,
            self.registers['A'],
            self.registers['B'],
            self.registers['C'],
        ]
        result = values[operand]
        return result
    
    def step(self):
        if self.pc >= len(self.program):
            self.debug.append(('HALT', '-', str(self)))
            return -1
        (opcode, operand) = (self.program[self.pc], self.program[self.pc + 1])
        if opcode == 0: # ADV: division A
            value = 2 ** self.combo(operand)
            self.registers['A'] = self.registers['A'] // value
            self.pc += 2
            self.debug.append(('ADV', operand, str(self)))
        elif opcode == 1: # BXL: bitwise XOR literal
            value = operand
            self.registers['B'] = self.registers['B'] ^ operand
            self.pc += 2
            self.debug.append(('BXL', operand, str(self)))
        elif opcode == 2: # BST: ???
            value = self.combo(operand) % 8
            self.registers['B'] = value
            self.pc += 2
            self.debug.append(('BST', operand, str(self)))
        elif opcode == 3: # JNZ: jump if not zero
            if self.registers['A'] != 0:
                value = operand
                self.pc = value
            else:
                self.pc += 2
            self.debug.append(('JNZ', operand, str(self)))
        elif opcode == 4: # BXC: bitwise XOR C
            _ = operand # Ignored
            self.registers['B'] = self.registers['B'] ^ self.registers['C']
            self.pc += 2
            self.debug.append(('BXC', operand, str(self)))
        elif opcode == 5: # OUT: output
            value = self.combo(operand) % 8
            self.outputs.append(value)
            self.pc += 2
            self.debug.append(('OUT', operand, str(self)))
        elif opcode == 6: # BDV: division B
            value = 2 ** self.combo(operand)
            self.registers['B'] = self.registers['A'] // value
            self.pc += 2
            self.debug.append(('BDV', operand, str(self)))
        elif opcode == 7: # CDV: division C
            value = 2 ** self.combo(operand)
            self.registers['C'] = self.registers['A'] // value
            self.pc += 2
            self.debug.append(('CDV', operand, str(self)))
        return 0
    
    def next(self):
        result = None
        count = len(self.outputs)
        while True:
            return_code = self.step()
            if return_code != 0:
                break
            if len(self.outputs) > count:
                result = self.outputs[-1]
                assert len(self.outputs) == (count + 1)
                break
        return result

    def run(self):
        return_code = 0
        while True:
            return_code = self.step()
            if return_code != 0:
                break
        result = return_code
        return result

class Day18: # RAM Run
    '''
    https://adventofcode.com/2024/day/18
    '''
    def get_parsed_input(self, raw_input_lines: list[str]):
        falling_bytes = []
        (rows, cols) = (7, 7)
        for raw_input_line in raw_input_lines:
            (col, row) = tuple(map(int, raw_input_line.split(',')))
            if col >= 7 or row >= 7:
                rows = 71
                cols = 71
            falling_bytes.append((row, col))
        result = (rows, cols, falling_bytes)
        return result
    
    def show(self, rows, cols, step_count, falling_bytes, steps):
        result = []
        for row in range(rows):
            row_data = []
            for col in range(cols):
                char = '.'
                if (row, col) in falling_bytes[:step_count]:
                    char = '#'
                if (row, col) in steps:
                    if char == '#':
                        char = 'X'
                    else:
                        char = 'O'
                row_data.append(char)
            result.append(''.join(row_data))
        return result
    
    def solve(self, rows, cols, falling_bytes):
        falling_bytes_set = set(falling_bytes)
        assert (rows - 1, cols -1) not in falling_bytes_set
        min_step_count = None
        seen = set()
        work = [] # (step_count, (row, col))
        heapq.heappush(work, (0, (0, 0), set()))
        while len(work) > 0:
            (step_count, (row, col), steps) = heapq.heappop(work)
            steps.add((row, col))
            # print(step_count, (row, col))
            if (row, col) == (rows - 1, cols -1):
                min_step_count = step_count
                # for line in self.show(rows, cols, step_count, falling_bytes, steps):
                #     print(line)
                break
            if (row, col) in falling_bytes_set:
                N = min(len(falling_bytes), 1024)
                if (row, col) in set(falling_bytes[:N]):
                    # print('blocked:', N, (row, col))
                    continue
            if (row, col) in seen:
                continue
            seen.add((row, col))
            for (next_row, next_col) in (
                (row - 1, col    ),
                (row    , col - 1),
                (row + 1, col    ),
                (row    , col + 1),
            ):
                if not ((0 <= next_row < rows) and (0 <= next_col < cols)):
                    continue
                safe_ind = True
                if safe_ind and (next_row, next_col) not in steps:
                    heapq.heappush(work,
                        (step_count + 1, (next_row, next_col), set(steps))
                    )
        result = min_step_count
        return result
    
    def solve2_slowly(self, rows, cols, falling_bytes):
        falling_bytes_set = set(falling_bytes)
        delay = 1024
        exit_reachable = True
        while exit_reachable:
            print(delay)
            exit_reachable = False
            seen = set()
            work = []
            heapq.heappush(work, (0, (0, 0)))
            while len(work) > 0:
                (step_count, (row, col)) = heapq.heappop(work)
                if (row, col) == (rows - 1, cols -1):
                    exit_reachable = True
                    break
                if (row, col) in falling_bytes_set:
                    N = min(len(falling_bytes), delay + 2)
                    if (row, col) in set(falling_bytes[:N]):
                        continue
                if (row, col) in seen:
                    continue
                seen.add((row, col))
                for (next_row, next_col) in (
                    (row - 1, col    ),
                    (row    , col - 1),
                    (row + 1, col    ),
                    (row    , col + 1),
                ):
                    if not ((0 <= next_row < rows) and (0 <= next_col < cols)):
                        continue
                    safe_ind = True
                    if safe_ind:
                        heapq.heappush(work, (step_count + 1, (next_row, next_col)))
            delay += 1
        (block_row, block_col) = falling_bytes[delay]
        result = ','.join(map(str, (block_col, block_row)))
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        (rows, cols, falling_bytes) = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(rows, cols, falling_bytes),
            self.solve2_slowly(rows, cols, falling_bytes),
        )
        result = solutions
        return result

class Day17: # Chronospatial Computer
    '''
    https://adventofcode.com/2024/day/17
    '''
    def get_computer(self, raw_input_lines: list[str]):
        registers = {}
        program = []
        for raw_input_line in raw_input_lines:
            if raw_input_line.startswith('Register'):
                (left, right) = raw_input_line.split(': ')
                register_key = left[-1]
                register_value = int(right)
                registers[register_key] = register_value
            elif raw_input_line.startswith('Program'):
                (left, right) = raw_input_line.split(': ')
                program = list(map(int, right.split(',')))
        result = ChronospatialComputer(registers, program)
        return result
    
    def solve(self, computer):
        computer.run()
        result = ','.join(map(str, computer.outputs))
        # for line in computer.debug:
        #     print(line)
        return result
    
    def solve2_slowly(self, computer):
        a = 0
        while True:
            if a % 10_000 == 0:
                print(a)
            clone = computer.clone()
            clone.registers['A'] = a
            for num in clone.program:
                if clone.next() != num:
                    # print(num, clone.outputs)
                    break
            else:
                clone.run()
                if clone.outputs == clone.program:
                    break
            # print(','.join(map(str, computer.outputs)))
            a += 1
        result = a
        return result
    
    def solve2_in_reverse(self, computer):
        '''
        From observing the inputs given, the following appears to be true:
        - The program is always in a while loop structure that ends when A is 0
        - The output is always going to be equivalent to the values when you
          iteratively divide A by 8 and output the modulo 8 of A
        - The values of B and C are never carried over between loop iterations
        
        As a result, this means that we can use a reversed approach to find the
        correct value of A by computing possible values for, in order:
        - A % 8
        - (A // 8) % 8
        - (A // 64) % 8
        - and so on
        '''
        result = None
        targets = computer.program
        work = [] # (negative_progress, A)
        heapq.heappush(work, (0, 0))
        while len(work) > 0:
            (negative_progress, A) = heapq.heappop(work)
            progress = -negative_progress
            if progress == len(targets):
                result = A
                break
            for num in range(8):
                clone = computer.clone()
                clone.registers['A'] = 8 * A + num
                clone.run()
                if clone.outputs == targets[-progress - 1:]:
                    heapq.heappush(work, (-progress - 1, 8 * A + num))
        # Confirm the correct answer has been found
        clone = computer.clone()
        clone.registers['A'] = result
        clone.run()
        if clone.outputs != targets:
            raise Exception('Wrong answer!')
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        computer = self.get_computer(raw_input_lines)
        solutions = (
            self.solve(computer.clone()),
            self.solve2_in_reverse(computer.clone()),
        )
        result = solutions
        return result

class Day16: # Reindeer Maze
    DIRECTIONS = {
        'NORTH': (-1,  0, 'WEST', 'EAST'),
        'EAST':  ( 0,  1, 'NORTH', 'SOUTH'),
        'SOUTH': ( 1,  0, 'EAST', 'WEST'),
        'WEST':  ( 0, -1, 'SOUTH', 'NORTH'),
    }
    '''
    https://adventofcode.com/2024/day/16
    '''
    def get_parsed_input(self, raw_input_lines: list[str]):
        walls = set()
        result = []
        start = None
        end = None
        for (row, raw_input_line) in enumerate(raw_input_lines):
            for (col, char) in enumerate(raw_input_line):
                if char == 'S':
                    start = (row, col, 'EAST')
                elif char == 'E':
                    end = (row, col)
                elif char == '#':
                    walls.add((row, col))
        result = (walls, start, end)
        return result
    
    def solve(self, walls, start, end):
        best_score = float('inf')
        seen = {}
        work = []
        heapq.heappush(work, (0, start))
        while len(work) > 0:
            (score, (row, col, facing)) = heapq.heappop(work)
            if (row, col) == end:
                best_score = score
                break
            elif (row, col, facing) in seen and seen[(row, col, facing)] <= score:
                continue
            else:
                seen[(row, col, facing)] = score
                (offset_row, offset_col, turn_left, turn_right) = self.DIRECTIONS[facing]
                (next_row, next_col) = (row + offset_row, col + offset_col)
                if (next_row, next_col) not in walls:
                    heapq.heappush(work, (score + 1, (next_row, next_col, facing)))
                heapq.heappush(work, (score + 1000, (row, col, turn_left)))
                heapq.heappush(work, (score + 1000, (row, col, turn_right)))
        result = best_score
        return result
    
    def show(self, walls, path_tiles):
        cols = 0
        while (0, cols) in walls:
            cols += 1
        result = []
        row = 0
        while True:
            cell_count = 0
            row_data = []
            for col in range(cols):
                char = '.'
                if (row, col) in walls:
                    char = '#'
                    cell_count += 1
                if (row, col) in path_tiles:
                    assert (row, col) not in walls
                    char = 'O'
                    cell_count += 1
                row_data.append(char)
            result.append(''.join(row_data))
            row += 1
            if cell_count < 1:
                break
        for (row, col) in path_tiles:
            if 0 <= row <= len(result) and 0 <= col <= cols:
                continue
            result.append('bad tile:', (row, col))
        return result
    
    def solve2(self, walls, start, end, best_score):
        best_path_tiles = set()
        seen = {}
        work = []
        heapq.heappush(work, (0, start, set()))
        while len(work) > 0:
            (score, (row, col, facing), tiles) = heapq.heappop(work)
            tiles.add((row, col))
            if (row, col) == end:
                if score == best_score:
                    best_path_tiles |= tiles
                    continue
            elif score > best_score:
                continue
            elif (row, col, facing) in seen and seen[(row, col, facing)] < score:
                continue
            else:
                seen[(row, col, facing)] = score
                (offset_row, offset_col, turn_left, turn_right) = self.DIRECTIONS[facing]
                (next_row, next_col) = (row + offset_row, col + offset_col)
                if (next_row, next_col) not in walls:
                    next_tiles = set(tiles)
                    next_tiles.add((next_row, next_col))
                    heapq.heappush(work, (score + 1, (next_row, next_col, facing), next_tiles))
                heapq.heappush(work, (score + 1000, (row, col, turn_left), tiles))
                heapq.heappush(work, (score + 1000, (row, col, turn_right), tiles))
        result = len(best_path_tiles)
        # for line in self.show(walls, best_path_tiles):
        #     print(line)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        (walls, start, end) = self.get_parsed_input(raw_input_lines)
        best_score = self.solve(walls, start, end)
        solutions = (
            best_score,
            self.solve2(walls, start, end, best_score),
        )
        result = solutions
        return result

class Day15: # Warehouse Woes
    '''
    https://adventofcode.com/2024/day/15
    '''
    def get_parsed_input(self, raw_input_lines: list[str]):
        warehouse = {}
        robot = (0, 0)
        moves = []
        mode = 'WAREHOUSE'
        for (row, raw_input_line) in enumerate(raw_input_lines):
            if len(raw_input_line) < 1:
                mode = 'MOVES'
                continue
            if mode == 'WAREHOUSE':
                for (col, char) in enumerate(raw_input_line):
                    if char in '#O':
                        warehouse[(row, col)] = char
                    elif char == '@':
                        robot = (row, col)
            else:
                for char in raw_input_line:
                    if char == '^':
                        moves.append((-1,  0))
                    elif char == 'v':
                        moves.append(( 1,  0))
                    elif char == '<':
                        moves.append(( 0, -1))
                    elif char == '>':
                        moves.append(( 0,  1))
        result = (warehouse, robot, moves)
        return result
    
    def solve(self, warehouse, robot, moves):
        (robot_row, robot_col) = robot
        for move in moves:
            (row_offset, col_offset) = move
            # Find first open square in the move direction
            distance = 1
            wall_found = False
            boxes_to_move = set()
            while True:
                row = robot_row + distance * row_offset
                col = robot_col + distance * col_offset
                if (row, col) in warehouse and warehouse[(row, col)] == '#':
                    wall_found = True
                    break
                if (row, col) in warehouse and warehouse[(row, col)] == 'O':
                    boxes_to_move.add((row, col))
                if (row, col) not in warehouse:
                    break
                distance += 1
            # If a wall is involved in the chain, we can't move anything
            if wall_found:
                continue
            # Move all boxes encountered in the given direction
            for (row, col) in boxes_to_move:
                warehouse.pop((row, col))
            for (row, col) in boxes_to_move:
                warehouse[(row + row_offset , col + col_offset)] = 'O'
            robot_row += row_offset
            robot_col += col_offset
        result = sum(
            (100 * row + col) for (row, col) in warehouse if 
            warehouse[(row, col)] == 'O'
        )
        return result

    def show(self, warehouse, robot):
        cols = 0
        while (0, cols) in warehouse:
            cols += 1
        result = []
        row = 0
        while True:
            cell_count = 0
            row_data = []
            for col in range(cols):
                char = '.'
                if (row, col) in warehouse:
                    char = warehouse[(row, col)]
                    cell_count += 1
                if (row, col) == robot:
                    char = '@'
                row_data.append(char)
            result.append(''.join(row_data))
            row += 1
            if cell_count < 1:
                break
        return result
    
    def solve2(self, warehouse, robot, moves):
        # Double the width of the warehouse and boxes in it
        new_warehouse = {}
        for ((row, col), char) in warehouse.items():
            left = char
            right = char
            if char == 'O':
                left = '['
                right = ']'
            new_warehouse[(row, 2 * col)] = left
            new_warehouse[(row, 2 * col + 1)] = right
        warehouse = new_warehouse
        (robot_row, robot_col) = robot
        robot_col *= 2
        # Process robot's moves
        for (row_offset, col_offset) in moves:
            # Try to move in the given direction
            work = []
            work.append((robot_row, robot_col))
            valid_ind = True
            pushed = {}
            while len(work) > 0:
                (row, col) = work.pop()
                (next_row, next_col) = (row + row_offset, col + col_offset)
                if (next_row, next_col) in warehouse:
                    if warehouse[(next_row, next_col)] == '#':
                        valid_ind = False
                        break
                    elif warehouse[(next_row, next_col)] == '[':
                        if (next_row, next_col) not in pushed:
                            work.append((next_row, next_col))
                            pushed[(next_row, next_col)] = '['
                        if (next_row, next_col + 1) not in pushed:
                            work.append((next_row, next_col + 1))
                            pushed[(next_row, next_col + 1)] = ']'
                    elif warehouse[(next_row, next_col)] == ']':
                        if (next_row, next_col - 1) not in pushed:
                            work.append((next_row, next_col - 1))
                            pushed[(next_row, next_col - 1)] = '['
                        if (next_row, next_col) not in pushed:
                            work.append((next_row, next_col))
                            pushed[(next_row, next_col)] = ']'
            # Only move everything if it is valid
            if not valid_ind:
                continue
            robot_row += row_offset
            robot_col += col_offset
            for ((row, col), char) in pushed.items():
                warehouse.pop((row, col))
            for ((row, col), char) in pushed.items():
                warehouse[(row + row_offset, col + col_offset)] = char
        result = sum(
            (100 * row + col) for (row, col) in warehouse if 
            warehouse[(row, col)] == '['
        )
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        (warehouse, robot, moves) = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(copy.deepcopy(warehouse), robot, moves),
            self.solve2(copy.deepcopy(warehouse), robot, moves),
        )
        result = solutions
        return result

class Day14: # Restroom Redoubt
    '''
    https://adventofcode.com/2024/day/14
    '''
    def get_robots(self, raw_input_lines: list[str]):
        robots = []
        rows = 7
        cols = 11
        for raw_input_line in raw_input_lines:
            (raw_pos, raw_vel) = raw_input_line.split(' ')
            (pos_x, pos_y) = tuple(map(int, raw_pos[2:].split(',')))
            (vel_x, vel_y) = tuple(map(int, raw_vel[2:].split(',')))
            robots.append(((pos_y, pos_x), (vel_y, vel_x)))
            if pos_y > rows:
                rows = 103
            if pos_x > cols:
                cols = 101
        result = (robots, rows, cols)
        return result
    
    def show(self, robots, rows, cols):
        error = 0
        canvas = []
        midlines = (rows // 2, cols // 2)
        for row in range(rows):
            row_data = []
            for col in range(cols):
                robot_count = 0
                for ((pos_y, pos_x), (vel_y, vel_x)) in robots:
                    if pos_y == row and pos_x == col:
                        robot_count += 1
                char = ' '
                if row == midlines[0] or col == midlines[1]:
                    char = ' '
                if robot_count > 0:
                    char = str(robot_count % 10)
                row_data.append(char)
            canvas.append(''.join(row_data))
            error = max(error, sum(-1 for char in row_data if char != ' '))
        result = (error, canvas)
        return result
    
    def solve(self, robots, rows, cols):
        # self.show(robots, rows, cols)
        for _ in range(100):
            for i in range(len(robots)):
                ((pos_y, pos_x), (vel_y, vel_x)) = robots[i]
                new_pos_y = (pos_y + vel_y) % rows
                new_pos_x = (pos_x + vel_x) % cols
                robots[i] = ((new_pos_y, new_pos_x), (vel_y, vel_x))
        # self.show(robots, rows, cols)
        midlines = (rows // 2, cols // 2)
        upper_left = sum(
            1 for ((pos_y, pos_x), (_, _)) in robots if 
            (
                pos_y < midlines[0] and
                pos_x < midlines[1]
            )
        )
        upper_right = sum(
            1 for ((pos_y, pos_x), (_, _)) in robots if 
            (
                pos_y < midlines[0] and
                pos_x > midlines[1]
            )
        )
        lower_left = sum(
            1 for ((pos_y, pos_x), (_, _)) in robots if 
            (
                pos_y > midlines[0] and
                pos_x < midlines[1]
            )
        )
        lower_right = sum(
            1 for ((pos_y, pos_x), (_, _)) in robots if 
            (
                pos_y > midlines[0] and
                pos_x > midlines[1]
            )
        )
        result = upper_left * upper_right * lower_left * lower_right
        return result
    
    def solve2(self, robots, rows, cols):
        pixels = {}
        for ((pos_y, pos_x), (vel_y, vel_x)) in robots:
            if (pos_y, pos_x) not in pixels:
                pixels[(pos_y, pos_x)] = []
            pixels[(pos_y, pos_x)].append((vel_y, vel_x))
        max_symmetry = (0, None, None)
        for seconds_elapsed in range(1, 10_000 + 1):
            next_pixels = {}
            for ((pos_y, pos_x), velocities) in pixels.items():
                for (vel_y, vel_x) in velocities:
                    new_pos_y = (pos_y + vel_y) % rows
                    new_pos_x = (pos_x + vel_x) % cols
                    if (new_pos_y, new_pos_x) not in next_pixels:
                        next_pixels[(new_pos_y, new_pos_x)] = []
                    next_pixels[(new_pos_y, new_pos_x)].append((vel_y, vel_x))
            pixels = next_pixels
            # Test for symmetry
            symmetry = 0
            for (row, col) in pixels:
                if (row, cols - col - 1) in pixels:
                    symmetry += 0.5
            if symmetry > max_symmetry[0]:
                max_symmetry = (symmetry, seconds_elapsed, pixels)
        # Display the Easter egg, because it's cute!
        for row in range(rows):
            row_data = []
            for col in range(cols):
                char = '.'
                if (row, col) in max_symmetry[2]:
                    char = '#'
                row_data.append(char)
            print(''.join(row_data))
        result = max_symmetry[1]
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        (robots, rows, cols) = self.get_robots(raw_input_lines)
        solutions = (
            self.solve(copy.deepcopy(robots), rows, cols),
            self.solve2(copy.deepcopy(robots), rows, cols),
        )
        assert solutions[0] < 225509760
        result = solutions
        return result

class Day13: # Claw Contraption
    '''
    https://adventofcode.com/2024/day/13
    '''
    def get_claw_machines(self, raw_input_lines: list[str]):
        claw_machines = []
        claw_machine = {}
        mode = None
        for raw_input_line in raw_input_lines:
            if len(raw_input_line) > 0:
                (mode, data) = raw_input_line.split(': ')
                (raw_x, raw_y) = data.split(', ')
                if mode == 'Prize':
                    x = int(raw_x[2:])
                    y = int(raw_y[2:])
                    assert x > 0
                    assert y > 0
                    claw_machine[mode] = {
                        'X': x,
                        'Y': y,
                    }
                    claw_machines.append(claw_machine)
                    claw_machine = {}
                else:
                    x = int(raw_x[1:])
                    y = int(raw_y[1:])
                    assert x > 0
                    assert y > 0
                    claw_machine[mode] = {
                        'X': x,
                        'Y': y,
                    }
        result = claw_machines
        return result
    
    def solve(self, claw_machines):
        MAX_PRESSES = 100
        tokens_needed = []
        for claw_machine in claw_machines:
            target_xy = (claw_machine['Prize']['X'], claw_machine['Prize']['Y'])
            possible_tokens_used = set()
            for a_presses in range(MAX_PRESSES + 1):
                for b_presses in range(MAX_PRESSES + 1):
                    (x, y) = (0, 0)
                    x += a_presses * claw_machine['Button A']['X']
                    y += a_presses * claw_machine['Button A']['Y']
                    x += b_presses * claw_machine['Button B']['X']
                    y += b_presses * claw_machine['Button B']['Y']
                    if (x, y) == target_xy:
                        tokens_used = 3 * a_presses + b_presses
                        possible_tokens_used.add(tokens_used)
            tokens_needed.append(min(possible_tokens_used, default=0))
        result = sum(tokens_needed)
        return result
    
    def solve2(self, claw_machines, mod: int=10_000_000_000_000):
        '''
        Solve an example equation
          94 * A + 22 * B = 8400
          34 * A + 67 * B = 5400
        Solve for A in first equation
          94 * A + 22 * B = 8400
          94 * A = 8400 - 22 * B
          A = ((8400 - 22 * B) / 94)
        Solve for B in second equation
          34 * A + 67 * B = 5400
          67 * B = 5400 - 34 * A
          B = ((5400 - 34 * A) / 67)
        Substitute B
          94 * A + 22 * B = 8400
          94 * A + 22 * ((5400 - 34 * A) / 67) = 8400
          67 * 94 * A + 22 * (5400 - 34 * A) = 67 * 8400
          6298 * A + 118800 - 748 * A = 562800
          (6298 - 748) * A + 118800 = 562800
          5550 * A = 562800 - 118800
          A = 444000 / 5550
          A = 80
        Substitute A
          B = (5400 - 34 * 80) / 67
          B = (5400 - 2720) / 67
          B = 2680 / 67
          B = 40
        -----------------------
        Rewrite in general form
          ax * A + bx * B = px
          ay * A + by * B = py
        Solve for A in first equation
          ax * A + bx * B = px
          ax * A = px - bx * B
          A = (px - bx * B) / ax
        Solve for B in second equation
          ay * A + by * B = py
          by * B = py - ay * A
          B = (py - ay * A) / by
        Substitute B
          A = (px - bx * ((py - ay * A) / by)) / ax
          ax * A = px - bx * (py - ay * A) / by
          by * ax * A = by * px - bx * (py - ay * A)
          by * ax * A = by * px - bx * py + bx * ay * A
          by * ax * A - bx * ay * A = by * px - bx * py
          A * (by * ax - bx * ay) = by * px - bx * py
          A = (by * px - bx * py) / (by * ax - bx * ay)
        -----------------------
        Test general equation
          A = (by * px - bx * py) / (by * ax - bx * ay)
          ax = 94, ay = 34, bx = 22, by = 67, px = 8400, py = 5400
          A = (67 * 8400 - 22 * 5400) / (67 * 94 - 22 * 34)
          A = 80
          B = (py - ay * A) / by
          B = (5400 - 34 * 80) / 67
          B = 40
        '''
        tokens_needed = []
        for claw_machine in claw_machines:
            # A = (by * px - bx * py) / (by * ax - bx * ay)
            # B = (py - ay * A) / by
            ax = claw_machine['Button A']['X']
            ay = claw_machine['Button A']['Y']
            bx = claw_machine['Button B']['X']
            by = claw_machine['Button B']['Y']
            px = mod + claw_machine['Prize']['X']
            py = mod + claw_machine['Prize']['Y']
            A = (by * px - bx * py) / (by * ax - bx * ay)
            if A.is_integer():
                B = (py - ay * A) / by
                if B.is_integer():
                    tokens_needed.append(int(3 * A + B))
        result = sum(tokens_needed)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        claw_machines = self.get_claw_machines(raw_input_lines)
        solutions = (
            self.solve(claw_machines),
            self.solve2(claw_machines),
        )
        result = solutions
        return result

class Day12: # Garden Groups
    '''
    https://adventofcode.com/2024/day/12
    '''
    def get_parsed_input(self, raw_input_lines: list[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def get_regions(self, grid):
        # A region's key is equal to the (row, col) of its top-left plot
        rows = len(grid)
        cols = len(grid[0])
        regions = {}
        seen = set()
        for start_row in range(rows):
            for start_col in range(cols):
                if (start_row, start_col) in seen:
                    continue
                region = set()
                plant = grid[start_row][start_col]
                work = set()
                work.add((start_row, start_col))
                while len(work) > 0:
                    (row, col) = work.pop()
                    region.add((row, col))
                    if (row, col) in seen:
                        continue
                    seen.add((row, col))
                    for (n_row, n_col) in (
                        (row - 1, col),
                        (row    , col - 1),
                        (row + 1, col),
                        (row    , col + 1),
                    ):
                        if 0 <= n_row < rows and 0 <= n_col < cols:
                            if grid[n_row][n_col] == plant:
                                if (n_row, n_col) not in seen:
                                    work.add((n_row, n_col))
                regions[(start_row, start_col)] = region
        result = regions
        return result
    
    def solve(self, regions):
        fences = {}
        for ((start_row, start_col), region) in regions.items():
            area = len(region)
            perimeter = 0
            for (row, col) in region:
                for (n_row, n_col) in (
                    (row - 1, col),
                    (row    , col - 1),
                    (row + 1, col),
                    (row    , col + 1),
                ):
                    if (n_row, n_col) not in region:
                        perimeter += 1
            fences[(start_row, start_col)] = (area, perimeter)
        result = sum(
            (area * perimeter) for
            (area, perimeter) in fences.values()
        )
        return result
    
    def solve2(self, regions):
        fences = {}
        for ((start_row, start_col), region) in regions.items():
            area = len(region)
            # Calculate unique edges by giving each edge a canonical key
            # The canonical key of an edge is equal to:
            # - Its direction (e.g., 'TOP', 'BOTTOM', 'LEFT', 'RIGHT')
            # - The top-left-most plot that abuts that edge
            edges = set()
            for (row, col) in region:
                # Check for TOP edge
                if (row - 1, col) not in region:
                    outside = (row - 1, col)
                    inside = (row, col)
                    # Go LEFT until no longer on the same edge
                    while outside not in region and inside in region:
                        outside = (outside[0], outside[1] - 1)
                        inside = (inside[0], inside[1] - 1)
                    # Retreat one step to return to the same edge
                    inside = (inside[0], inside[1] + 1)
                    edges.add(('TOP', inside))
                # Check for BOTTOM edge
                if (row + 1, col) not in region:
                    outside = (row + 1, col)
                    inside = (row, col)
                    # Go LEFT until no longer on the same edge
                    while outside not in region and inside in region:
                        outside = (outside[0], outside[1] - 1)
                        inside = (inside[0], inside[1] - 1)
                    # Retreat one step to return to the same edge
                    inside = (inside[0], inside[1] + 1)
                    edges.add(('BOTTOM', inside))
                # Check for LEFT edge
                if (row, col - 1) not in region:
                    outside = (row, col - 1)
                    inside = (row, col)
                    # Go UP until no longer on the same edge
                    while outside not in region and inside in region:
                        outside = (outside[0] - 1, outside[1])
                        inside = (inside[0] - 1, inside[1])
                    # Retreat one step to return to the same edge
                    inside = (inside[0] + 1, inside[1])
                    edges.add(('LEFT', inside))
                # Check for RIGHT edge
                if (row, col + 1) not in region:
                    outside = (row, col + 1)
                    inside = (row, col)
                    # Go UP until no longer on the same edge
                    while outside not in region and inside in region:
                        outside = (outside[0] - 1, outside[1])
                        inside = (inside[0] - 1, inside[1])
                    # Retreat one step to return to the same edge
                    inside = (inside[0] + 1, inside[1])
                    edges.add(('RIGHT', inside))
            fences[(start_row, start_col)] = (area, len(edges))
        result = sum(
            (area * edge_count) for
            (area, edge_count) in fences.values()
        )
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        grid = self.get_parsed_input(raw_input_lines)
        regions = self.get_regions(grid)
        solutions = (
            self.solve(regions),
            self.solve2(regions),
        )
        result = solutions
        return result

class Day11: # Plutonian Pebbles
    '''
    https://adventofcode.com/2024/day/11
    '''
    def get_stones(self, raw_input_lines: list[str]):
        stones = tuple(map(int, raw_input_lines[0].split()))
        result = stones
        return result
    
    def blink(self, stone):
        next_stones = []
        if stone == 0:
            next_stones.append(1)
        elif len(str(stone)) % 2 == 0:
            stone_str = str(stone)
            N = len(stone_str) // 2
            (left, right) = tuple(map(int, (stone_str[:N], stone_str[N:])))
            next_stones.append(left)
            next_stones.append(right)
        else:
            next_stones.append(2024 * stone)
        result = next_stones
        return result
    
    def solve(self, stones):
        for _ in range(25):
            next_stones = []
            for stone in stones:
                next_stones += self.blink(stone)
            stones = next_stones
        result = len(stones)
        return result
    
    def solve2(self, stones):
        memo = {}
        def F(a, b) -> int:
            result = None
            if a < 1:
                result = 1
            elif a == 1:
                if len(str(b)) % 2 == 0:
                    result = 2
                else:
                    result = 1
            elif (a, b) in memo:
                # Pull from cache if possible
                result = memo[(a, b)]
            else:
                # Cache the result after calculating it recursively
                if b == 0:
                    result = F(a - 1, 1)
                else:
                    B = str(b)
                    if (len(B) % 2) == 0:
                        N = len(B) // 2
                        (b1, b2) = tuple(map(int, (B[:N], B[N:])))
                        result = F(a - 1, b1) + F(a - 1, b2)
                    else:
                        result = F(a - 1, 2024 * b)
                memo[(a, b)] = result
            return result
        result = 0
        for stone in stones:
            result += F(75, stone)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        stones = self.get_stones(raw_input_lines)
        solutions = (
            self.solve(list(stones)),
            self.solve2(list(stones)),
        )
        result = solutions
        return result

class Day10: # Hoof It
    '''
    https://adventofcode.com/2024/day/10
    '''
    def get_grid(self, raw_input_lines: list[str]):
        result = []
        for raw_input_line in raw_input_lines:
            line = (-1 if char == '.' else char for char in raw_input_line)
            result.append(list(map(int, line)))
        return result
    
    def solve(self, grid):
        trailheads = {}
        rows = len(grid)
        cols = len(grid[0])
        work = []
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == 0:
                    work.append(((row, col), (row, col)))
        while len(work) > 0:
            (head_row, head_col), (row, col) = work.pop()
            height = grid[row][col]
            if height == 9:
                if (head_row, head_col) not in trailheads:
                    trailheads[(head_row, head_col)] = set()
                trailheads[(head_row, head_col)].add((row, col))
                continue
            for (next_row, next_col) in (
                (row - 1, col + 0),
                (row + 0, col - 1),
                (row + 0, col + 1),
                (row + 1, col + 0),
            ):
                if not (0 <= next_row < rows and 0 <= next_col < cols):
                    continue
                next_height = grid[next_row][next_col]
                if next_height != (height + 1):
                    continue
                if next_height <= 9:
                    work.append(((head_row, head_col), (next_row, next_col)))
        # Sum of scores of all trailheads
        result = sum(len(tails) for tails in trailheads.values())
        return result
    
    def solve2(self, grid):
        trailheads = {}
        rows = len(grid)
        cols = len(grid[0])
        work = []
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == 0:
                    work.append(((row, col), (row, col)))
        while len(work) > 0:
            (head_row, head_col), (row, col) = work.pop()
            height = grid[row][col]
            if height == 9:
                if (head_row, head_col) not in trailheads:
                    trailheads[(head_row, head_col)] = 0
                trailheads[(head_row, head_col)] += 1
                continue
            for (next_row, next_col) in (
                (row - 1, col + 0),
                (row + 0, col - 1),
                (row + 0, col + 1),
                (row + 1, col + 0),
            ):
                if not (0 <= next_row < rows and 0 <= next_col < cols):
                    continue
                next_height = grid[next_row][next_col]
                if next_height != (height + 1):
                    continue
                if next_height <= 9:
                    work.append(((head_row, head_col), (next_row, next_col)))
        # Sum of scores of all trailheads
        result = sum(trailheads.values())
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

class Day09: # Disk Fragmenter
    '''
    https://adventofcode.com/2024/day/9
    '''
    def get_parsed_input(self, raw_input_lines: list[str]):
        result = list(map(int, raw_input_lines[0]))
        return result
    
    def solve(self, parsed_input):
        # Create disk
        disk = []
        mode = 'FILE'
        file_id = 0
        for num in parsed_input:
            if mode == 'FILE':
                disk += [file_id] * num
                file_id += 1
            else:
                disk += [None] * num
            mode = 'FREE' if mode == 'FILE' else 'FILE'
        # Defrag disk
        (left, right) = (0, len(disk) - 1)
        while left < right:
            if disk[left] is None:
                while disk[right] is None:
                    right -= 1
                if left < right:
                    (disk[left], disk[right]) = (disk[right], disk[left])
            left += 1
        # Calculate checksum
        checksum = 0
        for (position, file_id) in enumerate(disk):
            if file_id is not None:
                checksum += position * file_id
        result = checksum
        return result

    def disk2str(self, disk) -> str:
        result = ''.join('.' if char is None else str(char) for char in disk)
        return result
    
    def solve2(self, parsed_input):
        # Create disk
        files = {}
        disk = []
        mode = 'FILE'
        file_count = 0
        for num in parsed_input:
            if mode == 'FILE':
                files[file_count] = num
                disk += [file_count] * num
                file_count += 1
            else:
                disk += [None] * num
            mode = 'FREE' if mode == 'FILE' else 'FILE'
        # Defrag disk
        for target_file_id in reversed(range(file_count)):
            file_size = files[target_file_id]
            # Find right-most position of target file
            right = len(disk) - 1
            while disk[right] != target_file_id:
                right -= 1
            # Find left-most position of valid empty space
            left = 0
            while left < right:
                if disk[left] is None:
                    empty_space = 1
                    while disk[left + 1] == None and empty_space < file_size:
                        left += 1
                        empty_space += 1
                    if empty_space >= file_size:
                        for i in range(file_size):
                            (disk[left - i], disk[right - i]) = (disk[right - i], disk[left - i])
                        break
                left += 1
        # Calculate checksum
        checksum = 0
        for (position, file_id) in enumerate(disk):
            if file_id is not None:
                checksum += position * file_id
        result = checksum
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

class Day08: # Resonant Collinearity
    '''
    https://adventofcode.com/2024/day/8
    '''
    def get_parsed_input(self, raw_input_lines: list[str]):
        rows = len(raw_input_lines)
        cols = len(raw_input_lines[0])
        antennas = {}
        for (row, raw_input_line) in enumerate(raw_input_lines):
            for (col, char) in enumerate(raw_input_line):
                if char == '.':
                    continue
                if char not in antennas:
                    antennas[char] = set()
                antennas[char].add((row, col))
        result = (antennas, rows, cols)
        return result
    
    def solve(self, antennas, rows, cols):
        antinodes = set()
        for (frequency, positions) in antennas.items():
            for (row1, col1) in positions:
                for (row2, col2) in positions:
                    if (row2, col2) == (row1, col1):
                        continue
                    antinode1 = (row2 + (row2 - row1), col2 + (col2 - col1))
                    antinodes.add(antinode1)
                    antinode2 = (row1 + (row1 - row2), col1 + (col1 - col2))
                    antinodes.add(antinode2)
        result = sum((
            1 for (row, col) in antinodes if
            0 <= row < rows and
            0 <= col < cols
        ))
        return result
    
    def solve2(self, antennas, rows, cols):
        antinodes = set()
        for (frequency, positions) in antennas.items():
            for (row1, col1) in positions:
                for (row2, col2) in positions:
                    if (row2, col2) == (row1, col1):
                        continue
                    (row_diff, col_diff) = (row2 - row1, col2 - col1)
                    step = 1
                    while True:
                        (ar, ac) = (row1 + step * row_diff, col1 + step * col_diff)
                        if 0 <= ar < rows and 0 <= ac < cols:
                            antinodes.add((ar, ac))
                        else:
                            break
                        step += 1
        result = sum((
            1 for (row, col) in antinodes if
            0 <= row < rows and
            0 <= col < cols
        ))
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        (antennas, rows, cols) = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(antennas, rows, cols),
            self.solve2(antennas, rows, cols),
        )
        result = solutions
        return result

class Day07: # Bridge Repair
    '''
    https://adventofcode.com/2024/day/7
    '''
    def get_equations(self, raw_input_lines: list[str]):
        equations = []
        for raw_input_line in raw_input_lines:
            left, right = raw_input_line.split(': ')
            test_value = int(left)
            numbers = tuple(map(int, right.split()))
            assert len(numbers) > 1
            for num in numbers:
                assert num > 0
            equations.append((test_value, numbers))
        result = equations
        return result
    
    def solve(self, equations):
        valid_test_values = []
        for (test_value, numbers) in equations:
            valid_ind = False
            work = set()
            work.add((test_value, *numbers))
            while len(work) > 0:
                values = work.pop()
                assert len(values) >= 2
                if len(values) == 2:
                    if values[0] == values[1]:
                        valid_ind = True
                        break
                else:
                    (head, mid, tail) = values[0], values[1:-1], values[-1]
                    # Try addition
                    if (head - tail) >= 0:
                        addition = (head - tail, ) + mid
                        work.add((addition))
                    # Try multiplication
                    if (head / tail).is_integer():
                        multiplication = (head // tail, ) + mid
                        work.add((multiplication))
            if valid_ind:
                valid_test_values.append(test_value)
        result = sum(valid_test_values)
        return result
    
    def solve2(self, equations):
        valid_test_values = []
        for (test_value, numbers) in equations:
            valid_ind = False
            work = set()
            work.add((tuple(numbers)))
            while len(work) > 0:
                values = work.pop()
                assert len(values) >= 1
                if len(values) == 1:
                    if values[0] == test_value:
                        valid_ind = True
                        break
                else:
                    (head, mid, tail) = values[0], values[1], values[2:]
                    # Try addition
                    addition = (head + mid, ) + tail
                    work.add((addition))
                    # Try multiplication
                    multiplication = (head * mid, ) + tail
                    work.add((multiplication))
                    # Try concatenation
                    new_head = int(str(head) + str(mid))
                    concatenation = (new_head, ) + tail
                    work.add(concatenation)
            if valid_ind:
                valid_test_values.append(test_value)
        result = sum(valid_test_values)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        equations = self.get_equations(raw_input_lines)
        solutions = (
            self.solve(equations),
            self.solve2(equations),
        )
        result = solutions
        return result

class Day06: # Guard Gallivant
    '''
    https://adventofcode.com/2024/day/6
    '''
    directions = {
        'UP'   : (-1,  0, 'RIGHT'),
        'RIGHT': ( 0,  1, 'DOWN'),
        'DOWN' : ( 1,  0, 'LEFT'),
        'LEFT' : ( 0, -1, 'UP'),
    }
    def get_parsed_input(self, raw_input_lines: list[str]):
        rows = len(raw_input_lines)
        cols = len(raw_input_lines[0])
        obstacles = set()
        guard = (0, 0, 'UNKNOWN')
        result = []
        for (row, raw_input_line) in enumerate(raw_input_lines):
            for (col, char) in enumerate(raw_input_line):
                if char == '#':
                    obstacles.add((row, col))
                elif char in '>^v<':
                    directions = {
                        '>': 'RIGHT',
                        '^': 'UP',
                        'v': 'DOWN',
                        '<': 'LEFT',
                    }
                    guard = (row, col, directions[char])
            result.append(raw_input_line)
        result = (rows, cols, obstacles, guard)
        return result
    
    def solve(self, rows: int, cols: int, obstacles: set, guard: tuple):
        visits = set()
        while True:
            (row, col, direction) = guard
            if not (0 <= row < rows and 0 <= col < cols):
                break
            visits.add((row, col))
            while True:
                facing_row = row + self.directions[direction][0]
                facing_col = col + self.directions[direction][1]
                if (facing_row, facing_col) in obstacles:
                    # Turn 90 degrees
                    direction = self.directions[direction][2]
                else:
                    # Take a step forward
                    guard = (facing_row, facing_col, direction)
                    break
        result = len(visits)
        return result
    
    def solve2(self, rows: int, cols: int, original_obstacles: set, original_guard: tuple):
        positions = set()
        for obstacle_row in range(rows):
            for obstacle_col in range(cols):
                if (obstacle_row, obstacle_col) in original_obstacles:
                    continue
                if (obstacle_row, obstacle_col) == (original_guard[0], original_guard[1]):
                    continue
                obstacles = set(original_obstacles)
                obstacles.add((obstacle_row, obstacle_col))
                guard = original_guard
                pivots = set()
                while True:
                    (row, col, direction) = guard
                    if not (0 <= row < rows and 0 <= col < cols):
                        break
                    if (row, col, direction) in pivots:
                        positions.add((obstacle_row, obstacle_col))
                        break
                    while True:
                        facing_row = row + self.directions[direction][0]
                        facing_col = col + self.directions[direction][1]
                        if (facing_row, facing_col) in obstacles:
                            # Turn 90 degrees
                            pivots.add((row, col, direction))
                            direction = self.directions[direction][2]
                        else:
                            # Take a step forward
                            guard = (facing_row, facing_col, direction)
                            break
        result = len(positions)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        (rows, cols, obstacles, guard) = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(rows, cols, obstacles, guard),
            self.solve2(rows, cols, obstacles, guard),
        )
        result = solutions
        return result

class Day05: # Print Queue
    '''
    https://adventofcode.com/2024/day/5
    '''
    def get_parsed_input(self, raw_input_lines: list[str]):
        rules = set()
        updates = []
        mode = 'RULES'
        for raw_input_line in raw_input_lines:
            if len(raw_input_line) == 0:
                mode = 'UPDATES'
            else:
                if mode == 'RULES':
                    rules.add(tuple(map(int, raw_input_line.split('|'))))
                elif mode == 'UPDATES':
                    updates.append(list(map(int, raw_input_line.split(','))))
                else:
                    raise Exception('Unknown mode')
        result = (rules, updates)
        return result
    
    def collated(self, rules, update):
        def lt(a, b):
            result = 0
            if (a, b) in rules:
                result = -1
            elif (b, a) in rules:
                result = 1
            return result
        result = sorted(update, key=functools.cmp_to_key(lt))
        return result
    
    def solve(self, rules, updates):
        result = 0
        for update in updates:
            collated_update = self.collated(rules, update)
            if collated_update == update:
                n = len(collated_update) // 2
                result += collated_update[n]
        return result
    
    def solve2(self, rules, updates):
        result = 0
        for update in updates:
            collated_update = self.collated(rules, update)
            if collated_update != update:
                n = len(collated_update) // 2
                result += collated_update[n]
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        (rules, updates) = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(rules, updates),
            self.solve2(rules, updates),
        )
        result = solutions
        return result

class Day04: # Ceres Search
    '''
    https://adventofcode.com/2024/day/4
    '''
    def get_parsed_input(self, raw_input_lines: list[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, parsed_input):
        word = 'XMAS'
        word_count = 0
        for start_row in range(len(parsed_input)):
            for start_col in range(len(parsed_input[start_row])):
                if parsed_input[start_row][start_col] == word[0]:
                    for (row_mul, col_mul) in (
                        (-1, -1),
                        (-1,  0),
                        (-1,  1),
                        ( 0, -1),
                        ( 0,  1),
                        ( 1, -1),
                        ( 1,  0),
                        ( 1,  1),
                    ):
                        for i in range(len(word)):
                            row = start_row + row_mul * i
                            col = start_col + col_mul * i
                            if not(0 <= row < len(parsed_input)):
                                break
                            if not(0 <= col < len(parsed_input[row])):
                                break
                            if parsed_input[row][col] != word[i]:
                                break
                        else:
                            word_count += 1
        result = word_count
        return result
    
    def solve2(self, parsed_input):
        word_count = 0
        for start_row in range(len(parsed_input)):
            for start_col in range(len(parsed_input[start_row])):
                if parsed_input[start_row][start_col] == 'A':
                    corners = []
                    for (row_offset, col_offset) in (
                        (-1, -1),
                        (-1,  1),
                        ( 1, -1),
                        ( 1,  1),
                    ):
                        corners.append('')
                        row = start_row + row_offset
                        col = start_col + col_offset
                        if not(0 <= row < len(parsed_input)):
                            continue
                        if not(0 <= col < len(parsed_input[row])):
                            continue
                        corners[-1] = parsed_input[row][col]
                    if (
                        (
                            corners[0] + 'A' + corners[3] == 'MAS' or
                            corners[0] + 'A' + corners[3] == 'SAM'
                        ) and
                        (
                            corners[1] + 'A' + corners[2] == 'MAS' or
                            corners[1] + 'A' + corners[2] == 'SAM'
                        )
                    ):
                        word_count += 1
        result = word_count
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

class Day03: # Mull It Over
    '''
    https://adventofcode.com/2024/day/3
    '''
    def get_parsed_input(self, raw_input_lines: list[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, parsed_input):
        multiplications = []
        for line in parsed_input:
            for i in range(len(line)):
                if line[i:].startswith('mul('):
                    length = line[i:i + 12].find(')')
                    if length >= 0:
                        segment = line[i:i + length + 1]
                        try:
                            multiplication = tuple(map(int, segment[4:-1].split(',')))
                            multiplications.append(multiplication)
                        except ValueError:
                            pass
        result = sum((a * b for (a, b) in multiplications))
        return result
    
    def solve2(self, parsed_input):
        instructions = []
        for line in parsed_input:
            for i in range(len(line)):
                if line[i:].startswith('mul('):
                    length = line[i:i + 12].find(')')
                    if length >= 0:
                        segment = line[i:i + length + 1]
                        try:
                            (a, b) = tuple(map(int, segment[4:-1].split(',')))
                            instructions.append(('MUL', a, b))
                        except ValueError:
                            pass
                elif line[i:].startswith('do()'):
                    instructions.append(('DO', ))
                elif line[i:].startswith('don\'t()'):
                    instructions.append(('DONT', ))
        result = 0
        do = True
        for instruction in instructions:
            op = instruction[0]
            if op == 'MUL':
                (a, b) = instruction[1:]
                if do:
                    result += a * b
            elif op == 'DO':
                do = True
            elif op == 'DONT':
                do = False
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
    
    def check_report(self, report):
        safe_ind = True
        prev_diff = 0
        for i in range(1, len(report)):
            diff = report[i] - report[i - 1]
            if not(1 <= abs(diff) <= 3):
                safe_ind = False
                break
            if (
                (prev_diff < 0 and diff > 0) or
                (prev_diff > 0 and diff < 0)
            ):
                safe_ind = False
                break
            prev_diff = diff
        result = safe_ind
        return result
    
    def solve(self, reports):
        safe_count = 0
        for report in reports:
            if self.check_report(report):
                safe_count += 1
        result = safe_count
        return result
    
    def solve2(self, reports):
        safe_count = 0
        for report in reports:
            safe_ind = False
            if self.check_report(report):
                safe_ind = True
            else:
                for i in range(len(report)):
                    modified_report = tuple(report[:i] + report[i + 1:])
                    if self.check_report(modified_report):
                        safe_ind = True
                        break
            if safe_ind:
                safe_count += 1
        result = safe_count
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
    python AdventOfCode2024.py 14 < inputs/2024day14.in
    '''
    solvers = {
        1: (Day01, 'Historian Hysteria'),
        2: (Day02, 'Red-Nosed Reports'),
        3: (Day03, 'Mull It Over'),
        4: (Day04, 'Ceres Search'),
        5: (Day05, 'Print Queue'),
        6: (Day06, 'Guard Gallivant'),
        7: (Day07, 'Bridge Repair'),
        8: (Day08, 'Resonant Collinearity'),
        9: (Day09, 'Disk Fragmenter'),
       10: (Day10, 'Hoof It'),
       11: (Day11, 'Plutonian Pebbles'),
       12: (Day12, 'Garden Groups'),
       13: (Day13, 'Claw Contraption'),
       14: (Day14, 'Restroom Redoubt'),
       15: (Day15, 'Warehouse Woes'),
       16: (Day16, 'Reindeer Maze'),
       17: (Day17, 'Chronospatial Computer'),
       18: (Day18, 'RAM Run'),
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
