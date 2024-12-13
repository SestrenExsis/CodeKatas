'''
Created on 2024-11-30

@author: Sestren
'''
import argparse
import functools

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
                    claw_machine[mode] = {
                        'X': x,
                        'Y': y,
                    }
                    claw_machines.append(claw_machine)
                    claw_machine = {}
                else:
                    x = int(raw_x[1:])
                    y = int(raw_y[1:])
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
    
    def solve2(self, claw_machines):
        result = len(claw_machines)
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
    python AdventOfCode2024.py 12 < inputs/2024day12.in
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
