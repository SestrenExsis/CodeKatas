'''
Created on 2022-11-30

@author: Sestren
'''
import argparse
import collections
import copy
import functools
import heapq
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

class DeviceCPU:
    def __init__(self, instructions):
        self.x = 1
        self.instructions = instructions
        self.pc = 0
        self.cycles_remaining = 0
    
    def step(self):
        instruction = self.instructions[self.pc]
        if instruction[0] == 'noop':
            # NOOP takes 1 clock cycle to compute
            self.pc += 1
        elif instruction[0] == 'addx':
            # ADDX takes 2 clock cycles to compute
            if self.cycles_remaining == 0:
                self.cycles_remaining = 1
            else:
                self.cycles_remaining -= 1
                if self.cycles_remaining < 1:
                    self.x += instruction[1]
                    self.pc += 1

class DeviceCRT:
    def __init__(self):
        self.rows = 6
        self.cols = 40
        self.display = [['.'] * self.cols for _ in range(self.rows)]
        self.pos = (0, 0)
    
    def step(self, x):
        row, col = self.pos
        cell = '.'
        if col in (x - 1, x, x + 1):
            cell = '#'
        self.display[row][col] = cell
        col += 1
        if col >= self.cols:
            col = 0
            row += 1
            if row >= self.rows:
                row = 0
        self.pos = (row, col)
    
    def get_display(self):
        display = []
        for row in range(self.rows):
            row_data = ''.join(self.display[row])
            display.append(row_data)
        result = display
        return result

class Template: # Template
    '''
    https://adventofcode.com/2022/day/?
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

class Day16: # Proboscidea Volcanium
    '''
    https://adventofcode.com/2022/day/16
    '''
    def get_valves(self, raw_input_lines: List[str]):
        valves = {}
        for raw_input_line in raw_input_lines:
            a, b = raw_input_line.split('; ')
            c, d = a.split('=')
            parts = c.split(' ')
            valve_id = parts[1]
            flow_rate = int(d)
            if b[:7] == 'tunnels':
                b = b[23:]
            else:
                b = b[22:]
            tunnels = set(b.split(', '))
            bit_mask = 2 ** len(valves)
            valve = {
                'flow_rate': flow_rate,
                'tunnels': tunnels,
                'bit_mask': bit_mask,
            }
            valves[valve_id] = valve
        result = valves
        return result
    
    def solve(self, valves, max_time: int=30):
        # Many of the valves have flow rate of 0, so maybe ignore them
        # Simplify the cave structure first
        # time is defined as minutes elapsed
        # score is total pressure released by the final time
        # valve_id is the current valve
        # valves_open is a 64-bit mask
        max_score = 0
        work = [(0, 0, 'AA', 0)]
        # (time, score, valve_id, valves_open)
        seen = {}
        # (valve_id, valves_open): (score, time_left)
        while len(work) > 0:
            (time, score, valve_id, valves_open) = work.pop()
            if score > max_score:
                max_score = score
            if time >= max_time:
                continue
            valve = valves[valve_id]
            flow_rate = valve['flow_rate']
            tunnels = valve['tunnels']
            bit_mask = valve['bit_mask']
            for new_valve_id in tunnels:
                key = (new_valve_id, valves_open)
                if (
                    key not in seen or
                    seen[key] < (score, max_time - (time + 1))
                ):
                    seen[key] = (score, max_time - (time + 1))
                    work.append((time + 1, score, new_valve_id, valves_open))
            if (
                valves_open & bit_mask == 0 and
                flow_rate > 0
            ):
                new_score = score + flow_rate * (max_time - (time + 1))
                new_valves_open = valves_open | bit_mask
                key = (valve_id, new_valves_open)
                if (
                    key not in seen or
                    seen[key] < (new_score, max_time - (time + 1))
                ):
                    seen[key] = (new_score, max_time - (time + 1))
                    work.append((time + 1, new_score, valve_id, new_valves_open))
        result = max_score
        return result
    
    def solve2(self, valves):
        result = len(valves)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        valves = self.get_valves(raw_input_lines)
        solutions = (
            self.solve(valves),
            self.solve2(valves),
            )
        result = solutions
        return result

class Day15: # Beacon Exclusion Zone
    '''
    https://adventofcode.com/2022/day/15
    '''
    def get_sensors(self, raw_input_lines: List[str]):
        sensors = {}
        for raw_input_line in raw_input_lines:
            parts = raw_input_line.split(': ')
            part0 = parts[0].split(' ')
            sensor_x = int(part0[2][2:-1])
            sensor_y = int(part0[3][2:])
            part1 = parts[1].split(' ')
            beacon_x = int(part1[4][2:-1])
            beacon_y = int(part1[5][2:])
            sensors[(sensor_x, sensor_y)] = (beacon_x, beacon_y)
        result = sensors
        return result
    
    def solve(self, sensors, target_y: int=2_000_000):
        positions = set()
        for (sx, sy), (bx, by) in sensors.items():
            distance = abs(bx - sx) + abs(by - sy)
            span = distance - abs(sy - target_y)
            for offset in range(span + 1):
                positions.add((sx + offset, target_y))
                positions.add((sx - offset, target_y))
            positions.discard((bx, by))
        result = len(positions)
        return result
    
    def solve2_slowly(self, sensors, max_coord: int=4_000_000):
        candidates = set()
        for x in range(max_coord + 1):
            print('x:', x)
            for y in range(max_coord + 1):
                candidate = True
                for (sx, sy), (bx, by) in sensors.items():
                    nearest_beacon_distance = abs(bx - sx) + abs(by - sy)
                    distance = abs(sx - x) + abs(sy - y)
                    if distance <= nearest_beacon_distance:
                        candidate = False
                        break
                if candidate:
                    candidates.add((x, y))
                    break
            if len(candidates) > 0:
                break
        coord = candidates.pop()
        tuning_frequency = 4_000_000 * coord[0] + coord[1]
        result = tuning_frequency
        return result
    
    def solve2(self, sensors, max_coord: int=4_000_000):
        coord = None
        # Each sensor forms a perimeter of candidate coordinates
        sensors_list = []
        for (sx1, sy1), (bx1, by1) in sensors.items():
            distance = 1 + abs(bx1 - sx1) + abs(by1 - sy1)
            sensors_list.append((sx1, sy1, distance))
        for i in range(len(sensors_list)):
            candidates = set()
            (sx1, sy1, distance1) = sensors_list[i]
            for j in range(i + 1, len(sensors_list)):
                (sx2, sy2, distance2) = sensors_list[j]
                for y in range(max_coord + 1):
                    span = distance1 - abs(sy1 - y)
                    if span < 0:
                        continue
                    for x in (sx1 - span, sx1 + span):
                        d = abs(x - sx2) + abs(y - sy2)
                        if 0 <= x <= max_coord and d == distance2:
                            candidates.add((x, y))
            for (x, y) in candidates:
                candidate_ind = True
                for (sx, sy), (bx, by) in sensors.items():
                    nearest_beacon_distance = abs(bx - sx) + abs(by - sy)
                    distance = abs(sx - x) + abs(sy - y)
                    if distance <= nearest_beacon_distance:
                        candidate_ind = False
                        break
                if candidate_ind:
                    coord = (x, y)
                    break
            if coord is not None:
                break
        tuning_frequency = 4_000_000 * coord[0] + coord[1]
        result = tuning_frequency
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        sensors = self.get_sensors(raw_input_lines)
        solutions = (
            self.solve(sensors),
            self.solve2(sensors),
            )
        result = solutions
        return result

class Day14: # Regolith Reservoir
    '''
    https://adventofcode.com/2022/day/14
    '''
    def get_material(self, raw_input_lines: List[str]):
        material = {}
        for raw_input_line in raw_input_lines:
            raw_paths = raw_input_line.split(' -> ')
            prev_path = tuple(map(int, raw_paths[0].split(',')))
            for raw_path in raw_paths[1:]:
                curr_path = tuple(map(int, raw_path.split(',')))
                diff = curr_path[0] - prev_path[0]
                if diff == 0:
                    x = curr_path[0]
                    yy = (prev_path[1], curr_path[1])
                    for y in range(min(yy), max(yy) + 1):
                        material[(x, y)] = '#'
                else:
                    y = curr_path[1]
                    xx = (prev_path[0], curr_path[0])
                    for x in range(min(xx), max(xx) + 1):
                        material[(x, y)] = '#'
                prev_path = curr_path
        result = material
        return result
    
    def solve(self, material):
        max_y = max(y for (_, y) in material.keys())
        while True:
            (x, y) = (500, 0)
            while True:
                if (x, y + 1) not in material.keys():
                    y += 1
                elif (x - 1, y + 1) not in material.keys():
                    x -= 1
                    y += 1
                elif (x + 1, y + 1) not in material.keys():
                    x += 1
                    y += 1
                else:
                    material[(x, y)] = 'o'
                    break
                if y > max_y:
                    break
            if y > max_y:
                break
        result = sum(1 for v in material.values() if v == 'o')
        return result
    
    def solve2(self, material):
        floor = 2 + max(y for (_, y) in material.keys())
        while True:
            (x, y) = (500, 0)
            while True:
                if (x, y + 1) not in material.keys() and y + 1 < floor:
                    y += 1
                elif (x - 1, y + 1) not in material.keys() and y + 1 < floor:
                    x -= 1
                    y += 1
                elif (x + 1, y + 1) not in material.keys() and y + 1 < floor:
                    x += 1
                    y += 1
                else:
                    material[(x, y)] = 'o'
                    break
            if (x, y) == (500, 0):
                break
        result = sum(1 for v in material.values() if v == 'o')
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        material = self.get_material(raw_input_lines)
        solutions = (
            self.solve(copy.deepcopy(material)),
            self.solve2(copy.deepcopy(material)),
            )
        result = solutions
        return result

class Day13: # Distress Signal
    '''
    https://adventofcode.com/2022/day/13
    '''
    def get_packet_pairs(self, raw_input_lines: List[str]):
        packet_pairs = {}
        index = 0
        while index < len(raw_input_lines):
            left = eval(raw_input_lines[index])
            right = eval(raw_input_lines[index + 1])
            packet_pair = (left, right)
            packet_pairs[len(packet_pairs) + 1] = packet_pair
            index += 3
        result = packet_pairs
        return result
    
    def compare(self, left, right):
        result = 0
        if type(left) == int and type(right) == int:
            if left < right:
                result = -1
            elif left > right:
                result = 1
        elif type(left) == list and type(right) == list:
            N = min(len(left), len(right))
            comparison = 0
            for i in range(N):
                comparison = self.compare(left[i], right[i])
                if comparison < 0:
                    result = -1
                    break
                elif comparison > 0:
                    result = 1
                    break
            if comparison == 0:
                if len(left) < len(right):
                    result = -1
                elif len(left) > len(right):
                    result = 1
        else: # Assume one is a list and the other is an int
            if type(left) == int:
                result = self.compare([left], right)
            elif type(right) == int:
                result = self.compare(left, [right])
        return result
    
    def solve(self, packet_pairs):
        ordered_packets = set()
        for (pair_id, (left, right)) in packet_pairs.items():
            comparison = self.compare(left, right)
            if comparison < 0:
                ordered_packets.add(pair_id)
        result = sum(ordered_packets)
        return result
    
    def solve2(self, packets):
        packets.sort(key=functools.cmp_to_key(self.compare))
        divider_ids = []
        for packet_id, packet in enumerate(packets, start=1):
            if packet in ([[2]], [[6]]):
                divider_ids.append(packet_id)
        result = divider_ids[0] * divider_ids[1]
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        packet_pairs = self.get_packet_pairs(raw_input_lines)
        packets = []
        for (left, right) in packet_pairs.values():
            packets.append(left)
            packets.append(right)
        packets.append([[2]])
        packets.append([[6]])
        solutions = (
            self.solve(copy.deepcopy(packet_pairs)),
            self.solve2(packets),
            )
        result = solutions
        return result

class Day12: # Hill Climbing Algorithm
    '''
    https://adventofcode.com/2022/day/12
    '''
    def get_map_data(self, raw_input_lines: List[str]):
        heights = {}
        start = (0, 0)
        end = (0, 0)
        for row, raw_input_line in enumerate(raw_input_lines):
            for col in range(len(raw_input_line)):
                if raw_input_line[col] == 'S':
                    start = (row, col)
                    height = ord('a') - ord('a')
                elif raw_input_line[col] == 'E':
                    end = (row, col)
                    height = ord('z') - ord('a')
                else:
                    height = ord(raw_input_line[col]) - ord('a')
                heights[(row, col)] = height
        result = (heights, start, end)
        return result
    
    def solve(self, heights, start, end):
        min_distance = float('inf')
        visited = set()
        work = collections.deque()
        work.append((start[0], start[1], 0))
        while len(work) > 0:
            N = len(work)
            for _ in range(N):
                (row, col, distance) = work.pop()
                if (row, col) == end:
                    min_distance = distance
                    break
                if (row, col) in visited:
                    continue
                visited.add((row, col))
                for (next_row, next_col) in (
                    (row + 1, col    ),
                    (row - 1, col    ),
                    (row    , col + 1),
                    (row    , col - 1),
                ):
                    if (next_row, next_col) in heights:
                        height = heights[(row, col)]
                        next_height = heights[(next_row, next_col)]
                        if next_height <= height + 1:
                            work.appendleft((next_row, next_col, distance + 1))
            if min_distance < float('inf'):
                break
        result = min_distance
        return result
    
    def solve2(self, heights, start, end):
        min_distance = float('inf')
        visited = set()
        work = collections.deque()
        work.append((end[0], end[1], 0))
        while len(work) > 0:
            N = len(work)
            for _ in range(N):
                (row, col, distance) = work.pop()
                if heights[(row, col)] == 0:
                    min_distance = distance
                    break
                if (row, col) in visited:
                    continue
                visited.add((row, col))
                for (next_row, next_col) in (
                    (row + 1, col    ),
                    (row - 1, col    ),
                    (row    , col + 1),
                    (row    , col - 1),
                ):
                    if (next_row, next_col) in heights:
                        height = heights[(row, col)]
                        next_height = heights[(next_row, next_col)]
                        if next_height >= height - 1:
                            work.appendleft((next_row, next_col, distance + 1))
            if min_distance < float('inf'):
                break
        result = min_distance
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        (heights, start, end) = self.get_map_data(raw_input_lines)
        solutions = (
            self.solve(heights, start, end),
            self.solve2(heights, start, end),
            )
        result = solutions
        return result

class Day11: # Monkey in the Middle
    '''
    https://adventofcode.com/2022/day/11
    '''
    def get_monkeys(self, raw_input_lines: List[str]):
        monkeys = {}
        index = 0
        while index < len(raw_input_lines):
            monkey_id = raw_input_lines[index][-2]
            items = list(reversed(list(map(int,
                raw_input_lines[index + 1].split(': ')[1].split(', ')
            ))))
            operation = raw_input_lines[index + 2].split(' = ')[1]
            factor = int(raw_input_lines[index + 3].split(' ')[-1])
            true_monkey = raw_input_lines[index + 4].split(' ')[-1]
            false_monkey = raw_input_lines[index + 5].split(' ')[-1]
            monkey = {
                'items': collections.deque(items),
                'operation': operation,
                'factor': factor,
                'if_true': true_monkey,
                'if_false': false_monkey,
                'inspections': 0,
            }
            monkeys[monkey_id] = monkey
            index += 7
        result = monkeys
        return result
    
    def solve(self, monkeys):
        for round_id in range(20):
            for monkey_id, monkey in monkeys.items():
                while len(monkey['items']) > 0:
                    old = monkey['items'].pop()
                    expression = monkey['operation'].replace('old', str(old))
                    new = eval(expression) // 3
                    monkey['inspections'] += 1
                    target = monkey['if_false']
                    if new % monkey['factor'] == 0:
                        target = monkey['if_true']
                    monkeys[target]['items'].appendleft(new)
        monkey_business = []
        for monkey in monkeys.values():
            heapq.heappush(monkey_business, -1 * monkey['inspections'])
        result = -1 * heapq.heappop(monkey_business)
        result *= -1 * heapq.heappop(monkey_business)
        return result
    
    def solve2(self, monkeys):
        modulo = 1
        for monkey in monkeys.values():
            modulo *= monkey['factor']
        for round_id in range(10_000):
            for monkey_id, monkey in monkeys.items():
                while len(monkey['items']) > 0:
                    old = monkey['items'].pop()
                    expression = monkey['operation'].replace('old', str(old))
                    new = eval(expression) % modulo
                    monkey['inspections'] += 1
                    target = monkey['if_false']
                    if new % monkey['factor'] == 0:
                        target = monkey['if_true']
                    monkeys[target]['items'].appendleft(new)
        monkey_business = []
        for monkey in monkeys.values():
            heapq.heappush(monkey_business, -1 * monkey['inspections'])
        result = -1 * heapq.heappop(monkey_business)
        result *= -1 * heapq.heappop(monkey_business)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        monkeys = self.get_monkeys(raw_input_lines)
        solutions = (
            self.solve(copy.deepcopy(monkeys)),
            self.solve2(copy.deepcopy(monkeys)),
            )
        result = solutions
        return result

class Day10: # Cathode-Ray Tube
    '''
    https://adventofcode.com/2022/day/10
    '''
    def get_instructions(self, raw_input_lines: List[str]):
        instructions = []
        for raw_input_line in raw_input_lines:
            parts = raw_input_line.split()
            instruction = tuple(parts)
            if parts[0] == 'addx':
                instruction = (parts[0], int(parts[1]))
            instructions.append(instruction)
        result = instructions
        return result
    
    def solve(self, instructions):
        cpu = DeviceCPU(instructions)
        cycles = {}
        cycle_id = 1
        while cpu.pc < len(cpu.instructions):
            signal_strength = cpu.x * cycle_id
            cycles[cycle_id] = signal_strength
            cpu.step()
            cycle_id += 1
        result = sum(
            signal_strength for
            (cycle, signal_strength) in cycles.items() if
            cycle in (20, 60, 100, 140, 180, 220)
        )
        return result
    
    def solve2(self, instructions):
        cpu = DeviceCPU(instructions)
        crt = DeviceCRT()
        while cpu.pc < len(cpu.instructions):
            crt.step(cpu.x)
            cpu.step()
        result = '\n' + '\n'.join(crt.get_display())
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

class Day09: # Rope Bridge
    '''
    https://adventofcode.com/2022/day/9
    '''
    dirs = {
        'U': (-1,  0),
        'D': ( 1,  0),
        'L': ( 0, -1),
        'R': ( 0,  1),
    }

    def get_steps(self, raw_input_lines: List[str]):
        steps = []
        for raw_input_line in raw_input_lines:
            dir, count = raw_input_line.split(' ')
            steps.append((dir, int(count)))
        result = steps
        return result
    
    def solve(self, steps):
        tail_visits = set()
        tail = (0, 0)
        head = (0, 0)
        tail_visits.add(tail)
        for (dir, count) in steps:
            (drow, dcol) = self.dirs[dir]
            for _ in range(count):
                head = (head[0] + drow, head[1] + dcol)
                rowdiff = abs(head[0] - tail[0])
                coldiff = abs(head[1] - tail[1])
                if rowdiff > 1 or coldiff > 1 or sum((rowdiff, coldiff)) > 2:
                    if head[0] < tail[0]:
                        tail = (tail[0] - 1, tail[1])
                    elif head[0] > tail[0]:
                        tail = (tail[0] + 1, tail[1])
                    if head[1] < tail[1]:
                        tail = (tail[0], tail[1] - 1)
                    elif head[1] > tail[1]:
                        tail = (tail[0], tail[1] + 1)
                tail_visits.add(tail)
        result = len(tail_visits)
        return result
    
    def solve2(self, steps, knot_count: int=10):
        knots = []
        for _ in range(knot_count):
            knots.append((0, 0))
        head_visits = set()
        tail_visits = set()
        tail_visits.add(knots[-1])
        head_visits.add(knots[0])
        for (dir, count) in steps:
            (drow, dcol) = self.dirs[dir]
            for _ in range(count):
                knots[0] = (knots[0][0] + drow, knots[0][1] + dcol)
                head_visits.add(knots[0])
                for knot_id in range(1, knot_count):
                    rowdiff = abs(knots[knot_id - 1][0] - knots[knot_id][0])
                    coldiff = abs(knots[knot_id - 1][1] - knots[knot_id][1])
                    if rowdiff > 1 or coldiff > 1 or sum((rowdiff, coldiff)) > 2:
                        if knots[knot_id - 1][0] < knots[knot_id][0]:
                            knots[knot_id] = (knots[knot_id][0] - 1, knots[knot_id][1])
                        elif knots[knot_id - 1][0] > knots[knot_id][0]:
                            knots[knot_id] = (knots[knot_id][0] + 1, knots[knot_id][1])
                        if knots[knot_id - 1][1] < knots[knot_id][1]:
                            knots[knot_id] = (knots[knot_id][0], knots[knot_id][1] - 1)
                        elif knots[knot_id - 1][1] > knots[knot_id][1]:
                            knots[knot_id] = (knots[knot_id][0], knots[knot_id][1] + 1)
                tail_visits.add(knots[-1])
        result = len(tail_visits)
        # self.visualize(tail_visits)
        return result
    
    def visualize(self, visits: set):
        min_row = min(visit[0] for visit in visits)
        min_col = min(visit[1] for visit in visits)
        max_row = max(visit[0] for visit in visits)
        max_col = max(visit[1] for visit in visits)
        for row in range(min_row, max_row + 1):
            row_data = ''
            for col in range(min_col, max_col + 1):
                cell = '.'
                if (row, col) in visits:
                    cell = '#'
                row_data += cell
            print(row_data)
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        steps = self.get_steps(raw_input_lines)
        solutions = (
            self.solve(steps),
            self.solve2(steps),
            )
        result = solutions
        return result

class Day08: # Treetop Tree House
    '''
    https://adventofcode.com/2022/day/8
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, trees):
        rows = len(trees)
        cols = len(trees[0])
        visible_trees = set()
        for row in range(rows):
            # Look from the left
            tallest_tree_from_left = -1
            for col in range(cols):
                tree = int(trees[row][col])
                if tree > tallest_tree_from_left:
                    visible_trees.add((row, col))
                    tallest_tree_from_left = tree
            # Look from the right
            tallest_tree_from_right = -1
            for col in reversed(range(cols)):
                tree = int(trees[row][col])
                if tree > tallest_tree_from_right:
                    visible_trees.add((row, col))
                    tallest_tree_from_right = tree
        for col in range(cols):
            # Look from the top
            tallest_tree_from_top = -1
            for row in range(rows):
                tree = int(trees[row][col])
                if tree > tallest_tree_from_top:
                    visible_trees.add((row, col))
                    tallest_tree_from_top = tree
            # Look from the bottom
            tallest_tree_from_bottom = -1
            for row in reversed(range(rows)):
                tree = int(trees[row][col])
                if tree > tallest_tree_from_bottom:
                    visible_trees.add((row, col))
                    tallest_tree_from_bottom = tree
        result = len(visible_trees)
        return result
    
    def solve2(self, trees):
        rows = len(trees)
        cols = len(trees[0])
        scenic_scores = {}
        for row in range(rows):
            for col in range(cols):
                viewing_height = int(trees[row][col])
                # Look left
                left_tree_count = 0
                for c in reversed(range(col)):
                    left_tree_count += 1
                    tree = int(trees[row][c])
                    if tree >= viewing_height:
                        break
                # Look right
                right_tree_count = 0
                for c in range(col + 1, cols):
                    right_tree_count += 1
                    tree = int(trees[row][c])
                    if tree >= viewing_height:
                        break
                # Look up
                up_tree_count = 0
                for r in reversed(range(row)):
                    up_tree_count += 1
                    tree = int(trees[r][col])
                    if tree >= viewing_height:
                        break
                # Look down
                down_tree_count = 0
                for r in range(row + 1, rows):
                    down_tree_count += 1
                    tree = int(trees[r][col])
                    if tree >= viewing_height:
                        break
                scenic_score = left_tree_count * right_tree_count * up_tree_count * down_tree_count
                scenic_scores[(row, col)] = scenic_score
        result = max(scenic_scores.values())
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

class Day07: # No Space Left On Device
    '''
    https://adventofcode.com/2022/day/7
    '''
    def get_sizes(self, raw_input_lines: List[str]):
        working_dir = '/'
        lists = {}
        sizes = {}
        cursor = 0
        while cursor < len(raw_input_lines):
            if raw_input_lines[cursor][0] != '$':
                raise Exception('Invalid row. Row must start with $')
            command = tuple(raw_input_lines[cursor][2:].split(' '))
            if command[0] == 'cd':
                if command[1] == '/':
                    working_dir = '/'
                elif command[1] == '..':
                    if working_dir != '/':
                        working_dir = '/'.join(working_dir.split('/')[:-2]) + '/'
                else:
                    working_dir += command[1] + '/'
                cursor += 1
            elif command[0] == 'ls':
                cursor += 1
                dirs = []
                files = {}
                while (
                    cursor < len(raw_input_lines) and
                    raw_input_lines[cursor][0] != '$'
                ):
                    parts = raw_input_lines[cursor].split(' ')
                    if parts[0] == 'dir':
                        dirs.append(parts[1])
                    else:
                        file = parts[1]
                        filesize = int(parts[0])
                        files[file] = int(parts[0])
                        sizes[working_dir + file] = filesize
                    cursor += 1
                lists[working_dir] = (dirs, files)
        def sizeof(path: str):
            if path not in sizes:
                total_size = 0
                (dirs, files) = lists[path]
                for file in files:
                    total_size += sizes[path + file]
                for dir in dirs:
                    total_size += sizeof(path + dir + '/')
                sizes[path] = total_size
            return sizes[path]
        for path in lists:
            sizes[path] = sizeof(path)
        result = sizes
        return result
    
    def solve(self, sizes):
        result = sum(
            size for
            path, size in sizes.items() if
            path[-1] == '/' and size <= 100_000
        )
        return result
    
    def solve2(self, sizes):
        total_disk_space = 70_000_000
        available_disk_space = total_disk_space - sizes['/']
        disk_space_needed = max(0, 30_000_000 - available_disk_space)
        result = min(
            size for
            _, size in sizes.items() if
            size >= disk_space_needed
        )
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        sizes = self.get_sizes(raw_input_lines)
        solutions = (
            self.solve(sizes),
            self.solve2(sizes),
            )
        result = solutions
        return result

class Day06: # Tuning Trouble
    '''
    https://adventofcode.com/2022/day/6
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = raw_input_lines[0]
        return result
    
    def solve(self, signal):
        stream = iter(signal)
        result = ''
        chars = collections.deque()
        while len(chars) < 4:
            chars.append(next(stream))
        start = 4
        while len(set(chars)) < 4:
            chars.popleft()
            chars.append(next(stream))
            start += 1
        result = start
        return result
    
    def solve2(self, signal):
        stream = iter(signal)
        result = ''
        chars = collections.deque()
        while len(chars) < 14:
            chars.append(next(stream))
        start = 14
        while len(set(chars)) < 14:
            chars.popleft()
            chars.append(next(stream))
            start += 1
        result = start
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        signal = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(signal),
            self.solve2(signal),
            )
        result = solutions
        return result

class Day05: # Supply Stacks
    '''
    https://adventofcode.com/2022/day/5
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        stack_height = 0
        while '[' in raw_input_lines[stack_height]:
            stack_height += 1
        stack_count = max(map(int, raw_input_lines[stack_height].split()))
        stacks = []
        for stack_id in range(stack_count):
            col = 1 + 4 * stack_id
            stacks.append('')
            for row in reversed(range(stack_height)):
                cell = raw_input_lines[row][col]
                if cell != ' ':
                    stacks[-1] += raw_input_lines[row][col]
        instructions = []
        for raw_input_line in raw_input_lines[stack_height + 2:]:
            parts = raw_input_line.split(' ')
            crate_count = int(parts[1])
            source_column = int(parts[3]) - 1
            target_column = int(parts[5]) - 1
            instruction = (crate_count, source_column, target_column)
            instructions.append(instruction)
        result = (stacks, instructions)
        return result
    
    def solve(self, stacks, instructions):
        for (crate_count, source_column, target_column) in instructions:
            for _ in range(crate_count):
                crate = stacks[source_column][-1]
                stacks[source_column] = stacks[source_column][:-1]
                stacks[target_column] += crate
        result = ''
        for stack in stacks:
            result += stack[-1]
        return result
    
    def solve2(self, stacks, instructions):
        for (crate_count, source_column, target_column) in instructions:
            crates = stacks[source_column][-crate_count:]
            stacks[source_column] = stacks[source_column][:-crate_count]
            stacks[target_column] += crates
        result = ''
        for stack in stacks:
            result += stack[-1]
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        stacks, instructions = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(stacks[:], instructions),
            self.solve2(stacks[:], instructions),
            )
        result = solutions
        return result

class Day04: # Camp Cleanup
    '''
    https://adventofcode.com/2022/day/4
    '''
    def get_assignments(self, raw_input_lines: List[str]):
        assignments = []
        for raw_input_line in raw_input_lines:
            a, b = raw_input_line.split(',')
            a1, a2 = tuple(map(int, a.split('-')))
            b1, b2 = tuple(map(int, b.split('-')))
            assignments.append(((a1, a2), (b1, b2)))
        result = assignments
        return result
    
    def solve(self, assignments):
        containment_count = 0
        for (a1, a2), (b1, b2) in assignments:
            if (
                (a1 <= b1 and a2 >= b2) or
                (b1 <= a1 and b2 >= a2)
            ):
                containment_count += 1
        result = containment_count
        return result
    
    def solve2(self, assignments):
        overlap_count = 0
        for (a1, a2), (b1, b2) in assignments:
            if len(set(range(a1, a2 + 1)) & set(range(b1, b2 + 1))) > 0:
                overlap_count += 1
        result = overlap_count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        assignments = self.get_assignments(raw_input_lines)
        solutions = (
            self.solve(assignments),
            self.solve2(assignments),
            )
        result = solutions
        return result

class Day03: # Rucksack Reorganization
    '''
    https://adventofcode.com/2022/day/3
    '''
    def get_rucksacks(self, raw_input_lines: List[str]):
        rucksacks = []
        for raw_input_line in raw_input_lines:
            compartment_size = len(raw_input_line) // 2
            compartment_a = raw_input_line[:compartment_size]
            compartment_b = raw_input_line[compartment_size:]
            rucksacks.append((compartment_a, compartment_b))
        result = rucksacks
        return result
    
    def solve(self, rucksacks):
        priorities = []
        for (compartment_a, compartment_b) in rucksacks:
            common_items = set(compartment_a) & set(compartment_b)
            for item in common_items:
                priority = 0
                if item in 'abcdefghijklmnopqrstuvwxyz':
                    priority = 1 + ord(item) - ord('a')
                elif item in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    priority = 27 + ord(item) - ord('A')
                priorities.append(priority)
        result = sum(priorities)
        return result
    
    def solve2(self, rucksacks):
        priorities = []
        for index in range(0, len(rucksacks), 3):
            rucksack_a = rucksacks[index][0] + rucksacks[index][1]
            rucksack_b = rucksacks[index + 1][0] + rucksacks[index + 1][1]
            rucksack_c = rucksacks[index + 2][0] + rucksacks[index + 2][1]
            common_items = set(rucksack_a) & set(rucksack_b) & set(rucksack_c)
            for item in common_items:
                priority = 0
                if item in 'abcdefghijklmnopqrstuvwxyz':
                    priority = 1 + ord(item) - ord('a')
                elif item in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    priority = 27 + ord(item) - ord('A')
                priorities.append(priority)
        result = sum(priorities)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        rucksacks = self.get_rucksacks(raw_input_lines)
        solutions = (
            self.solve(rucksacks),
            self.solve2(rucksacks),
            )
        result = solutions
        return result

class Day02: # Rock Paper Scissors
    '''
    https://adventofcode.com/2022/day/2
    '''
    scoring_values = {
        'Rock': 1,
        'Paper': 2,
        'Scissors': 3,
        'Loss': 0,
        'Draw': 3,
        'Win': 6,
    }

    messages = {
        'A': 'Rock',
        'B': 'Paper',
        'C': 'Scissors',
    }

    responses = {
        'X': 'Rock',
        'Y': 'Paper',
        'Z': 'Scissors',
        ('Rock', 'Loss'): 'Scissors',
        ('Paper', 'Loss'): 'Rock',
        ('Scissors', 'Loss'): 'Paper',
        ('Rock', 'Draw'): 'Rock',
        ('Paper', 'Draw'): 'Paper',
        ('Scissors', 'Draw'): 'Scissors',
        ('Rock', 'Win'): 'Paper',
        ('Paper', 'Win'): 'Scissors',
        ('Scissors', 'Win'): 'Rock',
    }

    desired_outcomes = {
        'X': 'Loss',
        'Y': 'Draw',
        'Z': 'Win',
    }

    outcomes = {
        ('Rock', 'Rock'): 'Draw',
        ('Rock', 'Paper'): 'Win',
        ('Rock', 'Scissors'): 'Loss',
        ('Paper', 'Rock'): 'Loss',
        ('Paper', 'Paper'): 'Draw',
        ('Paper', 'Scissors'): 'Win',
        ('Scissors', 'Rock'): 'Win',
        ('Scissors', 'Paper'): 'Loss',
        ('Scissors', 'Scissors'): 'Draw',
    }

    def get_strategy_guide(self, raw_input_lines: List[str]):
        strategy_guide = []
        for raw_input_line in raw_input_lines:
            message, response = raw_input_line.split(' ')
            strategy_guide.append(
                (
                    self.messages[message],
                    self.responses[response],
                )
            )
        result = strategy_guide
        return result

    def get_strategy_guide_v2(self, raw_input_lines: List[str]):
        strategy_guide_v2 = []
        for raw_input_line in raw_input_lines:
            message, desired_outcome = raw_input_line.split(' ')
            strategy_guide_v2.append(
                (
                    self.messages[message],
                    self.desired_outcomes[desired_outcome],
                )
            )
        result = strategy_guide_v2
        return result
    
    def solve(self, strategy_guide):
        total_score = 0
        for (message, response) in strategy_guide:
            total_score += self.scoring_values[response]
            outcome = self.outcomes[(message, response)]
            total_score += self.scoring_values[outcome]
        result = total_score
        return result
    
    def solve2(self, strategy_guide_v2):
        total_score = 0
        for (message, desired_outcome) in strategy_guide_v2:
            outcome = desired_outcome
            response = self.responses[(message, outcome)]
            total_score += self.scoring_values[response]
            total_score += self.scoring_values[outcome]
        result = total_score
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        strategy_guide = self.get_strategy_guide(raw_input_lines)
        strategy_guide_v2 = self.get_strategy_guide_v2(raw_input_lines)
        solutions = (
            self.solve(strategy_guide),
            self.solve2(strategy_guide_v2),
            )
        result = solutions
        return result

class Day01: # Calorie Counting
    '''
    https://adventofcode.com/2022/day/1
    '''
    def get_elves(self, raw_input_lines: List[str]):
        elves = [[]]
        for raw_input_line in raw_input_lines:
            if len(raw_input_line) < 1:
                elves.append([])
            else:
                elves[-1].append(int(raw_input_line))
        result = elves
        return result
    
    def solve(self, elves):
        result = max(sum(calories) for calories in elves)
        return result
    
    def solve2(self, elves):
        heap = []
        for calories in elves:
            heapq.heappush(heap, -1 * sum(calories))
        top_calories = []
        top_calories.append(-1 * heapq.heappop(heap))
        top_calories.append(-1 * heapq.heappop(heap))
        top_calories.append(-1 * heapq.heappop(heap))
        result = sum(top_calories)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        elves = self.get_elves(raw_input_lines)
        solutions = (
            self.solve(elves),
            self.solve2(elves),
            )
        result = solutions
        return result

if __name__ == '__main__':
    '''
    Usage
    python AdventOfCode2022.py 15 < inputs/2022day15.in
    '''
    solvers = {
        1: (Day01, 'Calorie Counting'),
        2: (Day02, 'Rock Paper Scissors'),
        3: (Day03, 'Rucksack Reorganization'),
        4: (Day04, 'Camp Cleanup'),
        5: (Day05, 'Supply Stacks'),
        6: (Day06, 'Tuning Trouble'),
        7: (Day07, 'No Space Left On Device'),
        8: (Day08, 'Treetop Tree House'),
        9: (Day09, 'Rope Bridge'),
       10: (Day10, 'Cathode-Ray Tube'),
       11: (Day11, 'Monkey in the Middle'),
       12: (Day12, 'Hill Climbing Algorithm'),
       13: (Day13, 'Distress Signal'),
       14: (Day14, 'Regolith Reservoir'),
       15: (Day15, 'Beacon Exclusion Zone'),
       16: (Day16, 'Proboscidea Volcanium'),
    #    17: (Day17, 'Day17'),
    #    18: (Day18, 'Day18'),
    #    19: (Day19, 'Day19'),
    #    20: (Day20, 'Day20'),
    #    21: (Day21, 'Day21'),
    #    22: (Day22, 'Day22'),
    #    23: (Day23, 'Day23'),
    #    24: (Day24, 'Day24'),
    #    25: (Day25, 'Day25'),
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
