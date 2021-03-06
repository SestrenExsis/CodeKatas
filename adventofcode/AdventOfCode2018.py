'''
Created on Dec 30, 2020

@author: Sestren
'''
import argparse
import collections
import copy
import datetime
import functools
import heapq
import operator
import re
import time
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

class WristDeviceProgram:
    '''
    4 registers (0-3) or 6 registers (0-5)
    16 opcodes (0-15)
    instruction = opcode A B C
        inputs: A, B
        output: C (a register)
    "value A" or "register A"
    '''
    def __init__(self, register_count, program, ip_mode: int=-1):
        self.register = [0] * register_count
        self.program = program
        '''
        When the instruction pointer mode is -1, the program counter is
        separate (self.pc). Otherwise, the instruction pointer mode is
        the index of the register that contains the program counter
        '''
        self.ip_mode = ip_mode
        self.pc = 0
        self.debug = False
        self.execution_count = 0
    
    def run(self):
        if self.debug:
            with open('AdventOfCode2018.out', 'w') as out:
                out.write('Start\n')
        running = True
        while running:
            running = self.execute()
        if self.debug:
            with open('AdventOfCode2018.out', 'a') as out:
                out.write('End')
    
    def execute(self) -> bool:
        result = False
        if 0 <= self.ip_mode < len(self.register):
            self.register[self.ip_mode] = self.pc
        if self.pc < 0 or self.pc >= len(self.program):
            return result
        instruction, a, b, c = self.program[self.pc]
        command = ''
        if instruction == 'addr':
            # ADDR: register A + register B --> register C
            if (
                0 <= a < len(self.register) and
                0 <= b < len(self.register) and
                0 <= c < len(self.register)
            ):
                command = f'r[{c}] = r[{a}] + r[{b}]'
                self.register[c] = self.register[a] + self.register[b]
                result = True
        elif instruction == 'addi':
            # ADDI: register A + value B --> register C
            if (
                0 <= a < len(self.register) and
                0 <= c < len(self.register)
            ):
                command = f'r[{c}] = r[{a}] + {b}'
                self.register[c] = self.register[a] + b
                result = True
        elif instruction == 'mulr':
            # MULR: register A * register B --> register C
            if (
                0 <= a < len(self.register) and
                0 <= b < len(self.register) and
                0 <= c < len(self.register)
            ):
                command = f'r[{c}] = r[{a}] * r[{b}]'
                self.register[c] = self.register[a] * self.register[b]
                result = True
        elif instruction == 'muli':
            # MULI: register A * value B --> register C
            if (
                0 <= a < len(self.register) and
                0 <= c < len(self.register)
            ):
                command = f'r[{c}] = r[{a}] * {b}'
                self.register[c] = self.register[a] * b
                result = True
        elif instruction == 'banr':
            # BANR: register A & register B --> register C
            if (
                0 <= a < len(self.register) and
                0 <= b < len(self.register) and
                0 <= c < len(self.register)
            ):
                command = f'r[{c}] = r[{a}] & r[{b}]'
                self.register[c] = self.register[a] & self.register[b]
                result = True
        elif instruction == 'bani':
            # BANI: register A & value B --> register C
            if (
                0 <= a < len(self.register) and
                0 <= c < len(self.register)
            ):
                command = f'r[{c}] = r[{a}] & {b}'
                self.register[c] = self.register[a] & b
                result = True
        elif instruction == 'borr':
            # BORR: register A | register B --> register C
            if (
                0 <= a < len(self.register) and
                0 <= b < len(self.register) and
                0 <= c < len(self.register)
            ):
                command = f'r{c} = r[{a}] | r[{b}]'
                self.register[c] = self.register[a] | self.register[b]
                result = True
        elif instruction == 'bori':
            # BORI: register A | value B --> register C
            if (
                0 <= a < len(self.register) and
                0 <= c < len(self.register)
            ):
                command = f'r[{c}] = r[{a}] | {b}'
                self.register[c] = self.register[a] | b
                result = True
        elif instruction == 'setr':
            # SETR: register A --> register C
            if (
                0 <= a < len(self.register) and
                0 <= c < len(self.register)
            ):
                command = f'r[{c}] = r[{a}]'
                self.register[c] = self.register[a]
                result = True
        elif instruction == 'seti':
            # SETI: value A --> register C
            if 0 <= c < len(self.register):
                command = f'r[{c}] = {a}'
                self.register[c] = a
                result = True
        elif instruction == 'gtir':
            # GTIR: 1 if value A > register B else 0 --> register C
            if (
                0 <= b < len(self.register) and
                0 <= c < len(self.register)
            ):
                command = f'r[{c}] = {a} > r[{b}]'
                self.register[c] = 1 if a > self.register[b] else 0
                result = True
        elif instruction == 'gtri':
            # GTRI: 1 if register A > value B else 0 --> register C
            if (
                0 <= a < len(self.register) and
                0 <= c < len(self.register)
            ):
                command = f'r[{c}] = r[{a}] > {b}'
                self.register[c] = 1 if self.register[a] > b else 0
                result = True
        elif instruction == 'gtrr':
            # GTRR: 1 if register A > register B else 0 --> register C
            if (
                0 <= a < len(self.register) and
                0 <= b < len(self.register) and
                0 <= c < len(self.register)
            ):
                command = f'r[{c}] = r[{a}] > r[{b}]'
                self.register[c] = 1 if self.register[a] > self.register[b] else 0
                result = True
        elif instruction == 'eqir':
            # EQIR: 1 if value A == register B else 0 --> register C
            if (
                0 <= b < len(self.register) and
                0 <= c < len(self.register)
            ):
                command = f'r[{c}] = {a} == r[{b}]'
                self.register[c] = 1 if a == self.register[b] else 0
                result = True
        elif instruction == 'eqri':
            # EQRI: 1 if register A == value B else 0 --> register C
            if (
                0 <= a < len(self.register) and
                0 <= c < len(self.register)
            ):
                command = f'r[{c}] = r[{a}] == {b}'
                self.register[c] = 1 if self.register[a] == b else 0
                result = True
        elif instruction == 'eqrr':
            # EQRR: 1 if register A == register B else 0 --> register C
            if (
                0 <= a < len(self.register) and
                0 <= b < len(self.register) and
                0 <= c < len(self.register)
            ):
                command = f'r[{c}] = r[{a}] == r[{b}]'
                self.register[c] = 1 if self.register[a] == self.register[b] else 0
                result = True
        if self.debug:
            with open('AdventOfCode2018.out', 'a') as out:
                output_line = f'[{self.pc}] {command} {self.register}\n'
                out.write(output_line)
                print(output_line, end='')
        if 0 <= self.ip_mode < len(self.register):
            self.pc = self.register[self.ip_mode]
            self.execution_count += 1
        self.pc += 1
        return result

class Template: # Template
    '''
    Template
    https://adventofcode.com/2018/day/?
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

class Day25: # Four-Dimensional Adventure
    '''
    Four-Dimensional Adventure
    https://adventofcode.com/2018/day/25
    '''
    def get_points(self, raw_input_lines: List[str]):
        points = []
        for raw_input_line in raw_input_lines:
            w, x, y, z = tuple(map(int, raw_input_line.split(',')))
            points.append((w, x, y, z))
        result = points
        return result
    
    def solve(self, points):
        constellation_count = 0
        points_left = set(points)
        while len(points_left) > 0:
            work = [min(points_left)]
            while len(work) > 0:
                point1 = work.pop()
                w1, x1, y1, z1 = point1
                if point1 not in points_left:
                    continue
                points_left.remove(point1)
                next_points = set()
                for point2 in points_left:
                    w2, x2, y2, z2 = point2
                    distance = sum([
                        abs(w2 - w1),
                        abs(x2 - x1),
                        abs(y2 - y1),
                        abs(z2 - z1),
                    ])
                    if distance <= 3:
                        work.append(point2)
                        next_points.add(point2)
            constellation_count += 1
        result = constellation_count
        return result
    
    def solve2(self, points):
        result = len(points)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        points = self.get_points(raw_input_lines)
        solutions = (
            self.solve(points),
            self.solve2(points),
            )
        result = solutions
        return result

class Day24: # Immune System Simulator 20XX
    '''
    Immune System Simulator 20XX
    https://adventofcode.com/2018/day/24
    '''

    class Unit(object):
        def __init__(self,
            unit_id: int,
            team_id: int,
            group_id: int,
            count: int,
            hp: int,
            attack: str,
            dmg: int,
            init: int,
            weak: set,
            immune: set,
        ):
            self.unit_id = unit_id
            self.team_id = team_id
            self.group_id = group_id
            self.count = count
            self.hp = hp
            self.attack = attack
            self.dmg = dmg
            self.init = init
            self.weak = weak
            self.immune = immune

        def ep(self):
            result = self.count * self.dmg
            return result
        
        def attack_damage(self, target: 'Unit'):
            multiplier = 1
            if self.attack in target.immune:
                multiplier = 0
            elif self.attack in target.weak:
                multiplier = 2
            result = multiplier * self.ep()
            return result

    def get_units(self, raw_input_lines: List[str]):
        pattern_str = r'^(\d+) units each with (\d+) hit points (.*)with an attack that does (\d+) (.+) damage at initiative (\d+)$'
        pattern = re.compile(pattern_str)
        # team, unit_count, hp, initiative, attack_type, attack_damage, weaknesses, immunities
        units = {}
        team_id = -1
        group_ids_by_team = [0, 0]
        for unit_id, raw_input_line in enumerate(raw_input_lines):
            if len(raw_input_line) < 1:
                continue
            elif raw_input_line[-1] == ':':
                if raw_input_line[:-1] == 'Immune System':
                    team_id = 0
                else:
                    team_id = 1
            else:
                group_id = group_ids_by_team[team_id]
                group_ids_by_team[team_id] += 1
                matches = pattern.match(raw_input_line)
                count = int(matches.group(1))
                hp = int(matches.group(2))
                attack = matches.group(5)
                dmg = int(matches.group(4))
                init = int(matches.group(6))
                weak = set()
                immune = set()
                parts = matches.group(3)[1:-2].split(';')
                for part in parts:
                    part2 = part.strip().split(', ')
                    part3 = part2[0].split(' ')
                    if part3[0] == 'immune':
                        immune.add(part3[2])
                        for subpart in part2[1:]:
                            immune.add(subpart)
                    elif part3[0] == 'weak':
                        weak.add(part3[2])
                        for subpart in part2[1:]:
                            weak.add(subpart)
                unit = self.Unit(
                    unit_id, team_id, group_id,
                    count, hp, attack, dmg, init, weak, immune
                )
                units[unit_id] = unit
        result = units
        return result
    
    def write_combat_log(self, combat_log):
        with open('AdventOfCode2018.out', 'w') as out:
            for i in range(len(combat_log)):
                out.write('Round #{}\n'.format(i + 1))
                for j in range(len(combat_log[i])):
                    out.write(combat_log[i][j] + '\n')

    def show_attack(self, combat_log, attacker, target):
        attack_damage = attacker.attack_damage(target)
        kills = min(attack_damage // target.hp, target.count)
        team = ['Immune System', 'Infection'][attacker.team_id]
        combat_log[-1].append(f"    {team} group {attacker.group_id} attacks defending group {target.group_id}, killing {kills} units")
    
    def show_units(self, combat_log, units):
        combat_log[-1].append('Immune System:')
        for unit in units.values():
            if unit.team_id == 0:
                combat_log[-1].append(f"    Group {unit.group_id} contains {unit.count} units")
        combat_log[-1].append('Infection:')
        for unit in units.values():
            if unit.team_id == 1:
                combat_log[-1].append(f"    Group {unit.group_id} contains {unit.count} units")

    def fight(self, units, write_to_log: bool):
        combat_log = []
        while True:
            # units = sorted(units, key=lambda unit: (-unit.ep, -unit.init))
            combat_log.append([])
            self.show_units(combat_log, units)
            combat_log[-1].append('Attacks:')
            # Units select their targets in descending order by effective power
            # and then by initiative in the event of a tie
            roster = []
            for unit_id, unit in units.items():
                heapq.heappush(roster, (-unit.ep(), -unit.init, unit_id))
            # Select targets
            attacks = set()
            chosen_targets = set()
            while len(roster) > 0:
                (_, _, attacking_unit_id) = heapq.heappop(roster)
                attacker = units[attacking_unit_id]
                potential_attacks = []
                max_attack_damage = 1
                for target_unit_id, target in units.items():
                    if (
                        target.team_id == attacker.team_id or
                        target.count < 1 or
                        attacker.count < 1
                    ):
                        continue
                    attack_damage = attacker.attack_damage(target)
                    if (
                        attack_damage >= max_attack_damage and
                        target_unit_id not in chosen_targets
                    ):
                        heapq.heappush(potential_attacks, (
                            -attack_damage,
                            -target.ep(),
                            -target.init,
                            attacker.init,
                            attacking_unit_id,
                            target_unit_id,
                        ))
                        max_attack_damage = attack_damage
                attack = None
                while len(potential_attacks) > 0:
                    _, _, _, init, attacking_unit_id, target_unit_id = heapq.heappop(potential_attacks)
                    if target_unit_id not in chosen_targets:
                        attack = (init, attacking_unit_id, target_unit_id)
                        attacks.add(attack)
                        chosen_targets.add(target_unit_id)
                        break
            # Resolve attacks
            # Attacks are resolved in decreasing order of initiative
            total_kills = 0
            for attack in sorted(attacks, key=lambda x: -x[0]):
                _, attacking_unit_id, target_unit_id = attack
                if (
                    attacking_unit_id not in units or
                    target_unit_id not in units or
                    units[attacking_unit_id].count < 1
                ):
                    continue
                attacker = units[attacking_unit_id]
                target = units[target_unit_id]
                attack_damage = attacker.attack_damage(target)
                kills = min(target.count, attack_damage // target.hp)
                total_kills += kills
                self.show_attack(combat_log, attacker, target)
                target.count -= kills
                if target.count < 1:
                    units.pop(target_unit_id)
            # If nobody died, the fight stops
            if total_kills < 1:
                break
            # self.show_units(combat_log, units)
        outcome = -1
        units_left = [0, 0]
        units_left[0] = sum(
            max(0, unit.count) for
            unit in units.values() if
            unit.team_id == 0
            )
        units_left[1] = sum(
            max(0, unit.count) for
            unit in units.values() if
            unit.team_id == 1
            )
        if units_left[0] < 1 and units_left[1] > 0:
            outcome = 1
        elif units_left[0] > 0 and units_left[1] < 1:
            outcome = 0
        result = (outcome, units_left[0], units_left[1])
        if write_to_log:
            self.write_combat_log(combat_log)
        return result
    
    def solve(self, units):
        winning_team = self.fight(units, False)
        result = sum(max(0, unit.count) for unit in units.values())
        return result
    
    def solve2(self, units):
        # Find the smallest boost that allows the immune system to win
        max_boost = 10_000_000 # max(unit.count * unit.hp for unit in units.values())
        min_boost = 0
        boosts = {}
        while min_boost < max_boost:
            curr_units = copy.deepcopy(units)
            boost = min_boost + (max_boost - min_boost) // 2
            for unit_id in curr_units:
                unit = curr_units[unit_id]
                if unit.team_id == 0:
                    curr_units[unit_id].dmg += boost
            outcome, team0_left, team1_left = self.fight(curr_units, False)
            boosts[boost] = (outcome, team0_left, team1_left)
            if outcome == 0:
                max_boost = boost - 1
            elif outcome == 1:
                min_boost = boost + 1
            else:
                break
        for boost in range(min_boost - 1, max_boost + 1):
            curr_units = copy.deepcopy(units)
            for unit_id in curr_units:
                unit = curr_units[unit_id]
                if unit.team_id == 0:
                    curr_units[unit_id].dmg += boost
            outcome, team0_left, team1_left = self.fight(curr_units, False)
            boosts[boost] = (outcome, team0_left, team1_left)
        min_boost = min(boost for boost, (outcome, _, _) in boosts.items() if outcome == 0)
        result = boosts[min_boost][1]
        return result
    
    def main(self):
        # 26594 is too high for part 2
        raw_input_lines = get_raw_input_lines()
        units = self.get_units(raw_input_lines)
        solutions = (
            self.solve(copy.deepcopy(units)),
            self.solve2(copy.deepcopy(units)),
            )
        result = solutions
        return result

class Day23: # Experimental Emergency Teleportation
    '''
    Experimental Emergency Teleportation
    https://adventofcode.com/2018/day/23
    '''
    def get_nanobots(self, raw_input_lines: List[str]):
        nanobots = []
        for raw_input_line in raw_input_lines:
            a, b = raw_input_line[5:].split('>, r=')
            radius = int(b)
            pos = tuple(map(int, a.split(',')))
            nanobot = (radius, pos[0], pos[1], pos[2])
            nanobots.append(nanobot)
        result = nanobots
        return result
    
    def solve(self, nanobots):
        strongest_nanobot = sorted(nanobots)[-1]
        r1, x1, y1, z1 = strongest_nanobot
        within_range_count = 0
        for (r2, x2, y2, z2) in nanobots:
            if abs(x2 - x1) + abs(y2 - y1) + abs(z2 - z1) <= r1:
                within_range_count += 1
        result = within_range_count
        return result
    
    def solve2(self, nanobots):
        work = []
        for (radius, x, y, z) in nanobots:
            distance = abs(x) + abs(y) + abs(z)
            heapq.heappush(work, (max(0, distance - radius), 1))
            heapq.heappush(work, (distance + radius + 1, -1))
        count = 0
        max_count = 0
        closest_distance = 0
        while len(work) > 0:
            distance, delta = heapq.heappop(work)
            count += delta
            if count > max_count:
                closest_distance = distance
                max_count = count
        result = closest_distance
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        nanobots = self.get_nanobots(raw_input_lines)
        solutions = (
            self.solve(nanobots),
            self.solve2(nanobots),
            )
        result = solutions
        return result

class Day22: # Mode Maze
    '''
    Mode Maze
    https://adventofcode.com/2018/day/22
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        depth = int(raw_input_lines[0].split(' ')[1])
        raw_target = raw_input_lines[1].split(' ')[1]
        target = tuple(map(int, raw_target.split(',')))
        result = (depth, (target[1], target[0]))
        return result
    
    def solve(self, depth, target):
        '''
        Modular arithmetic properties
        Addition      : (a + b) % n = [a % n + b % n] % n
        Multiplication: (a * b) % n = [a % n * b % n] % n
        '''
        MODULO = 20183
        geo_indexes = {}
        geo_indexes[(0, 0)] = 0
        geo_indexes[target] = 0
        erosion_levels = {}
        for row in range(0, target[0] + 1):
            for col in range(0, target[1] + 1):
                if (row, col) not in geo_indexes:
                    geo_index = 0
                    if row == 0:
                        geo_index = col * 16_807
                    elif col == 0:
                        geo_index = row * 48_271
                    else:
                        a = erosion_levels[(row - 1, col)]
                        b = erosion_levels[(row, col - 1)]
                        geo_index = a * b
                    geo_indexes[(row, col)] = geo_index
                if (row, col) not in erosion_levels:
                    erosion_level = (geo_indexes[(row, col)] + depth) % MODULO
                    erosion_levels[(row, col)] = erosion_level
        total_risk_level = sum(
            erosion_level % 3 for
            erosion_level in
            erosion_levels.values()
            )
        result = total_risk_level
        return result
    
    def solve2(self, depth, target):
        '''
        Negative rows or negative columns are impassible terrain
        Moving costs 1 minute
        Equipping costs 7 minutes
        With NOTHING equipped, you can enter 1 or 2
        With TORCH equipped, you can enter 0 or 2
        With CLIMBING GEAR equipped, you can enter 0 or 1
        equipment can enter terrain that isn't equal to its index
        You start at (0, 0, 1)
        You need to be at (target[0], target[1], 1)
        '''
        MODULO = 20183
        NOTHING = 0
        TORCH = 1
        CLIMBING_GEAR = 2
        geo_indexes = {}
        geo_indexes[(0, 0)] = 0
        geo_indexes[target] = 0
        erosion_levels = {}
        furthest_row = -1
        furthest_col = -1
        def get_terrain(next_row, next_col):
            nonlocal depth, target
            nonlocal geo_indexes, erosion_levels, furthest_row, furthest_col
            row = furthest_row + 1
            while row <= next_row:
                for col in range(furthest_col + 1):
                    if (row, col) not in geo_indexes:
                        geo_index = 0
                        if row == 0:
                            geo_index = col * 16_807
                        elif col == 0:
                            geo_index = row * 48_271
                        else:
                            a = erosion_levels[(row - 1, col)]
                            b = erosion_levels[(row, col - 1)]
                            geo_index = a * b
                        geo_indexes[(row, col)] = geo_index
                    if (row, col) not in erosion_levels:
                        erosion_level = (geo_indexes[(row, col)] + depth) % MODULO
                        erosion_levels[(row, col)] = erosion_level
                row += 1
            furthest_row = max(furthest_row, next_row)
            col = furthest_col + 1
            while col <= next_col:
                for row in range(furthest_row + 1):
                    if (row, col) not in geo_indexes:
                        geo_index = 0
                        if row == 0:
                            geo_index = col * 16_807
                        elif col == 0:
                            geo_index = row * 48_271
                        else:
                            a = erosion_levels[(row - 1, col)]
                            b = erosion_levels[(row, col - 1)]
                            geo_index = a * b
                        geo_indexes[(row, col)] = geo_index
                    if (row, col) not in erosion_levels:
                        erosion_level = (geo_indexes[(row, col)] + depth) % MODULO
                        erosion_levels[(row, col)] = erosion_level
                col += 1
            furthest_col = max(furthest_col, next_col)
            terrain = erosion_levels[(next_row, next_col)] % 3
            return terrain
        min_t = float('inf')
        visits = {}
        work = [(0, 0, 0, TORCH)] # (t, row, col, equip)
        while len(work) > 0:
            (t, row, col, equip) = heapq.heappop(work)
            if (row, col, equip) not in visits:
                visits[(row, col, equip)] = t
            elif visits[(row, col, equip)] <= t:
                continue
            visits[(row, col, equip)] = min(visits[(row, col, equip)], t)
            if (row, col, equip) == (target[0], target[1], TORCH):
                min_t = t
                break
            terrain = get_terrain(row, col)
            # The enumeration values of the equipment match the terrain type
            # They are NOT allowed to function in
            next_equips = {NOTHING, TORCH, CLIMBING_GEAR}
            next_equips.remove(terrain)
            next_equips.remove(equip)
            next_equip = next_equips.pop()
            heapq.heappush(work, (t + 7, row, col, next_equip))
            for (next_row, next_col) in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if next_row < 0 or next_col < 0:
                    continue
                next_terrain = get_terrain(next_row, next_col)
                if equip != next_terrain:
                    heapq.heappush(work, (t + 1, next_row, next_col, equip))
        result = min_t
        # self.show_grid(erosion_levels, 0, 10, 0, 10)
        return result
    
    def show_grid(self, erosion_levels, min_row, max_row, min_col, max_col):
        for row in range(min_row, max_row + 1):
            row_data = []
            for col in range(min_col, max_col + 1):
                cell = '.'
                terrain = erosion_levels[(row, col)] % 3
                if terrain == 1:
                    cell = '='
                elif terrain == 2:
                    cell = '|'
                row_data.append(cell)
            print(''.join(row_data))
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        (depth, target) = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(depth, target),
            self.solve2(depth, target), #1042 is too low
            )
        result = solutions
        return result

class Day21: # Chronal Conversion
    '''
    Chronal Conversion
    https://adventofcode.com/2018/day/21
    '''
    def get_magic_number(self, raw_input_lines: List[str]):
        magic_number = int(raw_input_lines[8].split(' ')[1])
        result = magic_number
        return result
    
    def solve_hardcoded(self, magic_number):
        # Initial rewrite
        '''
        E = 0
        B = (E | 65_536)    # [Line_06]
        E = magic_number
        D = B & 255         # [Line_08]
        E += D
        E &= 16_777_215
        E *= 65_899
        E &= 16_777_215
        D = (256 > B)
        pc += D
        pc += 1
        goto [Line_28]
        D = 0
        C = D + 1           # [Line_18]
        C *= 256
        C = (C > B)
        pc += C
        pc += 1
        goto [Line_26]
        D += 1
        goto [Line_18]
        B = D               # [Line_26]
        goto [Line_08]
        D = (E == A)        # [Line_28]
        pc += D
        goto [Line_06]
        '''
        # Refactor
        result = -1
        A, B, C = 0, 0, 0
        while True:
            A = (C | 65_536)
            C = magic_number
            while True:
                B = A & 255
                C += B
                C &= 16_777_215
                C *= 65_899
                C &= 16_777_215
                if 256 > A:
                    break
                B = 0
                while 256 * (B + 1) <= A:
                    B += 1
                A = B
            if C >= 0:
                result = C
                break
        return result
    
    def solve(self, magic_number):
        C = 0
        result = -1
        while result == -1:
            A = C | 65_536
            C = magic_number
            while True:
                B = A & 255
                C += B
                C &= 16_777_215
                C *= 65_899
                C &= 16_777_215
                if 256 > A:
                    result = C
                    break
                else:
                    A //= 256
        return result
    
    def solve2(self, magic_number):
        seen = set()
        C = 0
        cycle_found_ind = False
        result = -1
        while not cycle_found_ind:
            A = C | 65_536
            C = magic_number
            while True:
                B = A & 255
                C += B
                C &= 16_777_215
                C *= 65_899
                C &= 16_777_215
                if 256 > A:
                    if C in seen:
                        cycle_found_ind = True
                    else:
                        seen.add(C)
                        result = C
                    break
                else:
                    A //= 256
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        magic_number = self.get_magic_number(raw_input_lines)
        solutions = (
            self.solve(magic_number),
            self.solve2(magic_number),
            )
        result = solutions
        return result

class Day20: # A Regular Map
    '''
    A Regular Map
    https://adventofcode.com/2018/day/20
    '''
    def get_route(self, raw_input_lines: List[str]):
        route = raw_input_lines[0]
        result = route
        return result
    
    def get_rooms(self, route):
        directions = {
            'N': (-1, 0),
            'S': ( 1, 0),
            'W': ( 0,-1),
            'E': ( 0, 1),
        }
        # Map out all rooms and doors between them
        rooms = collections.defaultdict(set)
        row = 0
        col = 0
        stack = []
        for i in range(len(route)):
            char = route[i]
            if char == '(':
                stack.append((row, col))
            elif char == '|':
                row, col = stack[-1]
            elif char == ')':
                stack.pop()
            elif char in directions:
                row_offset, col_offset = directions[char]
                next_row, next_col = row + row_offset, col + col_offset
                rooms[(row, col)].add((next_row, next_col))
                rooms[(next_row, next_col)].add((row, col))
                row = next_row
                col = next_col
        result = rooms
        return result
    
    def get_distances(self, rooms):
        # Find shortest paths to each room
        row, col = 0, 0
        distances = {}
        work = collections.deque()
        work.append((0, 0, 0)) # distance, row, col
        while len(work) > 0:
            distance, row, col = work.pop()
            if (row, col) in distances:
                continue
            distances[(row, col)] = distance
            for next_row, next_col in rooms[(row, col)]:
                work.appendleft((distance + 1, next_row, next_col))
        result = distances
        return result
    
    def solve(self, route):
        rooms = self.get_rooms(route)
        distances = self.get_distances(rooms)
        # Find furthest-away room
        result = max(distances.values())
        return result
    
    def solve2(self, route):
        rooms = self.get_rooms(route)
        distances = self.get_distances(rooms)
        result = sum(1 for distance in distances.values() if distance >= 1000)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        route = self.get_route(raw_input_lines)
        solutions = (
            self.solve(route),
            self.solve2(route),
            )
        result = solutions
        return result

class Day19: # Go With The Flow
    '''
    Go With The Flow
    https://adventofcode.com/2018/day/19
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        ip_mode = int(raw_input_lines[0].split(' ')[1])
        program = []
        for raw_input_line in raw_input_lines[1:]:
            parts = raw_input_line.split(' ' )
            instruction = (
                parts[0],
                int(parts[1]),
                int(parts[2]),
                int(parts[3])
                )
            program.append(instruction)
        result = ip_mode, program
        return result
    
    def solve(self, program, ip_mode):
        vm = WristDeviceProgram(6, program, ip_mode)
        vm.run()
        result = vm.register[0]
        return result
    
    def solve2_slow(self, program, ip_mode):
        vm = WristDeviceProgram(6, program, ip_mode)
        vm.debug = True
        vm.register[0] = 1
        vm.run()
        result = vm.register[0]
        return result

    def solve2_hardcoded(self):
        # First pass refactor
        '''
        CONST = 10_551_387
        a, b, c = 0, 1, 0
        b = 1           # [Line 1]
        c = 1           # [Line 2]
        if b * c == CONST:
            a += b
        c += 1
        if c <= CONST:
            goto [Line 2]
        b += 1
        if b <= CONST:
            goto [Line 1]
        return a
        '''
        # Second pass refactor
        '''
        a = 0
        CONST = 10_551_387
        for b in range(1, CONST + 1):
            for c in range(1, CONST + 1):
                if b * c == CONST:
                    a += b
            print(b, ':', a)
        result = a
        '''
        # Final pass refactor
        a = 0
        CONST = 10_551_387
        for b in range(1, CONST + 1):
            if CONST % b == 0:
                a += b
        result = a
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        ip_mode, program = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(program, ip_mode),
            self.solve2_hardcoded(),
            )
        result = solutions
        return result

class Day18: # Settlers of The North Pole
    '''
    Settlers of The North Pole
    https://adventofcode.com/2018/day/18
    '''
    def get_acres(self, raw_input_lines: List[str]):
        acres = {}
        for row, raw_input_line in enumerate(raw_input_lines):
            for col, cell in enumerate(raw_input_line):
                acres[(row, col)] = cell
        result = acres
        return result
    
    def simulate(self, acres):
        next_acres = {}
        for (row, col), cell in acres.items():
            next_acres[(row, col)] = cell
            woods = 0
            lumberyards = 0
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
                if (nrow, ncol) not in acres:
                    continue
                neighbor = acres[(nrow, ncol)]
                if neighbor == '|':
                    woods += 1
                elif neighbor == '#':
                    lumberyards += 1
            if cell == '.':
                if woods >= 3:
                    next_acres[(row, col)] = '|'
            elif cell == '|':
                if lumberyards >= 3:
                    next_acres[(row, col)] = '#'
            elif cell == '#':
                if woods < 1 or lumberyards < 1:
                    next_acres[(row, col)] = '.'
        result = next_acres
        return result
    
    def solve(self, acres, minutes: int):
        for _ in range(minutes):
            acres = self.simulate(acres)
        wooded_acre_count = sum(1 for cell in acres.values() if cell == '|')
        lumberyard_count = sum(1 for cell in acres.values() if cell == '#')
        result = wooded_acre_count * lumberyard_count
        return result
    
    def hashable_acres(self, acres) -> str:
        cells = []
        row = 0
        while True:
            if (row, 0) not in acres:
                break
            col = 0
            while (row, col) in acres:
                cells.append(acres[(row, col)])
                col += 1
            row += 1
        result = ''.join(cells)
        return result
    
    def solve2(self, acres, minutes: int):
        prior_acres = set()
        history = {}
        for minute in range(minutes):
            hashable_acres = self.hashable_acres(acres)
            if hashable_acres in prior_acres:
                break
            prior_acres.add(hashable_acres)
            history[hashable_acres] = (minute, acres)
            acres = self.simulate(acres)
        cycle_size = minute - history[self.hashable_acres(acres)][0]
        remaining_minutes = (minutes - minute) % cycle_size
        for minute in range(remaining_minutes):
            acres = self.simulate(acres)
        wooded_acre_count = sum(1 for cell in acres.values() if cell == '|')
        lumberyard_count = sum(1 for cell in acres.values() if cell == '#')
        result = wooded_acre_count * lumberyard_count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        acres = self.get_acres(raw_input_lines)
        solutions = (
            self.solve(copy.deepcopy(acres), 10),
            self.solve2(copy.deepcopy(acres), 1_000_000_000),
            )
        result = solutions
        return result

class Day17: # Reservoir Research
    '''
    Reservoir Research
    https://adventofcode.com/2018/day/17
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        clay = set()
        water = set()
        water.add((0, 500))
        result = []
        for raw_input_line in raw_input_lines:
            a, b = raw_input_line.split(', ')
            if a[0] == 'x':
                col = int(a.split('=')[1])
                c, d = b.split('..')
                min_row = int(c.split('=')[1])
                max_row = int(d)
                for row in range(min_row, max_row + 1):
                    clay.add((row, col))
            elif a[0] == 'y':
                row = int(a.split('=')[1])
                c, d = b.split('..')
                min_col = int(c.split('=')[1])
                max_col = int(d)
                for col in range(min_col, max_col + 1):
                    clay.add((row, col))
        result = clay, water
        return result
    
    def simulate_flow(self, clay, water):
        spring = next(iter(water))
        min_row = min(row for row, col in clay)
        max_row = max(row for row, col in clay)
        min_col = min(col for row, col in clay)
        max_col = max(col for row, col in clay)
        settled = set(clay)
        while True:
            possible_settling_rows = set()
            prev_water_count = len(water)
            # Flow from the spring space
            flow = [spring]
            visited = set()
            while len(flow) > 0:
                (row, col) = flow.pop()
                if (row, col) in visited or row > max_row:
                    continue
                visited.add((row, col))
                if (row, col) not in water:
                    possible_settling_rows.add(row)
                water.add((row, col))
                if (row + 1, col) in settled:
                    if (row, col - 1) not in settled:
                        flow.append((row, col - 1))
                    if (row, col + 1) not in settled:
                        flow.append((row, col + 1))
                else:
                    flow.append((row + 1, col))
            # Check for settling
            for row in possible_settling_rows:
                left_block_ind = False
                settling = set()
                for col in range(min_col, max_col + 1):
                    if (row, col) in clay:
                        if left_block_ind:
                            settled |= settling
                            settling = set()
                        left_block_ind = True
                    elif (row, col) in water:
                        if left_block_ind:
                            settling.add((row, col))
                    else:
                        left_block_ind = False
                        settling = set()
            # Check if new water flow
            if len(water) <= prev_water_count:
                break
        result = (water, settled)
        return result
    
    def solve(self, clay, water):
        (water, _) = self.simulate_flow(clay, water)
        min_row = min(row for row, col in clay)
        max_row = max(row for row, col in clay)
        result = len(set(
            (row, col) for (row, col) in
            water if
            min_row <= row <= max_row
            ))
        return result
    
    def solve2(self, clay, water):
        (water, settled) = self.simulate_flow(clay, water)
        result = len(settled) - len(clay)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        clay, water = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(copy.deepcopy(clay), copy.deepcopy(water)),
            self.solve2(copy.deepcopy(clay), copy.deepcopy(water)),
            )
        result = solutions
        return result

class Day16: # Chronal Classification
    '''
    Chronal Classification
    https://adventofcode.com/2018/day/16

    4 registers (0-3)
    16 opcodes (0-15)
    instruction = opcode A B C
        inputs: A, B
        output: C (a register)
    "value A" or "register A"
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        samples = []
        test_program = []
        input_mode = 'SAMPLES'
        for i, raw_input_line in enumerate(raw_input_lines):
            if input_mode == 'PROGRAM':
                test_program.append(
                    tuple(map(int, raw_input_line.split(' ')))
                    )
            else:
                if 'Before: ' in raw_input_line:
                    samples.append([])
                    array = raw_input_line.split(': [')[1][:-1]
                    before = list(map(int, array.split(', ')))
                    samples[-1].append(before)
                elif 'After: ' in raw_input_line:
                    array = raw_input_line.split(':  [')[1][:-1]
                    after = list(map(int, array.split(', ')))
                    samples[-1].append(after)
                elif len(raw_input_line) > 0:
                    instruction = tuple(map(int, raw_input_line.split(' ')))
                    samples[-1].append(instruction)
            if len(raw_input_line) < 1:
                if (
                    len(raw_input_lines[i - 1]) < 1 and
                    len(raw_input_lines[i - 2]) < 1
                ):
                    input_mode = 'PROGRAM'
        result = samples, test_program
        return result
    
    def test_samples(self, samples):
        tested_samples = collections.defaultdict(set)
        for i in range(len(samples)):
            before = samples[i][0]
            _, a, b, c = samples[i][1]
            after = samples[i][2]
            for instruction in (
                'addr',
                'addi',
                'mulr',
                'muli',
                'banr',
                'bani',
                'borr',
                'bori',
                'setr',
                'seti',
                'gtir',
                'gtri',
                'gtrr',
                'eqir',
                'eqri',
                'eqrr',
            ):
                vm = WristDeviceProgram(4, [
                    (instruction, a, b, c),
                    ])
                vm.register = before[:]
                vm.execute()
                if vm.register == after:
                    tested_samples[i].add(instruction)
        result = tested_samples
        return result
    
    def assign_opcodes(self, samples, tested_samples):
        opcodes = {}
        for sample, instructions in tested_samples.items():
            # print(sample, len(instructions), instructions)
            opcode = samples[sample][1][0]
            if opcode not in opcodes:
                opcodes[opcode] = instructions
            else:
                opcodes[opcode] &= instructions
        assigned_opcodes = [None] * len(opcodes)
        while True:
            for i in range(len(opcodes)):
                if assigned_opcodes[i] is not None:
                    continue
                if len(opcodes[i]) == 1:
                    assigned_opcode = next(iter(opcodes[i]))
                    assigned_opcodes[i] = assigned_opcode
                    for j in range(len(opcodes)):
                        if i == j:
                            continue
                        opcodes[j] -= {assigned_opcode}
            count = sum(
                1 for instruction in
                assigned_opcodes if
                instruction is None
                )
            if count < 1:
                break
        assert len(set(assigned_opcodes)) == len(opcodes)
        result = assigned_opcodes
        return result
    
    def solve(self, samples):
        tested_samples = self.test_samples(samples)
        result = sum(
            1 for instructions in
            tested_samples.values() if
            len(instructions) >= 3
            )
        return result
    
    def solve2(self, samples, test_program):
        tested_samples = self.test_samples(samples)
        opcodes = self.assign_opcodes(samples, tested_samples)
        program = []
        for opcode, a, b, c in test_program:
            instruction = opcodes[opcode]
            program.append((instruction, a, b, c))
        vm = WristDeviceProgram(4, program)
        vm.run()
        result = vm.register[0]
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        samples, test_program = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(samples),
            self.solve2(samples, test_program),
            )
        result = solutions
        return result

class Day15: # Beverage Bandits
    '''
    Beverage Bandits
    https://adventofcode.com/2018/day/15
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        walls = set()
        units = {}
        for row, raw_input_line in enumerate(raw_input_lines):
            for col, cell in enumerate(raw_input_line):
                if cell == '#':
                    walls.add((row, col))
                if cell in 'GE':
                    units[(row, col)] = (cell, 3, 200)
        result = (walls, units)
        return result
    
    def show_grid(self, walls, units):
        left = min(wall[1] for wall in walls)
        right = max(wall[1] for wall in walls)
        top = min(wall[0] for wall in walls)
        bottom = max(wall[0] for wall in walls)
        for row in range(top, bottom + 1):
            stats = []
            cells = []
            for col in range(left, right + 1):
                cell = '.'
                if (row, col) in walls:
                    cell = '#'
                if (row, col) in units:
                    unit = units[(row, col)]
                    cell = unit[0]
                    stats.append(unit[0] + '(' + str(unit[2]) + ')')
                cells.append(cell)
            print(''.join(cells) + '   ' + ', '.join(stats))
    
    def fight(self, walls, units):
        round_count = 0
        goblin_count = sum(1 for unit in units.values() if unit[0] == 'G')
        elf_count = sum(1 for unit in units.values() if unit[0] == 'E')
        combat_active = True
        combat_log = []
        while combat_active:
            combat_log.append([])
            # Iterate over all units in page order by current position
            initiative = list(sorted((row, col) for (row, col) in units))
            for (row, col) in initiative:
                if (row, col) not in units:
                    continue
                unit_type, attack_power, _ = units[(row, col)]
                # If opposing side has run out of units, combat ends
                if (
                    (unit_type == 'G' and elf_count < 1) or
                    (unit_type == 'E' and goblin_count < 1)
                ):
                    combat_active = False
                    break
                # Find nearest location to attack a target from
                valid_attack_locations = set()
                for (trow, tcol), target in units.items():
                    if target[0] != unit_type and target[2] > 0:
                        valid_attack_locations.add((trow - 1, tcol))
                        valid_attack_locations.add((trow + 1, tcol))
                        valid_attack_locations.add((trow, tcol - 1))
                        valid_attack_locations.add((trow, tcol + 1))
                nearest_attack_location = None
                if (row, col) in valid_attack_locations:
                    nearest_attack_location = (row, col, row, col)
                else:
                    work = []
                    heapq.heappush(work, (1, row + 1, col, row + 1, col))
                    heapq.heappush(work, (1, row - 1, col, row - 1, col))
                    heapq.heappush(work, (1, row, col + 1, row, col + 1))
                    heapq.heappush(work, (1, row, col - 1, row, col - 1))
                    visited = set()
                    visited.add((row, col))
                    while len(work) > 0 and nearest_attack_location is None:
                        (dist, nrow, ncol, r, c) = heapq.heappop(work)
                        if (
                            (nrow, ncol) in walls or
                            (nrow, ncol) in units or
                            (nrow, ncol) in visited
                        ):
                            continue
                        if (nrow, ncol) in valid_attack_locations:
                            nearest_attack_location = (nrow, ncol, r, c)
                        visited.add((nrow, ncol))
                        heapq.heappush(work, (dist + 1, nrow + 1, ncol, r, c))
                        heapq.heappush(work, (dist + 1, nrow, ncol + 1, r, c))
                        heapq.heappush(work, (dist + 1, nrow, ncol - 1, r, c))
                        heapq.heappush(work, (dist + 1, nrow - 1, ncol, r, c))
                if nearest_attack_location is not None:
                    distance = sum((
                        abs(nearest_attack_location[0] - row),
                        abs(nearest_attack_location[1] - col),
                        ))
                    if distance > 0:
                        # Move to the nearest attack location if not already
                        # in range to attack
                        mrow = nearest_attack_location[2]
                        mcol = nearest_attack_location[3]
                        combat_log[-1].append(
                            ' {} moves from ({}, {}) to ({}, {})'.
                            format(
                                unit_type, row, col,
                                mrow, mcol,
                                )
                            )
                        units[(mrow, mcol)] = units.pop((row, col))
                        distance = sum((
                            abs(nearest_attack_location[0] - mrow),
                            abs(nearest_attack_location[1] - mcol),
                            ))
                        row = mrow
                        col = mcol
                    # If one or more targets in range, attack one of them
                    # Choose the target with the lowest health, resolve
                    # ties using page order
                    targets = []
                    for (trow, tcol) in (
                        (row + 1, col),
                        (row - 1, col),
                        (row, col + 1),
                        (row, col - 1),
                        ):
                        if (
                            (trow, tcol) in units and
                            units[(trow, tcol)][0] != unit_type
                            ):
                            target = units[(trow, tcol)]
                            heapq.heappush(targets, (target[2], trow, tcol))
                    if len(targets) > 0:
                        _, trow, tcol = heapq.heappop(targets)
                        target = units[(trow, tcol)]
                        # Attack and deal damage to the target
                        target = (
                            target[0],
                            target[1],
                            target[2] - attack_power,
                            )
                        combat_log[-1].append(
                            ' {} at ({}, {}) attacks {} at ({}, {}), hp={}'.
                            format(
                                unit_type, row, col,
                                target[0], trow, tcol, target[2],
                                )
                            )
                        units[(trow, tcol)] = target
                        if target[2] < 1:
                            if target[0] == 'G':
                                goblin_count -= 1
                            elif target[0] == 'E':
                                elf_count -= 1
                            combat_log[-1].append(
                                ' {} dies at ({}, {})'.
                                format(target[0], trow, tcol)
                                )
                            units.pop((trow, tcol))
            else:
                round_count += 1
            if elf_count < 1 or goblin_count < 1:
                combat_active = False
                break
        result = units, round_count, combat_log
        return result
    
    def write_combat_log(self, combat_log):
        with open('AdventOfCode2018.out', 'w') as out:
            for i in range(len(combat_log)):
                out.write('Round #{}\n'.format(i + 1))
                for j in range(len(combat_log[i])):
                    out.write(combat_log[i][j] + '\n')
    
    def solve(self, walls, units):
        units, round_count, _ = self.fight(walls, units)
        remaining_health = sum(unit[2] for unit in units.values())
        result = round_count * remaining_health
        return result
    
    def solve2(self, walls, units):
        max_ap = 200
        min_ap = 1
        result = None
        while min_ap < max_ap:
            curr_ap = min_ap + (max_ap - min_ap) // 2
            curr_units = copy.deepcopy(units)
            for key, (unit_type, ap, hp) in curr_units.items():
                if unit_type == 'E':
                    curr_units[key] = (unit_type, curr_ap, hp)
            curr_units, round_count, combat_log = self.fight(walls, curr_units)
            remaining_health = sum(unit[2] for unit in curr_units.values())
            result = round_count * remaining_health
            elf_died = False
            for round_id in range(round_count):
                for entry in combat_log[round_id]:
                    if 'E dies ' in entry:
                        elf_died = True
                        break
                if elf_died:
                    break
            if elf_died:
                min_ap = curr_ap + 1
            else:
                max_ap = curr_ap
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        (walls, units) = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(walls, copy.deepcopy(units)),
            self.solve2(walls, copy.deepcopy(units)),
            )
        result = solutions
        return result

class Day14: # Chocolate Charts
    '''
    Chocolate Charts
    https://adventofcode.com/2018/day/14
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = int(raw_input_lines[0])
        return result
    
    def solve(self, recipe_count: int) -> str:
        recipes = '37'
        elf_a = 0
        elf_b = 1
        while len(recipes) < recipe_count + 10:
            score_a = int(recipes[elf_a])
            score_b = int(recipes[elf_b])
            total_score = score_a + score_b
            recipes += str(total_score)
            elf_a = (elf_a + score_a + 1) % len(recipes)
            elf_b = (elf_b + score_b + 1) % len(recipes)
        result = ''.join(
            map(str, recipes[recipe_count : recipe_count + 10])
            )
        return result
    
    def solve2(self, recipe_count: int) -> str:
        result = None
        digits = str(recipe_count)
        recipes = '37'
        elf_a = 0
        elf_b = 1
        while digits not in recipes[-7:]:
            score_a = int(recipes[elf_a])
            score_b = int(recipes[elf_b])
            total_score = score_a + score_b
            recipes += str(total_score)
            elf_a = (elf_a + score_a + 1) % len(recipes)
            elf_b = (elf_b + score_b + 1) % len(recipes)
        result = recipes.index(digits)
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

class Day13: # Mine Cart Madness
    '''
    Mine Cart Madness
    https://adventofcode.com/2018/day/13
    '''
    DIRECTIONS = '^>v<'
    OFFSETS = {
        '^': (-1, 0),
        '>': ( 0, 1),
        'v': ( 1, 0),
        '<': ( 0,-1),
    }

    def get_parsed_input(self, raw_input_lines: List[str]):
        tracks = {}
        carts = {}
        for row, raw_input_line in enumerate(raw_input_lines):
            for col, cell in enumerate(raw_input_line):
                if cell in '|-/\\+':
                    tracks[(row, col)] = cell
                elif cell in '^v<>':
                    direction = self.DIRECTIONS.index(cell)
                    carts[(row, col)] = (direction, 0)
                    if cell in '^v':
                        tracks[(row, col)] = '|'
                    elif cell in '<>':
                        tracks[(row, col)] = '-'
                    else:
                        assert False
        result = tracks, carts
        return result
    
    def solve(self, tracks, carts):
        first_crash_location = None
        while first_crash_location is None:
            next_carts = {}
            positions = set(carts.keys())
            for (row, col) in sorted(carts.keys()):
                positions.remove((row, col))
                (direction, intersection_count) = carts[(row, col)]
                offset = self.OFFSETS[self.DIRECTIONS[direction]]
                row += offset[0]
                col += offset[1]
                if (row, col) in positions:
                    first_crash_location = (col, row)
                    break
                positions.add((row, col))
                if tracks[(row, col)] == '/':
                    direction = self.DIRECTIONS.index('>^<v'[direction])
                elif tracks[(row, col)] == '\\':
                    direction = self.DIRECTIONS.index('<v>^'[direction])
                elif tracks[(row, col)] == '+':
                    if intersection_count % 3 == 0 :
                        direction = (direction - 1) % len(self.DIRECTIONS)
                    elif intersection_count % 3 == 2:
                        direction = (direction + 1) % len(self.DIRECTIONS)
                    intersection_count += 1
                next_carts[(row, col)] = (direction, intersection_count)
            carts = next_carts
        result = first_crash_location
        return result
    
    def solve2(self, tracks, carts):
        positions = set(carts.keys())
        while len(carts) > 1:
            next_carts = {}
            for (row, col) in sorted(carts.keys()):
                if (row, col) not in positions:
                    continue
                positions.remove((row, col))
                (direction, intersection_count) = carts[(row, col)]
                offset = self.OFFSETS[self.DIRECTIONS[direction]]
                row += offset[0]
                col += offset[1]
                if (row, col) in positions:
                    positions.remove((row, col))
                else:
                    positions.add((row, col))
                if len(positions) < 2:
                    break
                if tracks[(row, col)] == '/':
                    direction = self.DIRECTIONS.index('>^<v'[direction])
                elif tracks[(row, col)] == '\\':
                    direction = self.DIRECTIONS.index('<v>^'[direction])
                elif tracks[(row, col)] == '+':
                    if intersection_count % 3 == 0 :
                        direction = (direction - 1) % len(self.DIRECTIONS)
                    elif intersection_count % 3 == 2:
                        direction = (direction + 1) % len(self.DIRECTIONS)
                    intersection_count += 1
                next_carts[(row, col)] = (direction, intersection_count)
            else:
                carts = {}
                for cart in next_carts:
                    if cart in positions:
                        carts[cart] = next_carts[cart]
        result = tuple(reversed(positions.pop()))
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        (tracks, carts) = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(copy.deepcopy(tracks), copy.deepcopy(carts)),
            self.solve2(copy.deepcopy(tracks), copy.deepcopy(carts)),
            )
        result = solutions
        return result

class Day12: # Subterranean Sustainability
    '''
    Subterranean Sustainability
    https://adventofcode.com/2018/day/12
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        pots = set()
        for pot_id, char in enumerate(raw_input_lines[0].split(': ')[1]):
            if char == '#':
                pots.add(pot_id)
        rules = {}
        for raw_input_line in raw_input_lines[2:]:
            parts = raw_input_line.split(' => ')
            rules[parts[0]] = parts[1]
        # Assume that a plant cannot be created in the middle of nowhere
        assert '.....' not in rules or rules['.....'] == '.'
        result = pots, rules
        return result
    
    def show_pots(self, pots, left, right):
        row_data = []
        for pot_id in range(left, right + 1):
            pot = '.'
            if pot_id in pots:
                pot = '#'
            row_data.append(pot)
        result = ''.join(row_data)
        return result
    
    def solve(self, pots, rules):
        for _ in range(20):
            next_pots = set()
            left = min(pots) - 2
            right = max(pots) + 2
            for center in range(left, right + 1):
                chars = ['.', '.', '.', '.', '.']
                for offset in (-2, -1, 0, 1, 2):
                    if center + offset in pots:
                        chars[2 + offset] = '#'
                key = ''.join(chars)
                if key in rules and rules[key] == '#':
                    next_pots.add(center)
            pots = next_pots
        result = sum(pots)
        return result
    
    def solve2(self, pots, rules, target_generation: int=50_000_000_000):
        generation = 0
        delta = 0
        while generation < target_generation:
            next_pots = set()
            left = min(pots) - 2
            right = max(pots) + 2
            for center in range(left, right + 1):
                chars = ['.', '.', '.', '.', '.']
                for offset in (-2, -1, 0, 1, 2):
                    if center + offset in pots:
                        chars[2 + offset] = '#'
                key = ''.join(chars)
                if key in rules and rules[key] == '#':
                    next_pots.add(center)
            next_delta = sum(next_pots) - sum(pots)
            if generation > 1_000 and delta == next_delta:
                break
            pots = next_pots
            delta = next_delta
            generation += 1
        result = sum(pots) + (target_generation - generation) * delta
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        (pots, rules) = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(set(pots), rules),
            self.solve2(set(pots), rules),
            )
        result = solutions
        return result

class Day11: # Chronal Charge
    '''
    Chronal Charge
    https://adventofcode.com/2018/day/11
    '''
    def get_grid_serial_number(self, raw_input_lines: List[str]):
        result = int(raw_input_lines[0])
        return result
    
    def get_grid(self, grid_serial_number, size=300):
        grid = collections.defaultdict(int)
        for row in range(1, size + 1):
            for col in range(1, size + 1):
                rack_id = col + 10
                power_level = rack_id * row
                power_level += grid_serial_number
                power_level *= rack_id
                power_level = (power_level % 1_000) // 100
                power_level -= 5
                assert -5 <= power_level <= 4
                grid[(row, col)] = power_level
                grid[(row, col)] += grid[(row - 1, col)]
                grid[(row, col)] += grid[(row, col - 1)]
                grid[(row, col)] -= grid[(row - 1, col - 1)]
        result = grid
        return result
    
    def show_grid(self, grid, center, radius):
        for row in range(center[0] - radius, center[0] + radius + 1):
            row_data = []
            for col in range(center[1] - radius, center[1] + radius + 1):
                cell = ' 0'
                if (row, col) in grid:
                    val = grid[(row, col)]
                    val -= grid[(row - 1, col)]
                    val -= grid[(row, col - 1)]
                    val += grid[(row - 1, col - 1)]
                    if val < 0:
                        cell = str(val)
                    else:
                        cell = '+' + str(val)
                row_data.append(cell)
            print(' '.join(row_data))
    
    def solve(self, grid_serial_number):
        grid = self.get_grid(grid_serial_number, 300)
        best_coordinate = [0, 0]
        max_power_level = float('-inf')
        for row in range(1, 300 + 1 - 2):
            for col in range(1, 300 + 1 - 2):
                power_level = grid[(row + 2, col + 2)]
                power_level -= grid[(row - 1, col + 2)]
                power_level -= grid[(row + 2, col - 1)]
                power_level += grid[(row - 1, col - 1)]
                if power_level > max_power_level:
                    max_power_level = power_level
                    best_coordinate = (col, row)
        result = ','.join(map(str, best_coordinate))
        return result
    
    def solve2(self, grid_serial_number):
        grid_size = 300
        grid = self.get_grid(grid_serial_number, grid_size)
        best_coordinate = [0, 0, 0]
        max_power_level = float('-inf')
        for size in range(1, grid_size + 1):
            for row in range(1, grid_size + 2 - size):
                for col in range(1, grid_size + 2 - size):
                    power_level = grid[(row + size - 1, col + size - 1)]
                    power_level -= grid[(row - 1, col + size - 1)]
                    power_level -= grid[(row + size - 1, col - 1)]
                    power_level += grid[(row - 1, col - 1)]
                    if power_level > max_power_level:
                        max_power_level = power_level
                        best_coordinate = (col, row, size)
        result = ','.join(map(str, best_coordinate))
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        grid_serial_number = self.get_grid_serial_number(raw_input_lines)
        solutions = (
            self.solve(grid_serial_number),
            self.solve2(grid_serial_number),
            )
        result = solutions
        return result

class Day10: # The Stars Align
    '''
    The Stars Align
    https://adventofcode.com/2018/day/10
    '''
    def get_stars(self, raw_input_lines: List[str]):
        stars = []
        for raw_input_line in raw_input_lines:
            parts = raw_input_line.split('> velocity=<')
            a = parts[0].split('<')[1].split(', ')
            b = parts[1][:-1].split(', ')
            pos_x, pos_y = tuple(map(int, a))
            vel_x, vel_y = tuple(map(int, b))
            stars.append((pos_x, pos_y, vel_x, vel_y))
        result = stars
        return result
    
    def wait_for_message(self, stars):
        min_pos_y = float('inf')
        max_pos_y = float('-inf')
        min_pos_x = float('inf')
        max_pos_x = float('-inf')
        wait_time = 0
        while True:
            wait_time += 1
            min_pos_y = float('inf')
            max_pos_y = float('-inf')
            min_pos_x = float('inf')
            max_pos_x = float('-inf')
            for i in range(len(stars)):
                vx = stars[i][2]
                vy = stars[i][3]
                x = stars[i][0] + vx
                y = stars[i][1] + vy
                stars[i] = (x, y, vx, vy)
                min_pos_y = min(min_pos_y, y)
                max_pos_y = max(max_pos_y, y)
                min_pos_x = min(min_pos_x, x)
                max_pos_x = max(max_pos_x, x)
            height = max_pos_y - min_pos_y
            if height <= 10:
                break
        cells = set()
        for pos_x, pos_y, _, _ in stars:
            cells.add((pos_x, pos_y))
        display = []
        for y in range(min_pos_y, max_pos_y + 1, 1):
            row_data = []
            for x in range(min_pos_x, max_pos_x + 1, 1):
                cell = '.'
                if (x, y) in cells:
                    cell = '#'
                row_data.append(cell)
            display.append(''.join(row_data))
        message = '\n' + '\n'.join(display)
        result = (message, wait_time)
        return result
    
    def solve(self, stars):
        (message, _) = self.wait_for_message(stars)
        result = message
        return result
    
    def solve2(self, stars):
        (_, wait_time) = self.wait_for_message(stars)
        result = wait_time
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        stars = self.get_stars(raw_input_lines)
        solutions = (
            self.solve(stars[:]),
            self.solve2(stars[:]),
            )
        result = solutions
        return result

class Day09: # Marble Mania
    '''
    Marble Mania
    https://adventofcode.com/2018/day/9
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        parts = raw_input_lines[0].split(' ')
        player_count = int(parts[0])
        marble_count = int(parts[6])
        result = (player_count, marble_count)
        return result
    
    def solve(self, player_count, marble_count):
        scores = [0] * player_count
        marbles = collections.deque()
        marbles.append(0)
        player_id = 0
        for marble in range(1, marble_count + 1):
            if marble % 23 == 0:
                scores[player_id] += marble
                marbles.rotate(7)
                captured = marbles.popleft()
                scores[player_id] += captured
            else:
                marbles.rotate(-2)
                marbles.appendleft(marble)
            player_id = (player_id + 1) % player_count
        result = max(scores)
        return result
    
    def solve2(self, player_count, marble_count):
        result = self.solve(player_count, 100 * marble_count)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        player_count, marble_count = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(player_count, marble_count),
            self.solve2(player_count, marble_count),
            )
        result = solutions
        return result

class Day08: # Memory Maneuver
    '''
    Memory Maneuver
    https://adventofcode.com/2018/day/8
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = collections.deque(map(int, raw_input_lines[0].split(' ')))
        return result
    
    def solve(self, nums):
        child_node_count = nums.popleft()
        metadata_entry_count = nums.popleft()
        metadata_entry_sum = 0
        for _ in range(child_node_count):
            metadata_entry_sum += self.solve(nums)
        for _ in range(metadata_entry_count):
            metadata_entry_sum += nums.popleft()
        result = metadata_entry_sum
        return result
    
    def solve2(self, nums):
        child_node_count = nums.popleft()
        metadata_entry_count = nums.popleft()
        values = [
            self.solve2(nums) for _ in
            range(child_node_count)
            ]
        metadata_entries = [
            nums.popleft() for _ in
            range(metadata_entry_count)
            ]
        total_value = 0
        if child_node_count == 0:
            total_value = sum(metadata_entries)
        else:
            total_value = sum(
                values[i - 1] for 
                i in metadata_entries if 
                i - 1 in range(child_node_count)
                )
        result = total_value
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        parsed_input = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(copy.deepcopy(parsed_input)),
            self.solve2(copy.deepcopy(parsed_input)),
            )
        result = solutions
        return result

class Day07: # The Sum of Its Parts
    '''
    The Sum of Its Parts
    https://adventofcode.com/2018/day/7
    '''
    def get_dependencies(self, raw_input_lines: List[str]):
        dependencies = {}
        for raw_input_line in raw_input_lines:
            a = raw_input_line[36]
            b = raw_input_line[5]
            if a not in dependencies:
                dependencies[a] = set()
            if b not in dependencies:
                dependencies[b] = set()
            dependencies[a].add(b)
        result = dependencies
        return result
    
    def solve(self, dependencies):
        finished_steps = set()
        instructions = []
        while len(instructions) < len(dependencies):
            next_step = min(
                step for step, required_steps in
                dependencies.items() if
                len(required_steps) == 0 and
                step not in finished_steps
            )
            for instruction in dependencies:
                if next_step in dependencies[instruction]:
                    dependencies[instruction].remove(next_step)
            instructions.append(next_step)
            finished_steps.add(next_step)
        result = ''.join(instructions)
        return result
    
    def solve2(self, dependencies, worker_count: int=5, min_cost: int=60):
        tasks = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        costs = {
            task: min_cost + tasks.index(task) + 1 for task in tasks
            }
        workers = [(0, None)] * worker_count
        finished_steps = set()
        time = 0
        instructions = []
        while len(finished_steps) < len(dependencies):
            for worker_id in range(len(workers)):
                if workers[worker_id][0] > time:
                    continue
                try:
                    next_step = next(iter(sorted(
                        step for step, required_steps in
                        dependencies.items() if
                        len(required_steps) == 0 and
                        step not in finished_steps and
                        step not in instructions
                    )))
                    instructions.append(next_step)
                    workers[worker_id] = (time + costs[next_step], next_step)
                except StopIteration:
                    pass
            time += 1
            for worker_id in range(len(workers)):
                work_end, step = workers[worker_id]
                if time >= work_end and step is not None:
                    finished_steps.add(step)
                    for instruction in dependencies:
                        if step in dependencies[instruction]:
                            dependencies[instruction].remove(step)
        result = time
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        dependencies = self.get_dependencies(raw_input_lines)
        solutions = (
            self.solve(copy.deepcopy(dependencies)),
            self.solve2(copy.deepcopy(dependencies)),
            )
        result = solutions
        return result

class Day06: # Chronal Coordinates
    '''
    Chronal Coordinates
    https://adventofcode.com/2018/day/6
    '''
    def get_coordinates(self, raw_input_lines: List[str]):
        coordinates = []
        for raw_input_line in raw_input_lines:
            coordinates.append(tuple(map(int, raw_input_line.split(', '))))
        result = coordinates
        return result
    
    def print_grid(self, bounds, grid, tied):
        (left, right, top, bottom) = bounds
        for y in range(top, bottom + 1):
            row_data = []
            for x in range(left, right + 1):
                cell = '.'
                if (x, y) in tied:
                    cell = '_'
                elif (x, y) in grid:
                    cell = str(grid[(x, y)][1])
                row_data.append(cell)
            print(''.join(row_data))

    def solve(self, coordinates):
        left = float('inf')
        right = float('-inf')
        top = float('inf')
        bottom = float('-inf')
        for x, y in coordinates:
            left = min(left, x)
            right = max(right, x)
            top = min(top, y)
            bottom = max(bottom, y)
        # cells that are closest to more than one coordinate are tied
        tied = set()
        grid = {}
        work = collections.deque()
        for i, coordinate in enumerate(coordinates):
            work.append((0, i, coordinate))
        # Use BFS to fill the grid
        while len(work) > 0:
            step_count = len(work)
            for _ in range(step_count):
                (distance, index, coordinate) = work.pop()
                if coordinate in grid:
                    other_distance = grid[coordinate][0]
                    other_index = grid[coordinate][1]
                    if index != other_index and distance == other_distance:
                        tied.add(coordinate)
                    continue
                grid[coordinate] = (distance, index)
                for (x, y) in (
                    (coordinate[0] + 1, coordinate[1]    ),
                    (coordinate[0] - 1, coordinate[1]    ),
                    (coordinate[0]    , coordinate[1] + 1),
                    (coordinate[0]    , coordinate[1] - 1),
                    ):
                    if (
                        x < left or
                        x > right or
                        y < top or
                        y > bottom
                    ):
                        continue
                    work.appendleft((distance + 1, index, (x, y)))
        infinite = set()
        area = collections.defaultdict(int)
        for (x, y), (distance, index) in grid.items():
            if (x, y) not in tied:
                area[index] += 1
                # cells along the border that aren't tied
                # are guaranteed to go on infinitely
                if (
                    x <= left or
                    x >= right or
                    y <= top or
                    y >= bottom
                    ):
                    infinite.add(index)
        for index in infinite:
            del area[index]
        # self.print_grid((left, right, top, bottom), grid, tied)
        result = max(area.values())
        return result
    
    def solve2(self, coordinates, max_distance: int=10_000):
        left = float('inf')
        right = float('-inf')
        top = float('inf')
        bottom = float('-inf')
        for x, y in coordinates:
            left = min(left, x)
            right = max(right, x)
            top = min(top, y)
            bottom = max(bottom, y)
        N = len(coordinates)
        # Calculate distances horizontally
        cols = collections.defaultdict(int)
        for col in range(
            left - (1 + max_distance // N),
            right + (1 + max_distance // N),
            ):
            for coordinate in coordinates:
                cols[col] += abs(coordinate[0] - col)
        # Calculate distances vertically
        rows = collections.defaultdict(int)
        for row in range(
            top - (1 + max_distance // N),
            bottom + (1 + max_distance // N),
            ):
            for coordinate in coordinates:
                rows[row] += abs(coordinate[1] - row)
        # Combine horizontal and vertical distances to find cells
        # within the target distance
        count = 0
        for row, row_distance in rows.items():
            for col, col_distance in cols.items():
                if row_distance + col_distance < max_distance:
                    count += 1
        result = count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        coordinates = self.get_coordinates(raw_input_lines)
        solutions = (
            self.solve(coordinates),
            self.solve2(coordinates, 10_000),
            )
        result = solutions
        return result

class Day05: # Alchemical Reduction
    '''
    Alchemical Reduction
    https://adventofcode.com/2018/day/5
    '''
    def get_polymer(self, raw_input_lines: List[str]):
        polymer = list(raw_input_lines[0])
        result = polymer
        return result
    
    def get_fully_reacted_polymer(self, polymer):
        reaction_ind = True
        prev_polymer = polymer
        while reaction_ind:
            reaction_ind = False
            polymer = prev_polymer[:]
            i = len(polymer) - 1
            while i > 0:
                if (
                    prev_polymer[i] != prev_polymer[i - 1] and
                    prev_polymer[i].upper() == prev_polymer[i - 1].upper()
                    ):
                    polymer.pop(i)
                    polymer.pop(i - 1)
                    i -= 1
                    reaction_ind = True
                i -= 1
            prev_polymer = polymer
        result = polymer
        return result
    
    def solve(self, polymer):
        fully_reacted_polymer = self.get_fully_reacted_polymer(polymer)
        result = len(fully_reacted_polymer)
        return result
    
    def solve2(self, initial_polymer):
        shortest_polymer = initial_polymer
        for char in 'abcdefghijklmnopqrstuvwxyz':
            polymer = []
            for unit in initial_polymer:
                if unit == char or unit.lower() == char:
                    continue
                else:
                    polymer.append(unit)
            fully_reacted_polymer = self.get_fully_reacted_polymer(polymer)
            if len(fully_reacted_polymer) < len(shortest_polymer):
                shortest_polymer = fully_reacted_polymer
        result = len(shortest_polymer)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        polymer = self.get_polymer(raw_input_lines)
        solutions = (
            self.solve(polymer),
            self.solve2(polymer),
            )
        result = solutions
        return result

class Day04: # Repose Record
    '''
    Repose Record
    https://adventofcode.com/2018/day/4
    '''
    def get_sleep_times(self, raw_input_lines: List[str]):
        raw_records = []
        for raw_input_line in raw_input_lines:
            (a, b) = raw_input_line.split('] ')
            datetime_str = a[1:]
            year = int(datetime_str[:4])
            month = int(datetime_str[5:7])
            day = int(datetime_str[8:10])
            hour = int(datetime_str[11:13])
            minute = int(datetime_str[14:16])
            ordinal_day = datetime.date(year, month, day).toordinal()
            if hour == 23:
                ordinal_day += 1
                minute = 0
            raw_records.append((ordinal_day, minute, b))
        raw_records.sort()
        sleep_times = set()
        guard_id = -1
        for i in range(len(raw_records) - 1):
            day0, minute0, event0 = raw_records[i]
            _, minute1, event1 = raw_records[i + 1]
            if '#' in event0:
                guard_id = int(event0.split(' ')[1][1:])
            elif event0 == 'falls asleep':
                start = minute0
                end = minute1
                if '#' in event1:
                    end = 60
                for minute_id in range(start, end):
                    sleep_times.add((day0, minute_id, guard_id))
        result = sleep_times
        return result
    
    def solve(self, sleep_times):
        guards = collections.defaultdict(int)
        for _, minute_id, guard_id in sorted(sleep_times):
            guards[guard_id] += 1
        best_guard_id = max(guards, key=guards.get)
        minutes = collections.defaultdict(int)
        for _, minute_id, guard_id in sleep_times:
            if guard_id == best_guard_id:
                minutes[minute_id] += 1
        best_minute_id = max(minutes, key=minutes.get)
        result = best_guard_id * best_minute_id
        return result
    
    def solve2(self, sleep_times):
        guards = collections.defaultdict(int)
        for _, minute_id, guard_id in sorted(sleep_times):
            guards[(guard_id, minute_id)] += 1
        best_guard_id, best_minute_id = max(guards, key=guards.get)
        result = best_guard_id * best_minute_id
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        sleep_times = self.get_sleep_times(raw_input_lines)
        solutions = (
            self.solve(sleep_times),
            self.solve2(sleep_times),
            )
        result = solutions
        return result

class Day03: # No Matter How You Slice It
    '''
    No Matter How You Slice It
    https://adventofcode.com/2018/day/3
    '''
    def get_claims(self, raw_input_lines: List[str]):
        claims = {}
        for raw_input_line in raw_input_lines:
            parts = raw_input_line.split(' ')
            claim_id = int(parts[0][1:])
            left, top = tuple(map(int, parts[2][:-1].split(',')))
            width, height = tuple(map(int, parts[3].split('x')))
            claims[claim_id] = (left, top, width, height)
        result = claims
        return result
    
    def solve(self, claims):
        claimed_fabric = collections.defaultdict(set)
        for claim_id, (left, top, width, height) in claims.items():
            for row in range(top, top + height):
                for col in range(left, left + width):
                    claimed_fabric[(row, col)].add(claim_id)
        result = sum(
            1 for
            _, claims in
            claimed_fabric.items() if
            len(claims) >= 2
            )
        return result
    
    def solve2(self, claims):
        claimed_fabric = collections.defaultdict(set)
        for claim_id, (left, top, width, height) in claims.items():
            for row in range(top, top + height):
                for col in range(left, left + width):
                    claimed_fabric[(row, col)].add(claim_id)
        result = -1
        for claim_id, (left, top, width, height) in claims.items():
            overlap_ind = False
            for row in range(top, top + height):
                for col in range(left, left + width):
                    claims = claimed_fabric[(row, col)]
                    if len(claims) > 1:
                        overlap_ind = True
                        break
                if overlap_ind:
                    break
            if not overlap_ind:
                result = claim_id
                break
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        claims = self.get_claims(raw_input_lines)
        solutions = (
            self.solve(claims),
            self.solve2(claims),
            )
        result = solutions
        return result

class Day02: # Inventory Management System
    '''
    Inventory Management System
    https://adventofcode.com/2018/day/2
    '''
    def get_box_ids(self, raw_input_lines: List[str]):
        box_ids = []
        for raw_input_line in raw_input_lines:
            box_ids.append(raw_input_line)
        result = box_ids
        return result
    
    def solve(self, box_ids):
        counts = collections.defaultdict(set)
        for box_id in box_ids:
            chars = collections.Counter(box_id)
            for count in chars.values():
                counts[count].add(box_id)
        result = len(counts[2]) * len(counts[3])
        return result
    
    def solve2(self, box_ids):
        for i in range(len(box_ids)):
            box_id_1 = box_ids[i]
            for j in range(i + 1, len(box_ids)):
                box_id_2 = box_ids[j]
                idx = -1
                for k in range(len(box_id_1)):
                    if box_id_1[k] != box_id_2[k]:
                        if idx >= 0:
                            break
                        idx = k
                else:
                    if idx >= 0:
                        result = box_id_1[:idx] + box_id_1[idx + 1:]
                        return result
        return 'Solution not found!'
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        box_ids = self.get_box_ids(raw_input_lines)
        solutions = (
            self.solve(box_ids),
            self.solve2(box_ids),
            )
        result = solutions
        return result

class Day01: # Chronal Calibration
    '''
    Chronal Calibration
    https://adventofcode.com/2018/day/1
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(int(raw_input_line))
        return result
    
    def solve(self, parsed_input):
        result = sum(parsed_input)
        return result
    
    def solve2(self, parsed_input):
        seen = set()
        frequency = 0
        seen.add(frequency)
        result = None
        while result is None:
            for change in parsed_input:
                frequency += change
                if frequency in seen:
                    result = frequency
                    break
                seen.add(frequency)
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

if __name__ == '__main__':
    '''
    Usage
    python AdventOfCode2018.py 23 < inputs/2018day23.in
    '''
    solvers = {
        1: (Day01, 'Chronal Calibration'),
        2: (Day02, 'Inventory Management System'),
        3: (Day03, 'No Matter How You Slice It'),
        4: (Day04, 'Repose Record'),
        5: (Day05, 'Alchemical Reduction'),
        6: (Day06, 'Chronal Coordinates'),
        7: (Day07, 'The Sum of Its Parts'),
        8: (Day08, 'Memory Maneuver'),
        9: (Day09, 'Marble Mania'),
       10: (Day10, 'The Stars Align'),
       11: (Day11, 'Chronal Charge'),
       12: (Day12, 'Subterranean Sustainability'),
       13: (Day13, 'Mine Cart Madness'),
       14: (Day14, 'Chocolate Charts'),
       15: (Day15, 'Beverage Bandits'),
       16: (Day16, 'Chronal Classification'),
       17: (Day17, 'Reservoir Research'),
       18: (Day18, 'Settlers of The North Pole'),
       19: (Day19, 'Go With The Flow'),
       20: (Day20, 'A Regular Map'),
       21: (Day21, 'Chronal Conversion'),
       22: (Day22, 'Mode Maze'),
       23: (Day23, 'Experimental Emergency Teleportation'),
       24: (Day24, 'Immune System Simulator 20XX'),
       25: (Day25, 'Four-Dimensional Adventure'),
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
