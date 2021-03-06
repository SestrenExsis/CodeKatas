'''
Created 2021-04-04

@author: Sestren
'''
import argparse
import collections
import copy
import functools
import heapq
import hashlib
import itertools
import json
import random
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

class WizardSim:
    '''
    If at any point, the boss has less than 1 HP, the player wins
    If at any point, the hero has less than 1 HP, the player loses
    A round consists of the following, in order:
    Player's turn:
        1) Resolve ongoing spell effects at start of player's turn
        2) Update spell effect timers at start of player's turn
        3) Player casts a spell
    Boss's turn:
        4) Resolve ongoing spell effects at start of boss's turn
        5) Update spell effect timers at start of boss's turn
        6) Boss attacks player
    '''
    def __init__(self, hero_hp: int, hero_mana: int, boss_hp: int, boss_damage: int):
        self.time = 0
        self.total_mana_spent = 0
        self.hero = self.Hero(hero_hp, hero_mana)
        self.boss = self.Boss(boss_hp, boss_damage)
        self.active_spells = set()
        self.spells = []
        self.logs = ['\n---NEW SIM---']
        self.debug = False
        self.min_mana_to_win = None

    def __lt__(self, other):
        result = self.total_mana_spent < other.total_mana_spent
        return result
    
    def log(self, message: str):
        if self.debug:
            self.logs.append(message)

    class Hero:
        def __init__(self, hp: int, mana: int):
            self.hp = hp
            self.mana = mana
            self.defense = 0
            self.cursed = False
    
    class Boss:
        def __init__(self, hp: int, damage: int):
            self.hp = hp
            self.damage = damage
    
    class Spell:
        def __init__(self, name: str, mana_cost: int, effect: str, power: int, duration: int):
            self.name = name
            self.mana_cost = mana_cost
            self.effect = effect
            self.power = power
            self.duration = duration

        def __lt__(self, other):
            result = self.mana_cost < other.mana_cost
            return result
        
        def apply(self, hero, boss):
            if self.effect == 'hurt':
                boss.hp -= self.power
            elif self.effect == 'drain':
                boss.hp -= self.power
                hero.hp += self.power
            elif self.effect == 'armor':
                hero.defense = self.power
            elif self.effect == 'mana':
                hero.mana += self.power
    
    def tick(self):
        self.log('- Player has {} hit point{}, {} armor, {} mana'.format(
            self.hero.hp,
            '' if self.hero.hp == 1 else 's',
            self.hero.defense,
            self.hero.mana,
        ))
        self.log('- Boss has {} hit point{}'.format(
            self.boss.hp,
            '' if self.boss.hp == 1 else 's',
        ))
        self.time += 1
        for _, spell in self.spells:
            spell.apply(self.hero, self.boss)
        if self.min_mana_to_win is None and self.boss.hp < 1:
            self.min_mana_to_win = self.total_mana_spent
        while len(self.spells) > 0 and self.spells[0][0] <= self.time:
            _, expired_spell = heapq.heappop(self.spells)
            if expired_spell.duration > 1:
                self.log('{} wears off.'.format(expired_spell.name))
                if expired_spell.effect == 'armor':
                    self.hero.defense = 0
                assert expired_spell.name in self.active_spells
                self.active_spells.remove(expired_spell.name)
    
    def hero_turn(self, chosen_spell):
        self.log('\n-- Player turn -- {}'.format(self.total_mana_spent))
        if self.hero.cursed:
            self.hero.hp -= 1
        self.tick()
        if (
            self.min_mana_to_win is None and self.hero.hp < 1 or
            chosen_spell.name in self.active_spells
        ):
            self.min_mana_to_win = float('inf')
        if chosen_spell.duration == 0:
            chosen_spell.apply(self.hero, self.boss)
        else:
            heapq.heappush(self.spells, (self.time + chosen_spell.duration, chosen_spell))
            self.active_spells.add(chosen_spell.name)
        self.hero.mana -= chosen_spell.mana_cost
        if self.min_mana_to_win is None and self.hero.mana < 0:
            self.min_mana_to_win = float('inf')
        self.total_mana_spent += chosen_spell.mana_cost
        self.log('Player casts {}.'.format(chosen_spell.name))

    def boss_turn(self):
        self.log('\n-- Boss turn -- {}'.format(self.total_mana_spent))
        self.tick()
        dmg = max(1, self.boss.damage - self.hero.defense)
        self.hero.hp -= dmg
        if self.min_mana_to_win is None and self.hero.hp < 1:
            self.min_mana_to_win = float('inf')
        self.log('Boss attacks for {} damage'.format(dmg))

class Template: # Template
    '''
    Template
    https://adventofcode.com/2015/day/?
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

class Day25: # Let It Snow
    '''
    Let It Snow
    https://adventofcode.com/2015/day/25
    '''
    def get_coordinate(self, raw_input_lines: List[str]):
        row = None
        col = None
        tokens = raw_input_lines[0].split(' ')
        for token in tokens:
            if token[:-1].isdigit():
                if row is None:
                    row = int(token[:-1])
                else:
                    col = int(token[:-1])
        result = (row, col)
        return result
    
    def solve(self, target_row, target_col):
        SEED = 20151125
        MULTIPLIER = 252533
        MODULO = 33554393
        num = SEED
        nums = {}
        row = 1
        while (target_row, target_col) not in nums:
            for offset in range(row):
                nums[(row - offset, offset + 1)] = num
                num = (MULTIPLIER * num) % MODULO
            row += 1
        result = nums[(target_row, target_col)]
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        coordinate = self.get_coordinate(raw_input_lines)
        solutions = (
            self.solve(*coordinate),
            'Snow begins to fall.',
            )
        result = solutions
        return result

class Day24: # It Hangs in the Balance
    '''
    It Hangs in the Balance
    https://adventofcode.com/2015/day/24
    * Split into three groups
    * All three groups must weigh the same
    * Group 1 has fewest number of packages possible given above constraints
    * Group 1 has smallest product of weights possible given above constraints
    '''
    def get_weights(self, raw_input_lines: List[str]):
        weights = []
        for raw_input_line in raw_input_lines:
            weights.append(int(raw_input_line))
        result = weights
        return result
    
    def solve(self, weights):
        total_weight = sum(weights)
        group_weight = total_weight // 3
        assert group_weight == total_weight / 3
        min_quantum_entanglement = float('inf')
        for first_group_len in range(1, 8):
            for combo in itertools.combinations(weights, first_group_len):
                # NOTE: We are assuming that the rest can be split into two even groups
                # TODO: Verify if remainder can be split into two even groups
                if sum(combo) == total_weight / 3:
                    min_quantum_entanglement = min(
                        min_quantum_entanglement,
                        functools.reduce(lambda a, b: a * b, list(combo)),
                    )
        result = min_quantum_entanglement
        return result
    
    def solve2(self, weights):
        total_weight = sum(weights)
        group_weight = total_weight // 4
        assert group_weight == total_weight / 4
        min_quantum_entanglement = float('inf')
        for first_group_len in range(1, 8):
            for combo in itertools.combinations(weights, first_group_len):
                # NOTE: We are assuming that the rest can be split into two even groups
                # TODO: Verify if remainder can be split into three even groups
                if sum(combo) == total_weight / 4:
                    min_quantum_entanglement = min(
                        min_quantum_entanglement,
                        functools.reduce(lambda a, b: a * b, list(combo)),
                    )
        result = min_quantum_entanglement
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        weights = self.get_weights(raw_input_lines)
        solutions = (
            self.solve(weights),
            self.solve2(weights),
            )
        result = solutions
        return result

class Day23: # Opening the Turing Lock
    '''
    Opening the Turing Lock
    https://adventofcode.com/2015/day/23
    '''
    def get_program(self, raw_input_lines: List[str]):
        program = []
        for raw_input_line in raw_input_lines:
            instruction = raw_input_line[:3]
            raw_parameters = raw_input_line[4:]
            parameters = list(raw_parameters.split(', '))
            for i in range(len(parameters)):
                if parameters[i] not in ('a', 'b'):
                    parameters[i] = int(parameters[i])
            program.append((instruction, tuple(parameters)))
        result = program
        return result

    def run(self, pc, registers, program):
        while True:
            if not (0 <= pc < len(program)):
                break
            instruction, params = program[pc]
            if instruction == 'hlf':
                registers[params[0]] = registers[params[0]] // 2
                pc += 1
            elif instruction == 'tpl':
                registers[params[0]] = 3 * registers[params[0]]
                pc += 1
            elif instruction == 'inc':
                registers[params[0]] += 1
                pc += 1
            elif instruction == 'jmp':
                pc += params[0]
            elif instruction == 'jie':
                register = registers[params[0]]
                if register % 2 == 0:
                    pc += params[1]
                else:
                    pc += 1
            elif instruction == 'jio':
                register = registers[params[0]]
                if register == 1:
                    pc += params[1]
                else:
                    pc += 1
    
    def solve(self, program):
        pc = 0
        registers = {
            'a': 0,
            'b': 0,
        }
        self.run(pc, registers, program)
        result = registers['b']
        return result
    
    def solve2(self, program):
        pc = 0
        registers = {
            'a': 1,
            'b': 0,
        }
        self.run(pc, registers, program)
        result = registers['b']
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        program = self.get_program(raw_input_lines)
        solutions = (
            self.solve(program),
            self.solve2(program),
            )
        result = solutions
        return result

class Day22: # Wizard Simulator 20XX
    '''
    Wizard Simulator 20XX
    https://adventofcode.com/2015/day/22
    '''
    spellbook = {
        # spell_name: (mana_cost, effect, power, duration)
        'Magic Missile': WizardSim.Spell('Magic Missile', 53, 'hurt', 4, 0),
        'Drain': WizardSim.Spell('Drain', 73, 'drain', 2, 0),
        'Shield': WizardSim.Spell('Shield', 113, 'armor', 7, 6),
        'Poison': WizardSim.Spell('Poison', 173, 'hurt', 3, 6),
        'Recharge': WizardSim.Spell('Recharge', 229, 'mana', 101, 5),
    }

    def get_boss_stats(self, raw_input_lines: List[str]):
        boss_stats = {}
        for raw_input_line in raw_input_lines:
            stat, raw_value = raw_input_line.split(': ')
            value = int(raw_value)
            boss_stats[stat] = value
        result = boss_stats
        return result
    
    def get_min_mana_to_win(self, spellbook, sim: WizardSim) -> int:
        min_mana_to_win = float('inf')
        work = []
        for spell_name, spell in spellbook.items():
            heapq.heappush(work, (spell.mana_cost, copy.deepcopy(sim), spell_name))
        while len(work) > 0:
            total_mana_spent, sim, spell_name = heapq.heappop(work)
            sim.hero_turn(spellbook[spell_name])
            sim.boss_turn()
            if sim.min_mana_to_win is None:
                pass
            elif sim.min_mana_to_win >= float('inf'):
                continue
            else:
                sim.log('This kills the boss, and the player wins.')
                min_mana_to_win = sim.min_mana_to_win
                for entry in sim.logs:
                    print(entry)
                break
            for next_spell_name, next_spell in spellbook.items():
                mp = next_spell.mana_cost
                next_sim = copy.deepcopy(sim)
                heapq.heappush(work, (total_mana_spent + mp, next_sim, next_spell_name))
        result = min_mana_to_win
        return result
    
    def solve_slowly(self, spellbook, hero_hp, hero_mp, boss_hp, boss_dmg) -> int:
        sim = WizardSim(hero_hp, hero_mp, boss_hp, boss_dmg)
        result = self.get_min_mana_to_win(spellbook, sim)
        return result
    
    def solve_slowly2(self, spellbook, hero_hp, hero_mp, boss_hp, boss_dmg):
        sim = WizardSim(hero_hp, hero_mp, boss_hp, boss_dmg)
        sim.hero.cursed = True
        result = self.get_min_mana_to_win(spellbook, sim)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        boss_stats = self.get_boss_stats(raw_input_lines)
        boss_hp = boss_stats['Hit Points']
        boss_dmg = boss_stats['Damage']
        solutions = (
            self.solve_slowly(self.spellbook, 50, 500, boss_hp, boss_dmg),
            self.solve_slowly2(self.spellbook, 50, 500, boss_hp, boss_dmg),
            )
        result = solutions
        return result

class Day21: # RPG Simulator 20XX
    '''
    RPG Simulator 20XX
    https://adventofcode.com/2015/day/21
    '''
    def get_boss(self, raw_input_lines: List[str]):
        boss = {}
        for raw_input_line in raw_input_lines:
            stat, raw_value = raw_input_line.split(': ')
            value = int(raw_value)
            boss[stat] = value
        result = boss
        return result
    
    def fight(self, boss, hero_atk, hero_def) -> bool:
        victory = True
        hero_hp = 100
        boss_atk = boss['Damage']
        boss_def = boss['Armor']
        boss_hp = boss['Hit Points']
        while True:
            # Hero attacks
            hero_dmg = max(1, hero_atk - boss_def)
            boss_hp -= hero_dmg
            if boss_hp < 1:
                victory = True
                break
            # Boss attacks
            boss_dmg = max(1, boss_atk - hero_def)
            hero_hp -= boss_dmg
            if hero_hp < 1:
                victory = False
                break
        result = victory
        return result
    
    def solve(self, boss, weapons, armors, rings):
        min_cost_to_win = float('inf')
        for i in range(len(weapons)):
            weapon_cost, weapon_atk, _ = weapons[i]
            for j in range(len(armors)):
                armor_cost, _, armor_def = armors[j]
                for k1 in range(len(rings)):
                    ring1_cost, ring1_atk, ring1_def = rings[k1]
                    for k2 in range(k1 + 1, len(rings)):
                        ring2_cost, ring2_atk, ring2_def = rings[k2]
                        cost = sum([
                            weapon_cost,
                            armor_cost,
                            ring1_cost,
                            ring2_cost,
                        ])
                        hero_atk = weapon_atk + ring1_atk + ring2_atk
                        hero_def = armor_def + ring1_def + ring2_def
                        victory = self.fight(boss, hero_atk, hero_def)
                        if victory:
                            min_cost_to_win = min(min_cost_to_win, cost)
        result = min_cost_to_win
        return result
    
    def solve2(self, boss, weapons, armors, rings):
        max_cost_to_lose = float('-inf')
        for i in range(len(weapons)):
            weapon_cost, weapon_atk, _ = weapons[i]
            for j in range(len(armors)):
                armor_cost, _, armor_def = armors[j]
                for k1 in range(len(rings)):
                    ring1_cost, ring1_atk, ring1_def = rings[k1]
                    for k2 in range(k1 + 1, len(rings)):
                        ring2_cost, ring2_atk, ring2_def = rings[k2]
                        cost = sum([
                            weapon_cost,
                            armor_cost,
                            ring1_cost,
                            ring2_cost,
                        ])
                        hero_atk = weapon_atk + ring1_atk + ring2_atk
                        hero_def = armor_def + ring1_def + ring2_def
                        victory = self.fight(boss, hero_atk, hero_def)
                        if not victory:
                            max_cost_to_lose = max(max_cost_to_lose, cost)
        result = max_cost_to_lose
        return result
    
    def main(self):
        shop = {
            # Item: (Type, Cost, Damage, Armor)
            'Dagger': ('Weapon', 8, 4, 0),
            'Shortsword': ('Weapon', 10, 5, 0),
            'Warhammer': ('Weapon', 25, 6, 0),
            'Longsword': ('Weapon', 40, 7, 0),
            'Greataxe': ('Weapon', 74, 8, 0),
            'Leather': ('Armor', 13, 0, 1),
            'Chainmail': ('Armor', 31, 0, 2),
            'Splintmail': ('Armor', 53, 0, 3),
            'Bandedmail': ('Armor', 75, 0, 4),
            'Platemail': ('Armor', 102, 0, 5),
            'Ring of Damage +1': ('Ring', 25, 1, 0),
            'Ring of Damage +2': ('Ring', 50, 2, 0),
            'Ring of Damage +3': ('Ring', 100, 3, 0),
            'Ring of Defense +1': ('Ring', 20, 0, 1),
            'Ring of Defense +2': ('Ring', 40, 0, 2),
            'Ring of Defense +3': ('Ring', 80, 0, 3),
        }
        raw_input_lines = get_raw_input_lines()
        boss = self.get_boss(raw_input_lines)
        weapons = [
            stats[1:] for
            _, stats in shop.items() if
            stats[0] == 'Weapon'
        ]
        armors = [
            stats[1:] for
            _, stats in shop.items() if
            stats[0] == 'Armor'
        ]
        armors.append((0, 0, 0))
        rings = [
            stats[1:] for
            _, stats in shop.items() if
            stats[0] == 'Ring'
        ]
        rings.append((0, 0, 0))
        rings.append((0, 0, 0))
        # You must buy exactly 1 weapon
        # You can buy 0 or 1 armor
        # You can buy 0-2 rings
        # No duplicate items
        solutions = (
            self.solve(boss, weapons, armors, rings),
            self.solve2(boss, weapons, armors, rings),
            )
        result = solutions
        return result

class Day20: # Infinite Elves and Infinite Houses
    '''
    Infinite Elves and Infinite Houses
    https://adventofcode.com/2015/day/20

    1 -> 1 -> 1
    2 -> 1 + 2 -> 3
    3 -> 1 + 3 -> 4
    4 -> 1 + 2 + 4 -> 7
    5 -> 1 + 5 -> 6
    6 -> 1 + 2 + 3 + 6 = 12
    7 -> 1 + 7 = 8
    8 -> 1 + 2 + 4 + 8 = 15
    9 -> 1 + 3 + 9 = 13
    10 -> 1 + 2 + 5 + 10 = 18
    11 -> 1 + 11 = 12
    12 -> 1 + 2 + 3 + 4 + 6 + 12 = 28
    13 -> 1 + 13 = 14
    14 -> 1 + 2 + 7 + 14 = 24
    15 -> 1 + 3 + 5 + 15 = 24
    16 -> 1 + 2 + 4 + 8 + 16 = 31
    '''

    def get_target_present_count(self, raw_input_lines: List[str]):
        target_present_count = int(raw_input_lines[0])
        result = target_present_count
        return result

    def solve(self, target_present_count):
        target_sum_of_divisors = 1 + (target_present_count - 1) // 10
        houses = [0] * (target_sum_of_divisors + 1)
        for elf_id in range(1, target_sum_of_divisors + 1):
            for house_id in range(elf_id, target_sum_of_divisors + 1, elf_id):
                houses[house_id] += 10 * elf_id
        target_house_id = None
        for house_id, present_count in enumerate(houses):
            if present_count >= target_present_count:
                target_house_id = house_id
                break
        result = target_house_id
        return result
    
    def solve2_poorly(self, target_present_count):
        target_sum_of_divisors = 1 + (target_present_count - 1) // 10
        houses = [0] * (50 * target_sum_of_divisors + 1)
        for elf_id in range(1, target_sum_of_divisors + 1):
            for stop_id in range(50):
                house_id = (stop_id + 1) * elf_id
                houses[house_id] += 11 * elf_id
        target_house_id = None
        for house_id, present_count in enumerate(houses):
            if present_count >= target_present_count:
                target_house_id = house_id
                break
        result = target_house_id
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        target_present_count = self.get_target_present_count(raw_input_lines)
        solutions = (
            self.solve(target_present_count),
            self.solve2_poorly(target_present_count),
            )
        result = solutions
        return result

class Day19: # Medicine for Rudolph
    '''
    Medicine for Rudolph
    https://adventofcode.com/2015/day/19
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        replacements = collections.defaultdict(set)
        for raw_input_line in raw_input_lines:
            if len(raw_input_line) < 1:
                break
            source, target = raw_input_line.split(' => ')
            assert source == 'e' or 'e' not in source
            assert 'e' not in target
            components = []
            for char in target:
                if char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    components.append(char)
                else:
                    components[-1] = components[-1] + char
            replacements[source].add(tuple(components))
        assert 'e' not in raw_input_lines[-1]
        medicine_molecule = []
        for char in raw_input_lines[-1]:
            if char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                medicine_molecule.append(char)
            else:
                medicine_molecule[-1] = medicine_molecule[-1] + char
        assert ''.join(medicine_molecule) == raw_input_lines[-1]
        result = (replacements, tuple(medicine_molecule))
        return result
    
    def solve(self, replacements, medicine_molecule):
        new_molecules = set()
        for i, source in enumerate(medicine_molecule):
            a = list(medicine_molecule[:i])
            b = list(medicine_molecule[i + 1:])
            for target in replacements[source]:
                new_molecule = tuple(a + list(target) + b)
                new_molecules.add(new_molecule)
        result = len(new_molecules)
        return result
    
    def solve2_randomly(self, seed, replacements, medicine_molecule):
        # Based on https://www.reddit.com/r/adventofcode/comments/3xflz8/day_19_solutions/cy4cu5b?utm_source=share&utm_medium=web2x&context=3
        pairs = []
        for source, targets in replacements.items():
            for target in targets:
                pairs.append((source, ''.join(target)))
        min_step_count = float('inf')
        for _ in range(1_000):
            step_count = 0
            molecule = ''.join(medicine_molecule)
            random.shuffle(pairs)
            while len(molecule) > 1 or molecule[0] != seed:
                prev_molecule = molecule
                for source, target in pairs:
                    if target not in molecule:
                        continue
                    molecule = molecule.replace(target, source, 1)
                    step_count += 1
                if molecule == prev_molecule:
                    break
            if molecule == seed:
                min_step_count = min(min_step_count, step_count)
        result = min_step_count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        replacements, medicine_molecule = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(replacements, medicine_molecule),
            self.solve2_randomly('e', replacements, medicine_molecule),
            )
        result = solutions
        return result

class Day18: # Like a GIF For Your Yard
    '''
    Like a GIF For Your Yard
    https://adventofcode.com/2015/day/18
    '''
    def get_lights(self, raw_input_lines: List[str]):
        lights = {}
        for row, raw_input_line in enumerate(raw_input_lines):
            for col, char in enumerate(raw_input_line):
                lights[(row, col)] = 0 if char == '.' else 1
        result = lights
        return result
    
    def solve(self, lights, step_count):
        rows = 1 + max(key[0] for key in lights)
        cols = 1 + max(key[1] for key in lights)
        for _ in range(step_count):
            toggles = set()
            for row in range(rows):
                for col in range(cols):
                    neighbors = 0
                    for neighbor in (
                        (row - 1, col - 1),
                        (row + 0, col - 1),
                        (row + 1, col - 1),
                        (row - 1, col + 0),
                        (row + 1, col + 0),
                        (row - 1, col + 1),
                        (row + 0, col + 1),
                        (row + 1, col + 1),
                    ):
                        if neighbor in lights:
                            neighbors += lights[neighbor]
                    value = lights[(row, col)]
                    if value == 1 and (neighbors < 2 or neighbors > 3):
                        toggles.add((row, col))
                    if value == 0 and neighbors == 3:
                        toggles.add((row, col))
            for (row, col) in toggles:
                lights[(row, col)] = 1 - lights[(row, col)]
        result = sum(lights.values())
        return result
    
    def solve2(self, lights, step_count):
        rows = 1 + max(key[0] for key in lights)
        cols = 1 + max(key[1] for key in lights)
        stuck_lights = (
            (0, 0),
            (rows - 1, 0),
            (0, cols - 1),
            (rows - 1, cols - 1),
        )
        for stuck_light in stuck_lights:
            lights[stuck_light] = 1
        for _ in range(step_count):
            toggles = set()
            for row in range(rows):
                for col in range(cols):
                    if (row, col) in stuck_lights:
                        continue
                    neighbors = 0
                    for neighbor in (
                        (row - 1, col - 1),
                        (row + 0, col - 1),
                        (row + 1, col - 1),
                        (row - 1, col + 0),
                        (row + 1, col + 0),
                        (row - 1, col + 1),
                        (row + 0, col + 1),
                        (row + 1, col + 1),
                    ):
                        if neighbor in lights:
                            neighbors += lights[neighbor]
                    value = lights[(row, col)]
                    if value == 1 and (neighbors < 2 or neighbors > 3):
                        toggles.add((row, col))
                    if value == 0 and neighbors == 3:
                        toggles.add((row, col))
            for (row, col) in toggles:
                lights[(row, col)] = 1 - lights[(row, col)]
        result = sum(lights.values())
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        lights = self.get_lights(raw_input_lines)
        solutions = (
            self.solve(copy.deepcopy(lights), 100),
            self.solve2(copy.deepcopy(lights), 100),
            )
        result = solutions
        return result

class Day17: # No Such Thing as Too Much
    '''
    No Such Thing as Too Much
    https://adventofcode.com/2015/day/17
    '''
    def get_containers(self, raw_input_lines: List[str]):
        containers = []
        for raw_input_line in raw_input_lines:
            containers.append(int(raw_input_line))
        result = containers
        return result
    
    def solve(self, target_liters, containers):
        amounts = [1] + [0] * target_liters
        for container in containers:
            for i in range(target_liters, -1, -1):
                if i >= container:
                    amounts[i] += amounts[i - container]
        result = amounts[target_liters]
        return result
    
    def get_min_containers(self, target_liters, containers):
        amounts = collections.defaultdict(int)
        amounts[0] = 0
        for container in containers:
            for amount in list(amounts.keys()):
                new_amount = amount + container
                if new_amount not in amounts:
                    amounts[new_amount] = amounts[amount] + 1
                else:
                    amounts[new_amount] = min(
                        amounts[new_amount],
                        amounts[amount] + 1,
                    )
        min_containers = amounts[target_liters]
        result = min_containers
        return result
    
    def solve2(self, target_liters, containers):
        min_containers = self.get_min_containers(target_liters, containers)
        amounts = collections.defaultdict(list)
        amounts[0] = [[]]
        for i, container in enumerate(containers):
            for amount in list(amounts.keys()):
                new_amount = amount + container
                if new_amount not in amounts:
                    amounts[new_amount] = []
                    if len(amounts[amount]) < 1:
                        amounts[new_amount].append([i])
                    else:
                        for method in amounts[amount]:
                            amounts[new_amount].append(method + [i])
                else:
                    old_method_length = len(amounts[new_amount][0])
                    new_method_length = len(amounts[amount][0]) + 1
                    if new_method_length < old_method_length:
                        amounts[new_amount] = []
                        for method in amounts[amount]:
                            amounts[new_amount].append(method + [i])
                    elif new_method_length == old_method_length:
                        for method in amounts[amount]:
                            amounts[new_amount].append(method + [i])
        method_count = 0
        for method in amounts[target_liters]:
            if len(set(method)) == len(method):
                method_count += 1
        result = method_count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        containers = self.get_containers(raw_input_lines)
        solutions = (
            self.solve(150, containers),
            self.solve2(150, containers),
            )
        result = solutions
        return result

class Day16: # Aunt Sue
    '''
    Aunt Sue
    https://adventofcode.com/2015/day/16
    '''
    def get_sue_data(self, raw_input_lines: List[str]):
        sue_data = {}
        for raw_input_line in raw_input_lines:
            partition = raw_input_line.index(':')
            a = raw_input_line[:partition]
            b = raw_input_line[partition + 2:]
            sue_id = int(a.split(' ')[1])
            parts = b.split(', ')
            sue_data[sue_id] = {}
            for part in parts:
                key, val = part.split(': ')
                sue_data[sue_id][key] = int(val)
        result = sue_data
        return result
    
    def solve(self, sue_data, analysis):
        detected_sue_id = None
        for sue_id, data in sue_data.items():
            for key, val in analysis.items():
                if key not in data:
                    continue
                if val != data[key]:
                    break
            else:
                detected_sue_id = sue_id
                break
        result = detected_sue_id
        return result
    
    def solve2(self, sue_data, analysis, ranges):
        detected_sue_id = None
        for sue_id, data in sue_data.items():
            for key, analyzed_value in analysis.items():
                if key not in data:
                    continue
                if key not in ranges:
                    if analyzed_value != data[key]:
                        break
                elif ranges[key] == '<':
                    if data[key] >= analyzed_value:
                        break
                elif ranges[key] == '>':
                    if data[key] <= analyzed_value:
                        break
            else:
                detected_sue_id = sue_id
                break
        result = detected_sue_id
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        sue_data = self.get_sue_data(raw_input_lines)
        analysis = {
            'children': 3,
            'cats': 7,
            'samoyeds': 2,
            'pomeranians': 3,
            'akitas': 0,
            'vizslas': 0,
            'goldfish': 5,
            'trees': 3,
            'cars': 2,
            'perfumes': 1,
        }
        ranges = {
            'cats': '>',
            'trees': '>',
            'pomeranians': '<',
            'goldfish': '<',
        }
        solutions = (
            self.solve(sue_data, analysis),
            self.solve2(sue_data, analysis, ranges),
            )
        result = solutions
        return result

class Day15: # Science for Hungry People
    '''
    Science for Hungry People
    https://adventofcode.com/2015/day/15
    '''
    def get_recipes(self, raw_input_lines: List[str]):
        recipes = {}
        for raw_input_line in raw_input_lines:
            recipe_name, raw_recipe = raw_input_line.split(': ')
            recipe = {}
            raw_properties = raw_recipe.split(', ')
            for raw_property in raw_properties:
                property_name, a = raw_property.split(' ')
                recipe[property_name] = int(a)
            recipes[recipe_name] = recipe
        return recipes
    
    def solve(self, recipes, teaspoons: int):
        recipe_names = list(sorted(recipes))
        max_score = 0
        for a in range(teaspoons):
            for b in range(teaspoons - a):
                for c in range(teaspoons - a - b):
                    d = teaspoons - a - b - c
                    if d < 0:
                        break
                    amounts = [a, b, c, d]
                    capacity = 0
                    durability = 0
                    flavor = 0
                    texture = 0
                    for i in range(4):
                        recipe = recipes[recipe_names[i]]
                        capacity += recipe['capacity'] * amounts[i]
                        durability += recipe['durability'] * amounts[i]
                        flavor += recipe['flavor'] * amounts[i]
                        texture += recipe['texture'] * amounts[i]
                    capacity = 0 if capacity < 0 else capacity
                    durability = 0 if durability < 0 else durability
                    flavor = 0 if flavor < 0 else flavor
                    texture = 0 if texture < 0 else texture
                    score = capacity * durability * flavor * texture
                    max_score = max(max_score, score)
        result = max_score
        return result
    
    def solve2(self, recipes, teaspoons: int, target_calories: int):
        recipe_names = list(sorted(recipes))
        max_score = 0
        for a in range(teaspoons):
            for b in range(teaspoons - a):
                for c in range(teaspoons - a - b):
                    d = teaspoons - a - b - c
                    if d < 0:
                        break
                    amounts = [a, b, c, d]
                    capacity = 0
                    durability = 0
                    flavor = 0
                    texture = 0
                    calories = 0
                    for i in range(4):
                        recipe = recipes[recipe_names[i]]
                        capacity += recipe['capacity'] * amounts[i]
                        durability += recipe['durability'] * amounts[i]
                        flavor += recipe['flavor'] * amounts[i]
                        texture += recipe['texture'] * amounts[i]
                        calories += recipe['calories'] * amounts[i]
                    if calories != target_calories:
                        continue
                    capacity = 0 if capacity < 0 else capacity
                    durability = 0 if durability < 0 else durability
                    flavor = 0 if flavor < 0 else flavor
                    texture = 0 if texture < 0 else texture
                    score = capacity * durability * flavor * texture
                    max_score = max(max_score, score)
        result = max_score
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        recipes = self.get_recipes(raw_input_lines)
        solutions = (
            self.solve(recipes, 100),
            self.solve2(recipes, 100, 500),
            )
        result = solutions
        return result

class Day14: # Reindeer Olympics
    '''
    Reindeer Olympics
    https://adventofcode.com/2015/day/14
    '''
    def get_reindeer(self, raw_input_lines: List[str]):
        reindeer = {}
        for raw_input_line in raw_input_lines:
            parts = raw_input_line.split(' ')
            name = parts[0]
            speed = int(parts[3])
            fly_duration = int(parts[6])
            rest_duration = int(parts[13])
            reindeer[name] = (speed, fly_duration, rest_duration)
        result = reindeer
        return result
    
    def solve(self, reindeer):
        race = {}
        for name, info in reindeer.items():
            # distance, state, timer
            race[name] = [0, 'flying', info[1]]
        for _ in range(2503):
            for name in race:
                info = reindeer[name]
                state = race[name][1]
                if state == 'flying':
                    race[name][0] += info[0]
                race[name][2] -= 1
                if race[name][2] < 1:
                    if state == 'flying':
                        race[name][1] = 'resting'
                        race[name][2] = info[2]
                    else:
                        race[name][1] = 'flying'
                        race[name][2] = info[1]
        max_distance = 0
        for name in race:
            max_distance = max(max_distance, race[name][0])
        result = max_distance
        return result
    
    def solve2(self, reindeer):
        race = {}
        for name, info in reindeer.items():
            # distance, state, timer, points
            race[name] = [0, 'flying', info[1], 0]
        for _ in range(2503):
            for name in race:
                info = reindeer[name]
                state = race[name][1]
                if state == 'flying':
                    race[name][0] += info[0]
                race[name][2] -= 1
                if race[name][2] < 1:
                    if state == 'flying':
                        race[name][1] = 'resting'
                        race[name][2] = info[2]
                    else:
                        race[name][1] = 'flying'
                        race[name][2] = info[1]
            max_distance = max(info[0] for info in race.values())
            for name in race:
                if race[name][0] == max_distance:
                    race[name][3] += 1
        result = max(info[3] for info in race.values())
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        reindeer = self.get_reindeer(raw_input_lines)
        solutions = (
            self.solve(reindeer),
            self.solve2(reindeer),
            )
        result = solutions
        return result

class Day13: # Knights of the Dinner Table
    '''
    Knights of the Dinner Table
    https://adventofcode.com/2015/day/13
    '''
    def get_preferences(self, raw_input_lines: List[str]):
        preferences = collections.defaultdict(int)
        for raw_input_line in raw_input_lines:
            parts = raw_input_line.split(' ')
            person_a = parts[0]
            sign = -1 if parts[2] == 'lose' else 1
            value = sign * int(parts[3])
            person_b = parts[-1][:-1]
            pair = (min(person_a, person_b), max(person_a, person_b))
            preferences[pair] += value
        result = preferences
        return result
    
    def solve(self, preferences):
        max_happiness = float('-inf')
        guests = set(pair[0] for pair in preferences)
        guests |= set(pair[1] for pair in preferences)
        for seating in itertools.permutations(guests, len(guests)):
            happiness = 0
            for i in range(len(seating)):
                person_a = seating[i]
                person_b = seating[(i + 1) % len(seating)]
                pair = (min(person_a, person_b), max(person_a, person_b))
                happiness += preferences[pair]
            max_happiness = max(max_happiness, happiness)
        result = max_happiness
        return result
    
    def solve2(self, preferences):
        max_happiness = float('-inf')
        guests = set(pair[0] for pair in preferences)
        guests |= set(pair[1] for pair in preferences)
        guests.add('Me')
        for seating in itertools.permutations(guests, len(guests)):
            happiness = 0
            for i in range(len(seating)):
                person_a = seating[i]
                person_b = seating[(i + 1) % len(seating)]
                pair = (min(person_a, person_b), max(person_a, person_b))
                if pair in preferences:
                    happiness += preferences[pair]
            max_happiness = max(max_happiness, happiness)
        result = max_happiness
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        preferences = self.get_preferences(raw_input_lines)
        solutions = (
            self.solve(preferences),
            self.solve2(preferences),
            )
        result = solutions
        return result

class Day12: # JSAbacusFramework.io
    '''
    JSAbacusFramework.io
    https://adventofcode.com/2015/day/12
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = raw_input_lines[0]
        return result
    
    def solve(self, parsed_input):
        nums = []
        obj = json.loads(parsed_input)
        work = [obj]
        while len(work) > 0:
            element = work.pop()
            element_type = type(element)
            if element_type == list:
                for subelement in element:
                    work.append(subelement)
            elif element_type == dict:
                for subelement in element.values():
                    work.append(subelement)
            elif element_type == int:
                nums.append(element)
            elif element_type == str:
                pass
        result = sum(nums)
        return result
    
    def solve2(self, parsed_input):
        nums = []
        obj = json.loads(parsed_input)
        work = [obj]
        while len(work) > 0:
            element = work.pop()
            element_type = type(element)
            if element_type == list:
                for subelement in element:
                    work.append(subelement)
            elif element_type == dict:
                red_ind = False
                for value in element.values():
                    if value == 'red':
                        red_ind = True
                        break
                if not red_ind:
                    for subelement in element.values():
                        work.append(subelement)
            elif element_type == int:
                nums.append(element)
            elif element_type == str:
                pass
        result = sum(nums)
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

class Day11: # Corporate Policy
    '''
    Corporate Policy
    https://adventofcode.com/2015/day/11
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = raw_input_lines[0]
        return result
    
    def encode(self, password):
        encoded_password = 0
        for i, char in enumerate(reversed(password)):
            encoded_char = ord(char) - ord('a')
            encoded_password += (26 ** i) * encoded_char
        result = encoded_password
        return result

    def decode(self, encoded_password):
        password = collections.deque()
        while encoded_password > 0:
            char = chr(ord('a') + encoded_password % 26)
            encoded_password //= 26
            password.appendleft(char)
        result = password
        return result
    
    def check_validity(self, password):
        straight_ind = False
        banned_chars_ind = False
        paired_indices_found = set()
        for i, char in enumerate(password):
            if i >= 1:
                if password[i] == password[i - 1]:
                    paired_indices_found.add(i)
                    paired_indices_found.add(i - 1)
            if i >= 2:
                if (
                    ord(password[i]) == ord(password[i - 1]) + 1 and
                    ord(password[i - 1]) == ord(password[i - 2]) + 1
                ):
                    straight_ind = True
            if char in 'iol':
                banned_chars_ind = True
                break
        result = all([
            straight_ind,
            not banned_chars_ind,
            len(paired_indices_found) >= 4,
        ])
        return result
    
    def solve(self, prev_password):
        decoded_password = prev_password
        encoded_password = self.encode(prev_password)
        while True:
            encoded_password += 1
            decoded_password = self.decode(encoded_password)
            if self.check_validity(decoded_password):
                break
        result = ''.join(decoded_password)
        return result
    
    def solve2(self, prev_password):
        result = self.solve(self.solve(prev_password))
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        prev_password = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(prev_password),
            self.solve2(prev_password),
            )
        result = solutions
        return result

class Day10: # Elves Look, Elves Say
    '''
    Elves Look, Elves Say
    https://adventofcode.com/2015/day/10
    '''
    def get_sequence(self, raw_input_lines: List[str]):
        sequence = raw_input_lines[0]
        result = sequence
        return result
    
    def bruteforce(self, sequence, iteration_count):
        final_sequence = sequence
        for _ in range(iteration_count):
            counts = []
            for char in final_sequence:
                if len(counts) < 1 or counts[-1][1] != char:
                    counts.append([1, char])
                else:
                    counts[-1][0] += 1
            final_sequence = ''.join(
                ''.join(map(str, count)) for count in counts
            )
        result = final_sequence
        return result
    
    def solve(self, sequence, iteration_count):
        final_sequence = self.bruteforce(sequence, iteration_count)
        result = len(final_sequence)
        return result
    
    def main(self):
        assert self.bruteforce('1', 5) == '312211'
        raw_input_lines = get_raw_input_lines()
        sequence = self.get_sequence(raw_input_lines)
        solutions = (
            self.solve(sequence, 40),
            self.solve(sequence, 50),
            )
        result = solutions
        return result

class Day09: # All in a Single Night
    '''
    All in a Single Night
    https://adventofcode.com/2015/day/9
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        locations = set()
        distances = {}
        for row_data in raw_input_lines:
            source, _, destination, _, distance = row_data.split(' ')
            distance = int(distance)
            distances[(source, destination)] = distance
            distances[(destination, source)] = distance
            locations.add(source)
            locations.add(destination)
        result = (locations, distances)
        return result
    
    def bruteforce(self, locations, distances):
        min_distance = float('inf')
        for route in itertools.permutations(locations):
            distance = 0
            prev_stop = route[0]
            for stop in route[1:]:
                distance += distances[(prev_stop, stop)]
                prev_stop = stop
            min_distance = min(min_distance, distance)
        result = min_distance
        return result
    
    def bruteforce2(self, locations, distances):
        max_distance = float('-inf')
        for route in itertools.permutations(locations):
            distance = 0
            prev_stop = route[0]
            for stop in route[1:]:
                distance += distances[(prev_stop, stop)]
                prev_stop = stop
            max_distance = max(max_distance, distance)
        result = max_distance
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        locations, distances = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.bruteforce(locations, distances),
            self.bruteforce2(locations, distances),
            )
        result = solutions
        return result

class Day08: # Matchsticks
    '''
    Matchsticks
    https://adventofcode.com/2015/day/8
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, parsed_input):
        memory_char_count = 0
        literal_char_count = 0
        for row_data in parsed_input:
            prev_counts = (memory_char_count, literal_char_count)
            stack = []
            memory_char_count += 2
            for char in row_data[1:-1]:
                memory_char_count += 1
                stack.append(char)
                if (
                    len(stack) >= 1 and stack[0] != '\\' or
                    len(stack) >= 2 and stack[1] not in '\\"x' or
                    len(stack) >= 3 and stack[2] not in '0123456789abcdef' or
                    len(stack) >= 4 and stack[3] not in '0123456789abcdef'
                ):
                    literal_char_count += len(stack)
                    stack = []
                if (
                    (
                        len(stack) == 2 and
                        stack[0] == '\\' and
                        stack[1] in '\\"'
                    ) or
                    (
                        len(stack) == 4 and
                        stack[0] == '\\' and
                        stack[1] == 'x' and
                        stack[2] in '0123456789abcdef' and
                        stack[3] in '0123456789abcdef'
                    )
                ):
                    literal_char_count += 1
                    stack = []
        result = memory_char_count - literal_char_count
        return result
    
    def solve2(self, parsed_input):
        literal_char_count = 0
        encoded_char_count = 0
        for row_data in parsed_input:
            literal_char_count += len(row_data)
            encoded_char_count += 2
            for char in row_data:
                if char in ('"', '\\'):
                    encoded_char_count += 2
                else:
                    encoded_char_count += 1
        result = encoded_char_count - literal_char_count
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

class Day07: # Some Assembly Required
    '''
    Some Assembly Required
    https://adventofcode.com/2015/day/7
    '''
    def get_connections(self, raw_input_lines: List[str]):
        connections = {}
        for raw_input_line in raw_input_lines:
            a, wire = raw_input_line.split(' -> ')
            connection = list(a.split(' '))
            for i in range(len(connection)):
                try:
                    value = int(connection[i])
                    connection[i] = value
                except ValueError:
                    pass
            connections[wire] = tuple(connection)
        result = connections
        return result
    
    def solve(self, connections):
        OPS = {'NOT', 'AND', 'OR', 'LSHIFT', 'RSHIFT'}
        MODULO = 2 ** 16 # 0 to 65535
        wires = set(connections.keys())
        inputs = {}
        while len(wires) > 0:
            for wire in wires:
                needed = set()
                connection = connections[wire]
                for part in connection:
                    if (
                        part not in OPS and
                        type(part) == str and
                        part not in inputs.keys()
                    ):
                        needed.add(part)
                if len(needed) > 0:
                    continue
                if len(connection) == 1:
                    a = connection[0]
                    if type(a) is int:
                        inputs[wire] = a
                    else:
                        inputs[wire] = inputs[a]
                elif len(connection) == 2:
                    OP, a = connection
                    assert OP == 'NOT'
                    if type(a) is str:
                        a = inputs[a]
                    value = ~a
                    inputs[wire] = value
                elif len(connection) == 3:
                    a, OP, b = connection
                    value = 0
                    if type(a) is str:
                        a = inputs[a]
                    if type(b) is str:
                        b = inputs[b]
                    if OP == 'AND':
                        value = a & b
                    elif OP == 'OR':
                        value = a | b
                    elif OP == 'LSHIFT':
                        value = a << b
                    elif OP == 'RSHIFT':
                        value = a >> b
                    value = value % MODULO
                    inputs[wire] = value
                else:
                    raise AssertionError
            wires -= inputs.keys()
        result = inputs['a']
        return result
    
    def solve2(self, signal, connections):
        connections['b'] = (signal, )
        result = self.solve(connections)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        connections = self.get_connections(raw_input_lines)
        signal = self.solve(copy.deepcopy(connections))
        solutions = (
            signal,
            self.solve2(signal, copy.deepcopy(connections)),
            )
        result = solutions
        return result

class Day06: # Probably a Fire Hazard
    '''
    Probably a Fire Hazard
    https://adventofcode.com/2015/day/6
    '''
    def get_instructions(self, raw_input_lines: List[str]):
        instructions = []
        for raw_input_line in raw_input_lines:
            a, b = raw_input_line.split(' through ')
            mode = 'ERROR'
            if 'toggle' in a:
                a1, a2 = a.split(' ')
                left, top = tuple(map(int, a2.split(',')))
                mode = 'toggle'
            else:
                a1, a2, a3 = a.split(' ')
                left, top = tuple(map(int, a3.split(',')))
                if a2 == 'on':
                    mode = 'turn_on'
                elif a2 == 'off':
                    mode = 'turn_off'
            right, bottom = tuple(map(int, b.split(',')))
            instruction = (mode, left, top, right, bottom)
            instructions.append(instruction)
        result = instructions
        return result
    
    def solve(self, instructions):
        lights = [0] * (1000 * 1000)
        for mode, left, top, right, bottom in instructions:
            for row in range(top, bottom + 1):
                for col in range(left, right + 1):
                    index = 1000 * row + col
                    if mode == 'turn_on':
                        lights[index] = 1
                    elif mode == 'turn_off':
                        lights[index] = 0
                    elif mode == 'toggle':
                        lights[index] =  1 - lights[index]
        result = sum(lights)
        return result
    
    def solve2(self, instructions):
        lights = [0] * (1000 * 1000)
        for mode, left, top, right, bottom in instructions:
            for row in range(top, bottom + 1):
                for col in range(left, right + 1):
                    index = 1000 * row + col
                    if mode == 'turn_on':
                        lights[index] += 1
                    elif mode == 'turn_off':
                        lights[index] = max(0, lights[index] - 1)
                    elif mode == 'toggle':
                        lights[index] += 2
        result = sum(lights)
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

class Day05: # Doesn't He Have Intern-Elves For This?
    '''
    Doesn't He Have Intern-Elves For This?
    https://adventofcode.com/2015/day/5
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, parsed_input):
        '''
        Nice strings:
            Contain at least three vowels (aeiou)
            Contain at least one letter that appears twice in a row
            Do not contain the strings ab, cd, pq, or xy
        '''
        nice_string_count = 0
        for string in parsed_input:
            nice_ind = all([
                sum([
                    string.count('a'),
                    string.count('e'),
                    string.count('i'),
                    string.count('o'),
                    string.count('u'),
                ]) >= 3,
                any(
                    string[i] == string[i - 1] for
                    i in range(1, len(string))
                ),
                all([
                    'ab' not in string,
                    'cd' not in string,
                    'pq' not in string,
                    'xy' not in string,
                ]),
            ])
            if nice_ind:
                nice_string_count += 1
        result = nice_string_count
        return result
    
    def solve2(self, parsed_input):
        '''
        Nice strings:
            Contain a pair of any two letters that appear twice without overlapping
            Contain at least one letter which repeats with ONE letter between them
        '''
        nice_string_count = 0
        for string in parsed_input:
            pairs = {}
            duplicate_pair_found = False
            triplet_found = False
            for i in range(1, len(string)):
                pair = string[i - 1:i + 1]
                if pair in pairs and pairs[pair] < i - 1:
                    duplicate_pair_found = True
                if pair not in pairs:
                    pairs[pair] = i
                if i < len(string) - 1:
                    triplet = string[i - 1: i + 2]
                    if triplet[0] == triplet[2]:
                        triplet_found = True
            if duplicate_pair_found and triplet_found:
                nice_string_count += 1
        result = nice_string_count
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

class Day04: # The Ideal Stocking Stuffer
    '''
    The Ideal Stocking Stuffer
    https://adventofcode.com/2015/day/4
    '''
    def get_secret_key(self, raw_input_lines: List[str]):
        result = raw_input_lines[0]
        return result
    
    def solve(self, secret_key):
        num = 0
        while True:
            algorithm = hashlib.md5()
            key = secret_key + str(num)
            algorithm.update(key.encode('utf-8'))
            if algorithm.hexdigest()[:5] == '00000':
                break
            num += 1
        result = num
        return result
    
    def solve2(self, secret_key):
        num = 0
        while True:
            algorithm = hashlib.md5()
            key = secret_key + str(num)
            algorithm.update(key.encode('utf-8'))
            if algorithm.hexdigest()[:6] == '000000':
                break
            num += 1
        result = num
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        secret_key = self.get_secret_key(raw_input_lines)
        solutions = (
            self.solve(secret_key),
            self.solve2(secret_key),
            )
        result = solutions
        return result

class Day03: # Perfectly Spherical Houses in a Vacuum
    '''
    Perfectly Spherical Houses in a Vacuum
    https://adventofcode.com/2015/day/3
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = raw_input_lines[0]
        return result
    
    def solve(self, parsed_input):
        row = 0
        col = 0
        houses = set()
        for char in parsed_input:
            if char == '^':
                row -= 1
            elif char == '>':
                col += 1
            elif char == 'v':
                row += 1
            elif char == '<':
                col -= 1
            houses.add((row, col))
        result = len(houses)
        return result
    
    def solve2(self, parsed_input):
        santa_id = 0
        santas = [
            (0, 0),
            (0, 0),
            ]
        houses = set()
        for char in parsed_input:
            pos = santas[santa_id]
            if char == '^':
                santas[santa_id] = (pos[0] - 1, pos[1])
            elif char == 'v':
                santas[santa_id] = (pos[0] + 1, pos[1])
            elif char == '<':
                santas[santa_id] = (pos[0], pos[1] - 1)
            elif char == '>':
                santas[santa_id] = (pos[0], pos[1] + 1)
            houses.add(santas[santa_id])
            santa_id = 1 - santa_id
        result = len(houses)
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

class Day02: # I Was Told There Would Be No Math
    '''
    I Was Told There Would Be No Math
    https://adventofcode.com/2015/day/2
    '''
    def get_dimensions(self, raw_input_lines: List[str]):
        dimensions = []
        for raw_input_line in raw_input_lines:
            # Dimension = (Length, Width, Height)
            dimension = tuple(map(int, raw_input_line.split('x')))
            dimensions.append(dimension)
        result = dimensions
        return result
    
    def solve(self, dimensions: List[Tuple[int]]):
        wrapping_paper = []
        for dimension in dimensions:
            l, w, h = dimension
            smallest_side = min((l * w, w * h, h * l))
            needed = 2 * l * w + 2 * w * h + 2 * h * l + smallest_side
            wrapping_paper.append(needed)
        result = sum(wrapping_paper)
        return result
    
    def solve2(self, dimensions: List[Tuple[int]]):
        ribbon = []
        for dimension in dimensions:
            l, w, h = dimension
            perimeters = [
                2 * (l + w),
                2 * (w + h),
                2 * (h + l),
            ]
            volume = l * w * h
            needed = min(perimeters) + volume
            ribbon.append(needed)
        result = sum(ribbon)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        dimensions = self.get_dimensions(raw_input_lines)
        solutions = (
            self.solve(dimensions),
            self.solve2(dimensions),
            )
        result = solutions
        return result

class Day01: # Not Quite Lisp
    '''
    Not Quite Lisp
    https://adventofcode.com/2015/day/1
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = raw_input_lines[0]
        return result
    
    def solve(self, parsed_input):
        floor = 0
        for char in parsed_input:
            if char == '(':
                floor += 1
            elif char == ')':
                floor -= 1
        result = floor
        return result
    
    def solve2(self, parsed_input):
        floor = 0
        position = 0
        for position, char in enumerate(parsed_input, start=1):
            if char == '(':
                floor += 1
            elif char == ')':
                floor -= 1
            if floor == -1:
                result = position
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

if __name__ == '__main__':
    '''
    Usage
    python AdventOfCode2015.py 19 < inputs/2015day19.in
    '''
    solvers = {
        1: (Day01, 'Not Quite Lisp'),
        2: (Day02, 'I Was Told There Would Be No Math'),
        3: (Day03, 'Perfectly Spherical Houses in a Vacuum'),
        4: (Day04, 'The Ideal Stocking Stuffer'),
        5: (Day05, 'Doesn\'t He Have Intern-Elves For This?'),
        6: (Day06, 'Probably a Fire Hazard'),
        7: (Day07, 'Some Assembly Required'),
        8: (Day08, 'Matchsticks'),
        9: (Day09, 'All in a Single Night'),
       10: (Day10, 'Elves Look, Elves Say'),
       11: (Day11, 'Corporate Policy'),
       12: (Day12, 'JSAbacusFramework.io'),
       13: (Day13, 'Knights of the Dinner Table'),
       14: (Day14, 'Reindeer Olympics'),
       15: (Day15, 'Science for Hungry People'),
       16: (Day16, 'Aunt Sue'),
       17: (Day17, 'No Such Thing as Too Much'),
       18: (Day18, 'Like a GIF For Your Yard'),
       19: (Day19, 'Medicine for Rudolph'),
       20: (Day20, 'Infinite Elves and Infinite Houses'),
       21: (Day21, 'RPG Simulator 20XX'),
       22: (Day22, 'Wizard Simulator 20XX'),
       23: (Day23, 'Opening the Turing Lock'),
       24: (Day24, 'It Hangs in the Balance'),
       25: (Day25, 'Let It Snow'),
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
