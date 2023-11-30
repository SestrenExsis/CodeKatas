'''
Created on Nov 14, 2020

@author: Sestren
'''
import argparse
import asyncio
import collections
import heapq
import math
from typing import Dict, List

import intcode
    
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
    https://adventofcode.com/2019/day/?
    '''
    def get_parsed_input(self, raw_input_lines: List[str]) -> List[str]:
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def solve(self, parsed_input: List[str]) -> int:
        result = 0
        return result
    
    def solve2(self, parsed_input: List[str]) -> str:
        result = 0
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

# TODO(sestren): Solve intcode problems

class Vector3D():
    '''
    A simple 3-dimensional scalar
    '''
    def __init__(self, x:int, y:int, z:int):
        self.x = x
        self.y = y
        self.z = z
    
    def __copy__(self) -> 'Vector3D':
        result = Vector3D(
            x = self.x,
            y = self.y,
            z = self.z,
            )
        return result
    
    def __deepcopy__(self, memo) -> 'Vector3D':
        result = Vector3D(
            x = self.x,
            y = self.y,
            z = self.z,
            )
        return result
    
    def __str__(self) -> str:
        result = f'[{self.x}, {self.y}, {self.z}]'
        return result
    
    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z
    
    def __add__(self, v: 'Vector3D') -> 'Vector3D':
        try:
            result = Vector3D(
                x = self.x + v.x,
                y = self.y + v.y,
                z = self.z + v.z,
                )
            return result
        except (TypeError, NameError) as e:
            raise Exception(e)
    
    def __sub__(self, v: 'Vector3D') -> 'Vector3D':
        try:
            result = Vector3D(
                x = self.x - v.x,
                y = self.y - v.y,
                z = self.z - v.z,
                )
            return result
        except (TypeError, NameError) as e:
            raise Exception(e)
    
    def __neg__(self) -> 'Vector3D':
        try:
            result = Vector3D(
                x = -self.x,
                y = -self.y,
                z = -self.z,
                )
            return result
        except (TypeError, NameError) as e:
            raise Exception(e)
    
    def __mul__(self, v) -> 'Vector3D':
        try:
            if isinstance(v, Vector3D):
                result = Vector3D(
                    x = self.x * v.x,
                    y = self.y * v.y,
                    z = self.z * v.z,
                    )
            elif (
                isinstance(v, int) or
                isinstance(v, float)
                ):
                result = Vector3D(
                    x = self.x * v,
                    y = self.y * v,
                    z = self.z * v,
                    )
            return result
        except (TypeError, NameError) as e:
            raise Exception(e)
    
    def __div__(self, v) -> 'Vector3D':
        try:
            if isinstance(v, Vector3D):
                result = Vector3D(
                    x = self.x / v.x,
                    y = self.y / v.y,
                    z = self.z / v.z,
                    )
            elif (
                isinstance(v, int) or
                isinstance(v, float)
                ):
                result = Vector3D(
                    x = self.x / v,
                    y = self.y / v,
                    z = self.z / v,
                    )
            return result
        except (TypeError, NameError) as e:
            raise Exception(e)
    
    @property
    def magnitude(self) -> float:
        result = math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
        return result
    
    @property
    def normalized(self) -> 'Vector3D':
        magnitude: float = self.magnitude
        if (magnitude > 0.0):
            result = Vector3D(
                    x = self.x,
                    y = self.y,
                    z = self.z,
                    )
            result /= magnitude
        else:
            raise Exception('*** Vector: error, normalizing zero vector! ***')
    
    def dot(self, v: 'Vector3D') -> float:
        '''
        Dot product
        '''
        try:
            result = self.x * v.x + self.y * v.y + self.z * v.z
            return result
        except (TypeError, NameError) as e:
            raise Exception(e)
    
    def cross(self, v: 'Vector3D') -> 'Vector3D':
        '''
        Cross product
        '''
        try:
            result = Vector3D(
                    x = self.y * v.z - self.z * v.y,
                    y = self.z * v.x - self.x * v.z,
                    z = self.x * v.y - self.y * v.x,
                    )
            return result
        except (TypeError, NameError) as e:
            raise Exception(e)
        
class Day22:
    '''
    Slam Shuffle
    '''
    def __init__(self):
        self.debug = False
    
    def get_shuffling_instructions(self, raw_input_lines: 'List'):
        shuffling_instructions = []
        for raw_input_line in raw_input_lines:
            shuffling_instruction = None
            if raw_input_line == 'deal into new stack':
                shuffling_instruction = ('reverse', None)
            else:
                raw_instruction = raw_input_line.split(' ')
                shuffling_instruction = (raw_instruction[0], int(raw_instruction[-1]))
            shuffling_instructions.append(shuffling_instruction)
        return shuffling_instructions
    
    def solve(self, shuffling_instructions, deck_size: int=10007, find_card: int=2019):
        deck = list(map(str, range(deck_size)))
        for shuffle_type, amount in shuffling_instructions:
            if shuffle_type == 'reverse':
                deck = list(reversed(deck))
            elif shuffle_type == 'cut':
                deck = deck[amount:] + deck[:amount]
            elif shuffle_type == 'deal':
                n = len(deck)
                new_deck = deck[:]
                index = 0
                for card in deck:
                    new_deck[index % n] = card
                    index += amount
                deck = new_deck
            else:
                continue
        if self.debug:
            print(' '.join(deck))
        result = deck.index(str(find_card))
        return result
    
    def get_polynomial_of_shuffle(self, deck_size: int, shuffling_instructions: list):
        # ax + b
        a = 1
        b = 0
        for shuffle_type, amount in reversed(shuffling_instructions):
            assert shuffle_type in ('reverse', 'cut', 'deal')
            if shuffle_type == 'reverse':
                a = -a
                b = deck_size - b - 1
            elif shuffle_type == 'cut':
                b = (b + amount) % deck_size
            elif shuffle_type == 'deal':
                z = pow(amount, deck_size - 2, deck_size) # modinv(n, L)
                a = (a * z) % deck_size
                b = (b * z) % deck_size
        result = (a, b)
        return result
    
    def get_polynomial_power(self, a, b, repeat, deck_size):
        # f(x) = ax + b
        # g(x) = cx + d
        # f^2(x) = a(ax + b) + b = aax + ab + b
        # f(g(x)) = a(cx + d) + b = acx + ad + b
        result = None
        if repeat == 0:
            result = (1, 0)
        elif repeat % 2 == 0:
            result = self.get_polynomial_power(
                (a * a) % deck_size,
                (a * b + b) % deck_size,
                repeat // 2,
                deck_size,
                )
        else:
            (c, d) = self.get_polynomial_power(a, b, repeat - 1, deck_size)
            result = ((a * c) % deck_size, (a * d + b) % deck_size)
        return result
    
    def solve2(self,
            shuffling_instructions,
            deck_size: int=119315717514047,
            repeat: int=101741582076661,
            find_card: int=2020,
            ):
        a, b = self.get_polynomial_of_shuffle(deck_size, shuffling_instructions)
        a, b = self.get_polynomial_power(a, b, repeat, deck_size)
        card_index = (find_card * a + b) % deck_size
        result = card_index
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        shuffling_instructions = self.get_shuffling_instructions(raw_input_lines)
        solutions = (
            self.solve(shuffling_instructions),
            self.solve2(shuffling_instructions),
            )
        result = solutions
        return result

class Day20:
    '''
    Donut Maze
    '''
    def get_grid(self, raw_input_lines: list) -> dict:
        grid = {}
        for row, raw_input_line in enumerate(raw_input_lines):
            for col, cell in enumerate(raw_input_line):
                grid[(row, col)] = cell
        return grid
    
    def get_nodes(self, grid: dict) -> dict:
        # 'DI': ((row1, col1), (row2, col2))
        nodes = collections.defaultdict(set)
        for (row1, col1), cell in grid.items():
            if cell.isupper():
                for (row2, col2) in (
                    (row1 + 1, col1 + 0),
                    (row1 + 0, col1 + 1),
                    ):
                    try:
                        if grid[(row2, col2)].isupper():
                            node_name = cell + grid[(row2, col2)]
                            for (row3, col3) in (
                                (row1 + 2, col1 + 0),
                                (row1 + 0, col1 + 2),
                                (row1 - 1, col1 + 0),
                                (row1 + 0, col1 - 1),
                                ):
                                try:
                                    if grid[(row3, col3)] == '.':
                                        node_pos = (row3, col3)
                                        nodes[node_name].add(node_pos)
                                except KeyError:
                                    pass
                    except KeyError:
                        pass
        result = nodes
        return result

    def get_board(self, grid: dict, traversed: set) -> str:
        min_row = min(row for (row, col) in grid)
        max_row = max(row for (row, col) in grid)
        min_col = min(col for (row, col) in grid)
        max_col = max(col for (row, col) in grid)
        board = []
        for row in range(min_row, max_row + 1):
            row_data = []
            for col in range(min_col, max_col + 1):
                cell = ' '
                if (col, row) in traversed:
                    cell = '*'
                elif (col, row) in grid:
                    cell = grid[(col, row)]
                row_data.append(cell)
            board.append(row_data)
        result = '\n' + '\n'.join(''.join(cell for cell in row_data) for row_data in board)
        return result
    
    def solve(self, grid):
        nodes = self.get_nodes(grid)
        portals = {}
        for node_name, node_points in nodes.items():
            if len(node_points) == 2:
                iter_node_points = iter(node_points)
                node_a = next(iter_node_points)
                node_b = next(iter_node_points)
                portals[node_a] = node_b
                portals[node_b] = node_a
        traversed = set()
        shortest_path = None
        source_pos = next(iter(nodes['AA']))
        work = collections.deque()
        work.append((0, source_pos))
        while len(work) > 0:
            distance, (row, col) = work.pop()
            if (row, col) in traversed:
                continue
            traversed.add((row, col))
            if (row, col) in nodes['ZZ']:
                shortest_path = distance
                break
            for (next_row, next_col) in (
                (row - 1, col + 0),
                (row + 0, col + 1),
                (row + 1, col + 0),
                (row + 0, col - 1),
                ):
                if grid[(next_row, next_col)] == '.':
                    work.appendleft((distance + 1, (next_row, next_col)))
            if (row, col) in portals:
                portal_exit = portals[(row, col)]
                work.appendleft((distance + 1, portal_exit))
        result = shortest_path
        return result
    
    def solve2(self, grid):
        rows = 1 + max(row for (row, col) in grid)
        cols = 1 + max(col for (row, col) in grid)
        nodes = self.get_nodes(grid)
        portals = {}
        for node_name, node_points in nodes.items():
            if len(node_points) == 2:
                iter_node_points = iter(node_points)
                node_a = next(iter_node_points)
                node_b = next(iter_node_points)
                portals[node_a] = node_b
                portals[node_b] = node_a
        traversed = set()
        shortest_path = None
        source_pos = next(iter(nodes['AA']))
        work = collections.deque()
        work.append((0, (*source_pos, 0)))
        while len(work) > 0:
            distance, (row, col, depth) = work.pop()
            if (row, col, depth) in traversed:
                continue
            traversed.add((row, col, depth))
            if (row, col) in nodes['ZZ'] and depth == 0:
                shortest_path = distance
                break
            for (next_row, next_col) in (
                (row - 1, col + 0),
                (row + 0, col + 1),
                (row + 1, col + 0),
                (row + 0, col - 1),
                ):
                if grid[(next_row, next_col)] == '.':
                    work.appendleft((distance + 1, (next_row, next_col, depth)))
            if (row, col) in portals:
                portal_exit = portals[(row, col)]
                next_depth = depth - 1 if (
                    row in (2, rows - 3) or 
                    col in (2, cols - 3)
                    ) else depth + 1
                if next_depth >= 0:
                    work.appendleft((distance + 1, (*portal_exit, next_depth)))
        result = shortest_path
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
        
class Day18:
    '''
    Many-Worlds Interpretation
    '''
    def __init__(self):
        self.node_names = set('@abcdefghijklmnopqrstuvwxyz')
    
    def get_grid(self, raw_input_lines: 'List') -> dict:
        grid = {}
        for row, raw_input_line in enumerate(raw_input_lines):
            for col, cell in enumerate(raw_input_line):
                grid[(row, col)] = cell
        return grid
    
    def get_nodes(self, grid: dict) -> dict:
        nodes = {}
        for (row, col) in grid:
            cell = grid[(row, col)]
            if cell in self.node_names:
                nodes[cell] = (row, col)
        result = nodes
        return result
    
    def get_paths_from_node(self, grid: dict, source_pos: tuple) -> dict:
        # BFS
        paths_from_node = {}
        traversed = set()
        work = [(source_pos, 0, tuple())]
        while len(work) > 0:
            (y, x), distance, keys_needed = work.pop(0)
            if (y, x) in traversed:
                continue
            traversed.add((y, x))
            if grid[(y, x)].islower() and source_pos != (y, x):
                paths_from_node[grid[(y, x)]] = (distance, frozenset(keys_needed))
            for (dy, dx) in (
                (y - 1, x + 0),
                (y + 0, x + 1),
                (y + 1, x + 0),
                (y + 0, x - 1),
                ):
                if grid[(dy, dx)] != '#':
                    next_nodes = keys_needed + (grid[(dy, dx)].lower(),) if grid[(dy, dx)].isupper() else keys_needed
                    work.append(((dy, dx), distance + 1, next_nodes))
        return paths_from_node
    
    def solve(self, grid):
        nodes = self.get_nodes(grid)
        # Create node paths
        node_paths = { source_node:dict() for source_node in nodes.keys() }
        for source_node in nodes:
            paths_from_node = self.get_paths_from_node(grid, nodes[source_node])
            for target_node, path_distance in paths_from_node.items():
                node_paths[source_node][target_node] = path_distance
        # Search
        result = None
        key_count = len(nodes) - 1
        work = [(0, (('@',), frozenset()))]
        memo = dict()
        while len(work) > 0:
            distance, node = heapq.heappop(work)
            if node in memo:
                continue
            memo[node] = distance
            positions, keys = node
            if len(keys) == key_count:
                result = distance
                break
            for i in range(len(positions)):
                for target_node, (path_distance, keys_needed) in node_paths[positions[i]].items():
                    if len(keys_needed - keys) == 0 and target_node not in keys:
                        next_positions = positions[:i] + (target_node,) + positions[i + 1:]
                        next_work = ((distance + path_distance), (next_positions, keys | frozenset(target_node)))
                        heapq.heappush(work, next_work)
        return result
    
    def solve2(self, original_grid):
        # Alter grid for part 2
        grid = dict(original_grid)
        nodes = self.get_nodes(grid)
        origin_row, origin_col = nodes['@']
        del nodes['@']
        for (offset_row, offset_col, overwrite_value) in (
            (-1, -1, '0'),
            (-1,  0, '#'),
            (-1, +1, '1'),
            ( 0, -1, '#'),
            ( 0,  0, '#'),
            ( 0, +1, '#'),
            (+1, -1, '2'),
            (+1,  0, '#'),
            (+1, +1, '3'),
            ):
            row = origin_row + offset_row
            col = origin_col + offset_col
            grid[(row, col)] = overwrite_value
            if overwrite_value != '#':
                nodes[overwrite_value] = (row, col)
        # Create node paths
        node_paths = { source_node:dict() for source_node in nodes.keys() }
        for source_node in nodes:
            paths_from_node = self.get_paths_from_node(grid, nodes[source_node])
            for target_node, path_distance in paths_from_node.items():
                node_paths[source_node][target_node] = path_distance
        # Search
        result = None
        key_count = len(nodes) - 4
        work = [(0, (('0', '1', '2', '3'), frozenset()))]
        memo = dict()
        while len(work) > 0:
            distance, node = heapq.heappop(work)
            if node in memo:
                continue
            memo[node] = distance
            positions, keys = node
            if len(keys) == key_count:
                result = distance
                break
            for i in range(len(positions)):
                for target_node, (path_distance, keys_needed) in node_paths[positions[i]].items():
                    if len(keys_needed - keys) == 0 and target_node not in keys:
                        next_positions = positions[:i] + (target_node,) + positions[i + 1:]
                        next_work = ((distance + path_distance), (next_positions, keys | frozenset(target_node)))
                        heapq.heappush(work, next_work)
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
        
class Day16:
    '''
    Flawed Frequency Transmission
    '''
    def get_parsed_input(self, raw_input_lines: 'List'):
        result = list(map(int, list(raw_input_lines[0])))
        return result
    
    def process_signal(self, original_signal, message_offset=1):
        pattern = [0, 1, 0, -1]
        p = len(pattern)
        n = len(original_signal)
        signal = [0] * n
        for i in range(n):
            for j in range(n):
                k = ((j + message_offset) // (i + 1)) % p
                signal[i] += original_signal[j] * pattern[k]
            signal[i] = abs(signal[i]) % 10
        result = signal
        return result
    
    def solve(self, original_signal, phase_count=100):
        pattern = [0, 1, 0, -1]
        signal = original_signal[:]
        n = len(signal)
        for phase in range(phase_count):
            signal = self.process_signal(signal, message_offset=1)
        result = ''.join(map(str, signal[:8]))
        return result
    
    def solve2(self, original_signal, phase_count=100):
        pattern = [0, 1, 0, -1]
        signal = original_signal * 10000
        offset = int(''.join(map(str, signal[:7])), 10)
        n = len(signal)
        for phase in range(phase_count):
            partial_sum = sum(signal[i] for i in range(offset, n))
            for i in range(offset, n):
                t = partial_sum
                partial_sum -= signal[i]
                signal[i] = t % 10 if t >= 0 else (-t) % 10
        result = int(''.join(map(str, signal[offset: offset + 8])), 10)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        parsed_input = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(parsed_input, 100),
            self.solve2(parsed_input, 100),
            )
        result = solutions
        return result

class Day14:
    '''
    Space Stoichiometry
    '''
    def get_recipes(self, raw_input_lines: 'List'):
        recipes = {}
        for raw_input_line in raw_input_lines:
            raw_input, raw_output = raw_input_line.split(' => ')
            recipe_amount, recipe = raw_output.split(' ')
            recipe_amount = int(recipe_amount)
            raw_ingredients = raw_input.split(', ')
            ingredients = {}
            for raw_ingredient in raw_ingredients:
                ingredient_amount, ingredient = raw_ingredient.split(' ')
                ingredients[ingredient] = int(ingredient_amount)
            assert recipe not in recipes
            recipes[recipe] = (recipe_amount, ingredients)
        return recipes
    
    def find_fuel(self, recipes, fuel_required, ore_gathered):
        ingredients_required = collections.defaultdict(int)
        recipe_amount, ingredients = recipes['FUEL']
        for ingredient, amount in ingredients.items():
            ingredients_required[ingredient] += fuel_required * amount
        ore_left =  ore_gathered
        fuel_amount = 0
        while ore_left > 0 and fuel_amount < fuel_required:
            while True:
                try:
                    ingredient = next(
                        __ingredient for __ingredient, __amount in 
                        ingredients_required.items() if 
                        __ingredient not in ('ORE') and __amount > 0
                        )
                except StopIteration:
                    break
                recipe_amount, recipe = recipes[ingredient]
                serving_count = math.ceil(ingredients_required[ingredient] / recipe_amount)
                ingredients_required[ingredient] -= serving_count * recipe_amount
                for sub_ingredient, sub_amount in recipe.items():
                    ingredients_required[sub_ingredient] += serving_count * sub_amount
            ore_left = ore_gathered - ingredients_required['ORE']
            if ore_left >= 0:
                fuel_amount += fuel_required
        result = fuel_amount
        return result
    
    def solve(self, recipes):
        # Minimum ORE needed to make 1 FUEL?
        recipe_amount, ingredients = recipes['FUEL']
        assert recipe_amount == 1
        ingredients_required = collections.defaultdict(int)
        for ingredient, amount in ingredients.items():
            ingredients_required[ingredient] += amount
        while True:
            try:
                ingredient = next(
                    __ingredient for __ingredient, __amount in 
                    ingredients_required.items() if 
                    __ingredient not in ('ORE') and __amount > 0
                    )
            except StopIteration:
                break
            recipe_amount, recipe = recipes[ingredient]
            ingredients_required[ingredient] = ingredients_required[ingredient] - recipe_amount
            for sub_ingredient, sub_amount in recipe.items():
                ingredients_required[sub_ingredient] += sub_amount
        result = ingredients_required['ORE']
        return result
    
    def solve2(self, recipes, ore_gathered=10 ** 12):
        left = 1
        right = ore_gathered
        result = None
        while left < right:
            fuel_required = (left + right) // 2
            fuel_amount = self.find_fuel(recipes, fuel_required, ore_gathered)
            if fuel_amount == 0:
                right = fuel_required - 1
            else:
                left = fuel_required + 1
                result = fuel_amount
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        recipes = self.get_recipes(raw_input_lines)
        solutions = (
            self.solve(recipes),
            self.solve2(recipes, ore_gathered=10 ** 12),
            )
        result = solutions
        return result

class Day12:
    '''
    The N-Body Problem
    '''
    class Moon:
        
        def __init__(self, position: Vector3D):
            self.position: Vector3D = position
            self.velocity: Vector3D = Vector3D(x=0, y=0, z=0)
        
        def __str__(self):
            result = '[' + str(self.position) + ' ' + str(self.velocity) + ']'
            return result
        
        @property
        def x(self):
            result = (self.position.x, self.velocity.x)
            return result
        
        @property
        def y(self):
            result = (self.position.y, self.velocity.y)
            return result
        
        @property
        def z(self):
            result = (self.position.z, self.velocity.z)
            return result
        
        @property
        def potential_energy(self):
            result = sum(map(abs, self.position))
            return result
        
        @property
        def kinetic_energy(self):
            result = sum(map(abs, self.velocity))
            return result
            
    def get_moons(self, raw_input_lines: 'List'):
        moon_names = ['Io', 'Europa', 'Ganymede', 'Callisto']
        moons = {}
        for i, raw_input_line in enumerate(raw_input_lines):
            raw_pos = raw_input_line[1:-1].split(', ')
            x = int(raw_pos[0].split('=')[1])
            y = int(raw_pos[1].split('=')[1])
            z = int(raw_pos[2].split('=')[1])
            position = Vector3D(x=x, y=y, z=z)
            moon = Day12.Moon(position)
            moons[moon_names[i]] = moon
        result = moons
        return result
    
    def solve(self, moons: dict, time_steps=1000):
        for t in range(time_steps):
            # apply gravity
            gravity = {}
            for source_name, source in moons.items():
                gravity[source_name] = Vector3D(x=0, y=0, z=0)
                for target_name, target in moons.items():
                    if target_name == source_name:
                        continue
                    if source.position.x > target.position.x:
                        gravity[source_name].x += -1
                    elif source.position.x < target.position.x:
                        gravity[source_name].x += 1
                    if source.position.y > target.position.y:
                        gravity[source_name].y += -1
                    elif source.position.y < target.position.y:
                        gravity[source_name].y += 1
                    if source.position.z > target.position.z:
                        gravity[source_name].z += -1
                    elif source.position.z < target.position.z:
                        gravity[source_name].z += 1
            for moon_name in gravity:
                moons[moon_name].velocity += gravity[moon_name]
            # apply velocity
            for moon in moons.values():
                moon.position += moon.velocity
        result = sum(moon.potential_energy * moon.kinetic_energy for moon in moons.values())
        return result
    
    def solve2(self, moons: dict):
        # Solve each dimension separately, then find the LCM of those three numbers
        x_states = set()
        y_states = set()
        z_states = set()
        x_states.add((moons['Io'].x, moons['Europa'].x, moons['Ganymede'].x, moons['Callisto'].x))
        y_states.add((moons['Io'].y, moons['Europa'].y, moons['Ganymede'].y, moons['Callisto'].y))
        z_states.add((moons['Io'].z, moons['Europa'].z, moons['Ganymede'].z, moons['Callisto'].z))
        x_cycles = None
        y_cycles = None
        z_cycles = None
        t = 0
        while True:
            gravity = {}
            for source_name, source in moons.items():
                gravity[source_name] = Vector3D(x=0, y=0, z=0)
                for target_name, target in moons.items():
                    if target_name == source_name:
                        continue
                    if source.position.x > target.position.x:
                        gravity[source_name].x += -1
                    elif source.position.x < target.position.x:
                        gravity[source_name].x += 1
                    if source.position.y > target.position.y:
                        gravity[source_name].y += -1
                    elif source.position.y < target.position.y:
                        gravity[source_name].y += 1
                    if source.position.z > target.position.z:
                        gravity[source_name].z += -1
                    elif source.position.z < target.position.z:
                        gravity[source_name].z += 1
            for moon_name in gravity:
                moons[moon_name].velocity += gravity[moon_name]
            # apply velocity
            for moon in moons.values():
                moon.position += moon.velocity
            t += 1
            if x_cycles is None:
                x = (moons['Io'].x, moons['Europa'].x, moons['Ganymede'].x, moons['Callisto'].x)
                if x in x_states:
                    x_cycles = len(x_states)
                x_states.add(x)
            if y_cycles is None:
                y = (moons['Io'].y, moons['Europa'].y, moons['Ganymede'].y, moons['Callisto'].y)
                if y in y_states:
                    y_cycles = len(y_states)
                y_states.add(y)
            if z_cycles is None:
                z = (moons['Io'].z, moons['Europa'].z, moons['Ganymede'].z, moons['Callisto'].z)
                if z in z_states:
                    z_cycles = len(z_states)
                z_states.add(z)
            if x_cycles is not None and y_cycles is not None and z_cycles is not None:
                break
        a, b, c = sorted([x_cycles, y_cycles, z_cycles])
        step = c
        result = step
        while result % b != 0:
            result += step
        step = result
        while result % a != 0:
            result += step
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        moons = self.get_moons(raw_input_lines)
        moons2 = self.get_moons(raw_input_lines)
        solutions = (
            self.solve(moons, time_steps=1000),
            self.solve2(moons2),
            )
        result = solutions
        return result

class Day10:
    '''
    Monitoring Station
    '''
    def get_asteroids(self, raw_input_lines: 'List'):
        asteroids = set()
        for row, row_data in enumerate(raw_input_lines):
            for col, cell in enumerate(row_data):
                if cell == '#':
                    asteroids.add((col, row))
        return asteroids
    
    def solve(self, asteroids: 'set'):
        asteroid_visibility = {}
        for (source_col, source_row) in asteroids:
            current_visibility = set()
            for (target_col, target_row) in asteroids:
                angle = math.atan2(target_row - source_row, target_col - source_col)
                current_visibility.add(angle)
            asteroid_visibility[(source_col, source_row)] = current_visibility
        result = max((len(v), k) for k, v in asteroid_visibility.items())
        return result[0]
    
    def get_field_map(self, asteroids, source, target, vaporized, width=33, height=33):
        field_map = []
        for y in range(height):
            current_row = ''
            for x in range(width):
                cell = '.'
                if (x, y) == source:
                    cell = 'S'
                elif (x, y) == target:
                    cell = 'T'
                elif (x, y) in vaporized:
                    cell = '*'
                elif (x, y) in asteroids:
                    cell = '#'
                current_row += cell
            field_map.append(current_row)
        result = field_map
        return result

    def get_angle(self, source: tuple, target: tuple):
        angle = math.degrees(math.atan2(
            source[1] - target[1],
            source[0] - target[0],
            ))
        result = (angle + 270) % 360
        return result
    
    def solve2(self, asteroids: 'set', vaporized_target=200, visual_debug=False):
        target_visibility = {}
        for source in asteroids:
            target_angles = collections.defaultdict(set)
            for target in asteroids:
                if target == source:
                    continue
                angle = self.get_angle(source, target)
                target_angles[angle].add(target)
            target_visibility[source] = target_angles
        target_count, source = max((len(v), k) for k, v in target_visibility.items())
        assert(target_count >= vaporized_target)
        target_angles = target_visibility[source]
        vaporized = []
        for target_angle in sorted(target_angles):
            first_target = next(iter(sorted(
                target_angles[target_angle], 
                key=lambda t: (t[0] - source[0]) ** 2 + (t[1] - source[1]) ** 2,
                )))
            vaporized.append(first_target)
            if visual_debug:
                print(first_target, target_angle)
                field_map = self.get_field_map(asteroids, source, first_target, vaporized)
                for row in field_map:
                    print(row)
                print(' ')
        assert len(vaporized) >= vaporized_target
        result = vaporized[vaporized_target - 1]
        return result[0] * 100 + result[1]
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        asteroids = self.get_asteroids(raw_input_lines)
        solutions = (
            self.solve(asteroids),
            self.solve2(asteroids, 200, False),
            )
        result = solutions
        return result

class Day08:
    '''
    Space Image Format
    '''
    def get_parsed_input(self, raw_input_lines: 'List'):
        return raw_input_lines[0]
    
    def solve(self, parsed_input: 'List', width=25, height=6):
        # layer with fewest 0 digits
        span = width * height
        start = 0
        min_zero_count = None
        result = None
        while start + span < len(parsed_input):
            zero_count = parsed_input[start:start + span].count('0')
            if min_zero_count is None or zero_count < min_zero_count:
                min_zero_count = zero_count
                one_count = parsed_input[start:start + span].count('1')
                two_count = parsed_input[start:start + span].count('2')
                result = one_count * two_count
            start += span
        return result
    
    def solve2(self, parsed_input: 'List', width=25, height=6):
        cell_count = width * height
        image = []
        for _ in range(height):
            row = [3] * width
            image.append(row)
        layer_count = len(parsed_input) // cell_count
        for row in range(height):
            for col in range(width):
                for layer in range(layer_count):
                    index = layer * cell_count + row * width + col
                    if parsed_input[index] == '2':
                        continue
                    else:
                        image[row][col] = '#' if parsed_input[index] == '1' else '.'
                        break
        result = '\n' + '\n'.join(''.join(cell for cell in row) for row in image)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        parsed_input = self.get_parsed_input(raw_input_lines)
#         image = self.solve2(parsed_input)
#         for row in image:
#             print(''.join(row))
        solutions = (
            self.solve(parsed_input),
            self.solve2(parsed_input),
            )
        result = solutions
        return result

class Day06: # Universal Orbit Map
    '''
    Universal Orbit Map
    https://adventofcode.com/2019/day/6
    '''
    def get_orbits(self, raw_input_lines: List[str]) -> List[str]:
        orbits = {}
        for raw_input_line in raw_input_lines:
            target, source = raw_input_line.split(')')
            orbits[source] = target
        result = orbits
        return result
    
    def solve(self, orbits):
        depths = collections.defaultdict(int)
        depths['COM'] = 0
        work = collections.deque()
        work.append('COM')
        while len(work) > 0:
            current_source = work.pop()
            for source, target in orbits.items():
                if target == current_source:
                    work.appendleft(source)
                    depths[source] = depths[target] + 1
        result = sum(depths.values())
        return result
    
    def solve2(self, orbits):
        your_path = []
        current_orbit = 'YOU'
        while len(your_path) == 0 or current_orbit != 'COM':
            current_orbit = orbits[current_orbit]
            your_path.append(current_orbit)
        santas_path = []
        current_orbit = 'SAN'
        while len(santas_path) == 0 or current_orbit != 'COM':
            current_orbit = orbits[current_orbit]
            santas_path.append(current_orbit)
        result = -1
        for i in range(len(your_path)):
            current_orbit = your_path[i]
            try:
                j = santas_path.index(current_orbit)
                result = i + j
                break
            except (ValueError) as e:
                continue
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        orbits = self.get_orbits(raw_input_lines)
        solutions = (
            self.solve(orbits),
            self.solve2(orbits),
            )
        result = solutions
        return result

class Day05:
    '''
    Sunny with a Chance of Asteroids
    https://adventofcode.com/2019/day/5
    '''
    def get_parsed_input(self, raw_input_lines: List[str]) -> List[str]:
        nums = map(int, raw_input_lines[0].split(','))
        result = collections.defaultdict(int)
        for index, str_num in enumerate(nums):
            result[index] = int(str_num)
        return result
    
    def solve(self, parsed_input: Dict[int, int]) -> int:
        vm = intcode.IntcodeVM(parsed_input)
        vm.debug = False
        vm.send_input(1)
        result = None
        while True:
            vm.run()
            output = vm.get_next_output()
            if output is None:
                break
            result = output
        return result
    
    def solve2(self, parsed_input: Dict[int, int]) -> int:
        result = None
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

class Day04:
    '''
    Secure Container
    https://adventofcode.com/2019/day/4
    '''
    def get_parsed_input(self, raw_input_lines: 'List'):
        lower, upper = map(int, raw_input_lines[0].split('-'))
        return (lower, upper)
    
    def solve(self, lower: int, upper: int):
        valid_count = 0
        for password in range(lower, upper + 1):
            chars = str(password)
            n = len(chars)
            if n != 6:
                continue
            repeat_found = False
            valid = True
            for i in range(1, n):
                if int(chars[i]) < int(chars[i - 1]):
                    valid = False
                    break
                if chars[i] == chars[i - 1]:
                    repeat_found = True
            if not valid or not repeat_found:
                continue
            valid_count += 1
        result = valid_count
        return result
    
    def solve2(self, lower: int, upper: int):
        valid_count = 0
        for password in range(lower, upper + 1):
            chars = str(password)
            n = len(chars)
            if n != 6:
                continue
            valid = True
            streaks = [1]
            for i in range(1, n):
                if int(chars[i]) < int(chars[i - 1]):
                    valid = False
                    break
                if chars[i] == chars[i - 1]:
                    streaks[-1] += 1
                else:
                    streaks.append(1)
            if not valid or 2 not in streaks:
                continue
            valid_count += 1
        result = valid_count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        parsed_input = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(*parsed_input),
            self.solve2(*parsed_input),
            )
        result = solutions
        return result

class Day03:
    '''
    Crossed Wires
    https://adventofcode.com/2019/day/3
    '''
    def get_parsed_input(self, raw_input_lines: 'List'):
        wires = []
        for raw_input_line in raw_input_lines:
            wires.append(raw_input_line.split(','))
        return wires
    
    def solve(self, wires: 'List'):
        lines = [set(), set()]
        for i, wire in enumerate(wires):
            c = (0, 0)
            for move in wire:
                direction, distance = move[:1], int(move[1:])
                if direction == 'R':
                    d = (c[0] + distance, c[1])
                    lines[i].add((c, d))
                elif direction == 'D':
                    d = (c[0], c[1] + distance)
                    lines[i].add((c, d))
                elif direction == 'L':
                    d = (c[0] - distance, c[1])
                    lines[i].add((c, d))
                elif direction == 'U':
                    d = (c[0], c[1] - distance)
                    lines[i].add((c, d))
                c = d
        intersections = set()
        for a in lines[0]:
            for b in lines[1]:
                if a[0][0] == a[1][0] and b[0][1] == b[1][1]:
                    intersection = (a[0][0], b[0][1])
                    if (((b[0][0] <= intersection[0] <= b[1][0]) or 
                        (b[1][0] <= intersection[0] <= b[0][0])) and
                        ((a[0][1] <= intersection[1] <= a[1][1]) or
                         (a[1][1] <= intersection[1] <= a[0][1]))
                        ):
                        intersections.add(intersection)
                elif b[0][0] == b[1][0] and a[0][1] == a[1][1]:
                    intersection = (b[0][0], a[0][1])
                    if (((b[0][1] <= intersection[1] <= b[1][1]) or 
                        (b[1][1] <= intersection[1] <= b[0][1])) and
                        ((a[0][0] <= intersection[0] <= a[1][0]) or
                         (a[1][0] <= intersection[0] <= a[0][0]))
                        ):
                        intersections.add(intersection)
        closest_distance = None
        for intersection in intersections:
            distance = abs(intersection[0]) + abs(intersection[1])
            if closest_distance is None or distance < closest_distance:
                closest_distance = distance
        result = closest_distance
        return result
    
    def solve2(self, wires: 'List'):
        lines = [set(), set()]
        directions = {
            'R': (1, 0),
            'D': (0, 1),
            'L': (-1, 0),
            'U': (0, -1),
            }
        for i, wire in enumerate(wires):
            begin = (0, 0)
            total_distance = 0
            for move in wire:
                direction, distance = move[:1], int(move[1:])
                end = (
                    begin[0] + distance * directions[direction][0],
                    begin[1] + distance * directions[direction][1],
                    )
                lines[i].add((total_distance, begin, end))
                total_distance += distance
                begin = end
        intersections = set()
        for a in lines[0]:
            for b in lines[1]:
                if a[1][0] == a[2][0] and b[1][1] == b[2][1]:
                    intersection = ((a[1][0], b[1][1]), (a, b))
                    if (((b[1][0] <= intersection[0][0] <= b[2][0]) or 
                        (b[2][0] <= intersection[0][0] <= b[1][0])) and
                        ((a[1][1] <= intersection[0][1] <= a[2][1]) or
                         (a[2][1] <= intersection[0][1] <= a[1][1]))
                        ):
                        intersections.add(intersection)
                elif b[1][0] == b[2][0] and a[1][1] == a[2][1]:
                    intersection = ((b[1][0], a[1][1]), (a, b))
                    if (((b[1][1] <= intersection[0][1] <= b[2][1]) or 
                        (b[2][1] <= intersection[0][1] <= b[1][1])) and
                        ((a[1][0] <= intersection[0][0] <= a[2][0]) or
                         (a[2][0] <= intersection[0][0] <= a[1][0]))
                        ):
                        intersections.add(intersection)
        fewest_steps = None
        for intersection in intersections:
            point, (a, b) = intersection
            steps = a[0] + b[0]
            a_start, b_start = a[1], b[1]
            steps += abs(a_start[0] - point[0]) + abs(a_start[1] - point[1])
            steps += abs(b_start[0] - point[0]) + abs(b_start[1] - point[1])
            if fewest_steps is None or steps < fewest_steps:
                fewest_steps = steps
        result = fewest_steps
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        wires = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(wires),
            self.solve2(wires),
            )
        result = solutions
        return result

class Day02:
    '''
    1202 Program Alarm
    https://adventofcode.com/2019/day/2
    '''
    def get_parsed_input(self, raw_input_lines: List[str]) -> List[str]:
        nums = map(int, raw_input_lines[0].split(','))
        result = collections.defaultdict(int)
        for index, num in enumerate(nums):
            result[index] = num
        return result
    
    def solve(self, parsed_input: Dict[int, int]) -> int:
        vm = intcode.IntcodeVM(parsed_input)
        vm.program[1] = 12
        vm.program[2] = 2
        vm.run()
        result = vm.program[0]
        return result
    
    def solve2(self, parsed_input: Dict[int, int]) -> int:
        result = None
        for noun in range(100):
            for verb in range(100):
                vm = intcode.IntcodeVM(parsed_input)
                vm.program[1] = noun
                vm.program[2] = verb
                vm.run()
                if vm.program[0] == 19_690_720:
                    result = 100 * noun + verb
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

class Day01:
    '''
    The Tyranny of the Rocket Equation
    https://adventofcode.com/2019/day/1
    '''
    def get_parsed_input(self, raw_input_lines: 'List'):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(int(raw_input_line))
        return result
    
    def solve(self, parsed_input: 'List'):
        result = 0
        for mass in parsed_input:
            result += mass // 3 - 2
        return result
    
    def solve2(self, parsed_input: 'List'):
        result = 0
        for mass in parsed_input:
            total_fuel = 0
            current_mass = mass
            while True:
                fuel = max(0, current_mass // 3 - 2)
                total_fuel += fuel
                current_mass = fuel
                if fuel <= 0:
                    break
            result += total_fuel
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
    python AdventOfCode2019.py 1 < inputs/2019day01.in
    '''
    solvers = {
        1: (Day01, 'The Tyranny of the Rocket Equation'),
        2: (Day02, '1202 Program Alarm'),
        3: (Day03, 'Crossed Wires'),
        4: (Day04, 'Secure Container'),
        5: (Day05, 'Sunny with a Chance of Asteroids'),
        6: (Day06, 'Universal Orbit Map'),
    #     7: (Day07, 'Amplification Circuit'),
        8: (Day08, 'Space Image Format'),
    #     9: (Day09, 'Sensor Boost'),
       10: (Day10, 'Monitoring Station'),
    #    11: (Day11, 'Space Police'),
       12: (Day12, 'The N-Body Problem'),
    #    13: (Day13, 'Care Package'),
       14: (Day14, 'Space Stoichiometry'),
    #    15: (Day15, 'Oxygen System'),
       16: (Day16, 'Flawed Frequency Transmission'),
    #    17: (Day17, 'Set and Forget'),
       18: (Day18, 'Many-Worlds Interpretation'),
    #    19: (Day19, 'Tractor Beam'),
       20: (Day20, 'Donut Maze'),
    #    21: (Day21, 'Springdroid Adventure'),
       22: (Day22, 'Slam Shuffle'),
    #    23: (Day23, 'Category Six'),
    #    24: (Day24, 'Planet of Discord'),
    #    25: (Day25, 'Cryostasis'),
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
