'''
Created on Nov 24, 2020

@author: Sestren
'''
import argparse
import collections
import copy
import functools
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

class NodeRing:
    class Node:
        def __init__(self, value: int):
            self.value = value
            self.next = None

    def __init__(self, nums: List[int]):
        self.nodes = {}
        self.head = self.Node(None)
        curr_node = self.head
        for num in nums:
            curr_node.next = self.Node(num)
            curr_node = curr_node.next
            self.nodes[num] = curr_node
        curr_node.next = self.head.next
        self.head = self.head.next
        self.min_value = min(self.nodes)
        self.max_value = max(self.nodes)

    def pop3(self, node):
        curr_node = node.next
        node.next = curr_node.next.next.next
        result = curr_node
        return result

    def insert3(self, target, node):
        target_node = self.nodes[target]
        next_node = target_node.next
        target_node.next = node
        node.next.next.next = next_node

class Template: # Template
    '''
    https://adventofcode.com/2020/day/?
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

class Day25: # Combo Breaker
    '''
    Combo Breaker
    https://adventofcode.com/2020/day/25
    '''
    def get_public_keys(self, raw_input_lines: List[str]):
        public_keys = []
        for raw_input_line in raw_input_lines:
            public_keys.append(int(raw_input_line))
        result = public_keys
        return result
    
    def solve(self, public_keys):
        MODULO = 20_201_227
        loop_sizes = {}
        prev_key = 1
        for loop_size in range(1, MODULO):
            candidate_public_key = (7 * prev_key) % MODULO
            prev_key = candidate_public_key
            for i in range(len(public_keys)):
                if candidate_public_key == public_keys[i]:
                    if candidate_public_key not in loop_sizes:
                        loop_sizes[candidate_public_key] = loop_size
            if len(loop_sizes) > 1:
                break
        public_key_a = max(loop_sizes.keys())
        public_keys.remove(public_key_a)
        public_key_b = public_keys[0]
        loop_size_b = loop_sizes[public_key_b]
        encryption_key = (public_key_a ** loop_size_b) % MODULO
        result = encryption_key
        return result
    
    def solve2(self, public_keys):
        result = len(public_keys)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        public_keys = self.get_public_keys(raw_input_lines)
        solutions = (
            self.solve(public_keys),
            self.solve2(public_keys),
            )
        result = solutions
        return result

class Day24: # Lobby Layout
    '''
    Lobby Layout
    https://adventofcode.com/2020/day/24
    '''
    directions = {
        # (x, y, z) in a hex coordinate system
        'e':  (+1, -1, +0),
        'se': (+0, -1, +1),
        'sw': (-1, +0, +1),
        'w':  (-1, +1, +0),
        'nw': (+0, +1, -1),
        'ne': (+1, +0, -1),
    }

    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(raw_input_line)
        return result
    
    def get_black_tiles(self, parsed_input):
        black_tiles = set()
        for chars in parsed_input:
            tile = (0, 0, 0)
            stack = []
            for char in chars:
                if char in 'sn':
                    stack.append(char)
                else:
                    direction = ''.join(stack) + char
                    while len(stack) > 0:
                        stack.pop()
                    offset = self.directions[direction]
                    tile = (
                        tile[0] + offset[0],
                        tile[1] + offset[1],
                        tile[2] + offset[2],
                        )
            if tile in black_tiles:
                black_tiles.remove(tile)
            else:
                black_tiles.add(tile)
        result = black_tiles
        return result
    
    def solve(self, parsed_input):
        black_tiles = self.get_black_tiles(parsed_input)
        result = len(black_tiles)
        return result
    
    def visualize(self, black_tiles, distance: int=3):
        '''
        start from (0, 0, 0) and draw a number of hexes away in all directions
         . # o o o . 
        . o o o o o .
         o o o # # o 
        o # o # o # o
         o o o o # o 
        . o # o o # .
         . o o o # . 
        '''
        grid = ['']
        for z in range(-distance, distance + 1):
            row = collections.deque()
            for y in range(-distance, distance + 1):
                x = -(z + y)
                if -distance <= x <= distance:
                    cell = 'o '
                    if (x, y, z) in black_tiles:
                        cell = '# '
                    row.append(cell)
            buffer = '. ' * distance
            left_buffer: str = '' if z == 0 else buffer[-abs(z):]
            row.appendleft(left_buffer)
            right_buffer: str = left_buffer[::-1][1:]
            row.append(right_buffer)
            grid.append(''.join(row))
        result = grid
        return result
    
    def solve2(self, parsed_input, day_count: int=100):
        black_tiles = self.get_black_tiles(parsed_input)
        for _ in range(day_count):
            black_tile_neighbors = collections.defaultdict(int)
            white_tile_neighbors = collections.defaultdict(int)
            for tile in black_tiles:
                if tile not in black_tile_neighbors:
                    black_tile_neighbors[tile] = 0
                for offset in self.directions.values():
                    neighbor_tile = (
                        tile[0] + offset[0],
                        tile[1] + offset[1],
                        tile[2] + offset[2],
                        )
                    if neighbor_tile in black_tiles:
                        black_tile_neighbors[tile] += 1
                    else:
                        white_tile_neighbors[neighbor_tile] += 1
            next_black_tiles = set()
            for tile in black_tile_neighbors:
                if black_tile_neighbors[tile] in (1, 2):
                    next_black_tiles.add(tile)
            for tile in white_tile_neighbors:
                if white_tile_neighbors[tile] == 2:
                    next_black_tiles.add(tile)
            black_tiles = next_black_tiles
            # grid = self.visualize(black_tiles, 4)
            # print('\n'.join(grid))
        result = len(black_tiles)
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

class Day23: # Crab Cups
    '''
    Crab Cups
    https://adventofcode.com/2020/day/23
    '''
    def get_cups(self, raw_input_lines: List[str]):
        result = list(map(int, list(raw_input_lines[0])))
        return result
    
    def solve(self, cups):
        curr_cup = cups[0]
        for _ in range(100):
            cursor = cups.index(curr_cup)
            pickup = []
            for _ in range(3):
                if cursor + 1 < len(cups):
                    pickup.append(cups.pop(cursor + 1))
                else:
                    pickup.append(cups.pop(0))
            dest_val = 9 if curr_cup == 1 else curr_cup - 1
            while True:
                try:
                    dest_idx = cups.index(dest_val)
                    cups = cups[:dest_idx + 1] + pickup + cups[dest_idx + 1:]
                    break
                except ValueError:
                    dest_val = 9 if dest_val == 1 else dest_val - 1
            cursor = (cups.index(curr_cup) + 1) % len(cups)
            curr_cup = cups[cursor]
        index = cups.index(1)
        result = ''.join(map(str, cups[index + 1:] + cups[:index]))
        return result
    
    def solve2slow(self, cups, cup_count: int=1_000_000, rounds: int=10_000_000):
        last_cup = cups[-1] + cup_count - len(cups)
        cups += list(range(cups[-1] + 1, last_cup + 1))
        curr_cup = cups[0]
        for i in range(rounds):
            if i % 500 == 0:
                print(i)
            cursor = cups.index(curr_cup)
            pickup = []
            for _ in range(3):
                if cursor + 1 < len(cups):
                    pickup.append(cups.pop(cursor + 1))
                else:
                    pickup.append(cups.pop(0))
            dest_val = 9 if curr_cup == 1 else curr_cup - 1
            while True:
                try:
                    dest_idx = cups.index(dest_val)
                    cups = cups[:dest_idx + 1] + pickup + cups[dest_idx + 1:]
                    break
                except ValueError:
                    dest_val = 9 if dest_val == 1 else dest_val - 1
            cursor = (cups.index(curr_cup) + 1) % len(cups)
            curr_cup = cups[cursor]
        index = cups.index(1)
        cup1 = cups[(index + 1) % len(cups)]
        cup2 = cups[(index + 2) % len(cups)]
        result = cup1 * cup2
        return result
    
    def play(self, ring, curr_cup):
        pickup = ring.pop3(curr_cup)
        a, b, c = pickup.value, pickup.next.value, pickup.next.next.value
        target = curr_cup.value - 1
        while target in (a, b, c, 0):
            target -= 1
            if target < ring.min_value:
                target = ring.max_value
        ring.insert3(target, pickup)

    def solve2(self, cups, cup_count: int=1_000_000, rounds: int=10_000_000):
        cups.extend(list(range(max(cups) + 1, cup_count + 1)))
        ring = NodeRing(cups)
        curr_cup = ring.head
        for _ in range(rounds):
            self.play(ring, curr_cup)
            curr_cup = curr_cup.next
        curr_cup = ring.nodes[1]
        cup1 = curr_cup.next.value
        cup2 = curr_cup.next.next.value
        result = cup1 * cup2
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        cups = self.get_cups(raw_input_lines)
        solutions = (
            self.solve(cups[:]),
            self.solve2(cups[:]),
            )
        result = solutions
        return result

class Day22: # Crab Combat
    '''
    Crab Combat
    https://adventofcode.com/2020/day/22
    '''
    def get_decks(self, raw_input_lines: List[str]):
        decks = {}
        for raw_input_line in raw_input_lines:
            if 'Player' in raw_input_line:
                deck_id = int(raw_input_line[7])
                decks[deck_id] = collections.deque()
            elif len(raw_input_line) > 0:
                decks[deck_id].append(int(raw_input_line))
        result = decks
        return result
    
    def play(self, deck1, deck2) -> int:
        winner = -1
        while len(deck1) > 0 and len(deck2) > 0:
            card1 = deck1.popleft()
            card2 = deck2.popleft()
            if card1 > card2:
                deck1.append(card1)
                deck1.append(card2)
            elif card2 > card1:
                deck2.append(card2)
                deck2.append(card1)
        if len(deck1) > 0:
            winner = 1
        else:
            winner = 2
        result = winner
        return result
    
    def solve(self, deck1, deck2):
        score = 0
        multiplier = 1
        winner = self.play(deck1, deck2)
        if winner == 2:
            winning_deck = deck2
        else:
            winning_deck = deck1
        while len(winning_deck) > 0:
            top_card = winning_deck.pop()
            score += multiplier * top_card
            multiplier += 1
        result = score
        return result
    
    def play2(self, deck1, deck2) -> int:
        keys = set()
        winner = -1
        while len(deck1) > 0 and len(deck2) > 0:
            key = (tuple(deck1), tuple(deck2))
            if key in keys:
                # First player wins by default
                return 1
            keys.add(key)
            card1 = deck1.popleft()
            card2 = deck2.popleft()
            round_winner = -1
            if card1 <= len(deck1) and card2 <= len(deck2):
                subdeck1 = copy.deepcopy(deck1)
                while len(subdeck1) > card1:
                    subdeck1.pop()
                subdeck2 = copy.deepcopy(deck2)
                while len(subdeck2) > card2:
                    subdeck2.pop()
                round_winner = self.play2(subdeck1, subdeck2)
            elif card1 > card2:
                round_winner = 1
            elif card2 > card1:
                round_winner = 2
            if round_winner == 1:
                deck1.append(card1)
                deck1.append(card2)
            elif round_winner == 2:
                deck2.append(card2)
                deck2.append(card1)
        if len(deck1) > 0:
            winner = 1
        else:
            winner = 2
        result = winner
        return result
    
    def solve2(self, deck1, deck2):
        score = 0
        multiplier = 1
        winner = self.play2(deck1, deck2)
        if winner == 2:
            winning_deck = deck2
        else:
            winning_deck = deck1
        while len(winning_deck) > 0:
            card = winning_deck.pop()
            score += multiplier * card
            multiplier += 1
        result = score
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        decks = self.get_decks(raw_input_lines)
        solutions = (
            self.solve(copy.deepcopy(decks[1]), copy.deepcopy(decks[2])),
            self.solve2(copy.deepcopy(decks[1]), copy.deepcopy(decks[2])),
            )
        result = solutions
        return result

class Day21: # Allergen Assessment
    '''
    Allergen Assessment
    https://adventofcode.com/2020/day/21
    '''
    def get_foods(self, raw_input_lines: List[str]):
        foods = []
        for raw_input_line in raw_input_lines:
            a, b = raw_input_line.split(' (contains ')
            # Assume each ingredient appears only once per food
            food_ingredients = set(a.split(' '))
            # Assume each allergen appears only once per food
            food_allergens = set(b[:-1].split(', '))
            foods.append((food_ingredients, food_allergens))
        result = foods
        return result
    
    def identify_allergens(self, foods):
        ingredients = set()
        allergens = {}
        for food_ingredients, food_allergens in foods:
            for food_ingredient in food_ingredients:
                ingredients.add(food_ingredient)
            for food_allergen in food_allergens:
                if food_allergen not in allergens:
                    allergens[food_allergen] = set(food_ingredients)
                else:
                    current_ingredients = allergens[food_allergen]
                    allergens[food_allergen] = set.intersection(
                        current_ingredients,
                        food_ingredients,
                        )
        identified = set()
        identities = {}
        while len(identified) < len(allergens):
            for allergen, possible_ingredients in allergens.items():
                if len(possible_ingredients) == 1:
                    ingredient = next(iter(possible_ingredients))
                    identified.add(ingredient)
                    identities[ingredient] = allergen
                elif len(possible_ingredients) > 1:
                    possible_ingredients -= identified
        result = identities
        return result
    
    def solve(self, foods):
        identities = self.identify_allergens(foods)
        safe_ingredient_count = 0
        for food_ingredients, _ in foods:
             for food_ingredient in food_ingredients:
                if food_ingredient not in identities:
                    safe_ingredient_count += 1
        result = safe_ingredient_count
        return result
    
    def solve2(self, foods):
        identities = self.identify_allergens(foods)
        dangerous_ingredients = sorted(identities.keys(), key=identities.get)
        result = ','.join(dangerous_ingredients)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        foods = self.get_foods(raw_input_lines)
        solutions = (
            self.solve(foods),
            self.solve2(foods),
            )
        result = solutions
        return result

class Day20: # Jurassic Jigsaw
    '''
    Jurassic Jigsaw
    https://adventofcode.com/2020/day/20
    '''
    def get_tiles(self, raw_input_lines: List[str]):
        tiles = {}
        tile_id = None
        tile = []
        for raw_input_line in raw_input_lines:
            if len(raw_input_line) < 1:
                tiles[tile_id] = tile
                tile = []
                continue
            elif 'Tile' in raw_input_line:
                tile_id = int(raw_input_line.split(' ')[1][:-1])
            else:
                tile.append(raw_input_line)
        tiles[tile_id] = tile
        result = tiles
        return result
    
    def get_edges(self, tile: List[str]) -> List[str]:
        rows = len(tile)
        cols = len(tile[0])
        top = tile[0]
        bottom = tile[-1]
        left = ''.join(tile[i][0] for i in range(rows))
        right = ''.join(tile[i][cols - 1] for i in range(rows))
        result = [top, bottom, left, right]
        return result
    
    def get_connections(self, tiles):
        connections = {}
        for tile_id, tile in tiles.items():
            connections[tile_id] = set()
            edges = self.get_edges(tile)
            for other_tile_id, other_tile in tiles.items():
                if other_tile_id == tile_id:
                    continue
                connected_ind = False
                other_edges = self.get_edges(other_tile)
                for edge in edges:
                    for other_edge in other_edges:
                        if edge in (other_edge, other_edge[::-1]):
                            connected_ind = True
                            break
                    if connected_ind:
                        break
                if connected_ind:
                    connections[tile_id].add(other_tile_id)
        result = connections
        return result
    
    # Consider caching this if it's too slow using functools.lru_cache
    # Use tile_id instead of tile, if you do
    def get_rotations(self, tile):
        rows = len(tile)
        cols = len(tile[0])
        assert rows == cols
        rotations =[
            list(['+' for col in range(cols)] for row in range(rows)),
            list(['+' for col in range(cols)] for row in range(rows)),
            list(['+' for col in range(cols)] for row in range(rows)),
            list(['+' for col in range(cols)] for row in range(rows)),
            list(['+' for col in range(cols)] for row in range(rows)),
            list(['+' for col in range(cols)] for row in range(rows)),
            list(['+' for col in range(cols)] for row in range(rows)),
            list(['+' for col in range(cols)] for row in range(rows)),
        ]
        for row in range(rows):
            for col in range(cols):
                # 0: rows go top-to-bottom
                #    cols go left-to-right
                rotations[0][row][col] = tile[row][col]
                # 1: rows go top-to-bottom
                #    cols go right-to-left
                rotations[1][row][cols - col - 1] = tile[row][col]
                # 2: rows go bottom-to-top
                #    cols go left-to-right
                rotations[2][rows - row - 1][col] = tile[row][col]
                # 3: rows go bottom-to-top
                #    cols go right-to-left
                rotations[3][rows - row - 1][cols - col - 1] = tile[row][col]
                # 4: rows go left-to-right
                #    cols go top-to-bottom
                rotations[4][col][row] = tile[row][col]
                # 5: rows go left-to-right
                #    cols go bottom-to-top
                rotations[5][col][rows - row - 1] = tile[row][col]
                # 6: rows go right-to-left
                #    cols go top-to-bottom
                rotations[6][cols - col - 1][row] = tile[row][col]
                # 7: rows go right-to-left
                #    cols go bottom-to-top
                rotations[7][cols - col - 1][rows - row - 1] = tile[row][col]
        for i in range(8):
            for row in range(rows):
                rotations[i][row] = ''.join(rotations[i][row])
        result = rotations
        return result
    
    def solve(self, tiles):
        connections = self.get_connections(tiles)
        # Based on the instructions, a corner tile is assumed to be a tile that
        # connects with exactly two other tiles.
        corner_tiles = []
        for tile_id, other_tile_ids in connections.items():
            if len(other_tile_ids) == 2:
                corner_tiles.append(tile_id)
        result = 1
        for tile_id in corner_tiles:
            result *= tile_id
        return result
    
    def solve2(self, tiles):
        # grab an arbitrary tile to start placing, call that the "center"
        placed_tiles = {} # (row, col) : reoriented_tile
        tiles_left = set(tiles.keys())
        starting_tile_id = next(iter(tiles_left))
        starting_tile = tiles[starting_tile_id]
        seen = set()
        work = [(0, 0, starting_tile_id, starting_tile)]
        while len(tiles_left) > 0 and len(work) > 0:
            tile_row, tile_col, tile_id, tile = work.pop()
            placed_tiles[(tile_row, tile_col)] = tile
            tiles_left.remove(tile_id)
            edges = self.get_edges(tile) # [top, bottom, left, right]
            # check for neighboring tiles that fit an edge
            for next_tile_row, next_tile_col, edge, next_edge_id in (
                (tile_row - 1, tile_col    , edges[0], 1),
                (tile_row + 1, tile_col    , edges[1], 0),
                (tile_row    , tile_col - 1, edges[2], 3),
                (tile_row    , tile_col + 1, edges[3], 2),
                ):
                for next_tile_id in tiles_left:
                    next_tile = tiles[next_tile_id]
                    match_found_ind = False
                    # check all possible rotations of neighboring tile
                    for rotated_tile in self.get_rotations(next_tile):
                        next_edge = self.get_edges(rotated_tile)[next_edge_id]
                        if next_edge == edge and next_tile_id not in seen:
                            work.append((
                                next_tile_row,
                                next_tile_col,
                                next_tile_id,
                                rotated_tile,
                                ))
                            seen.add(next_tile_id)
                            match_found_ind = True
                            break
                    if match_found_ind:
                        break
        assert len(tiles_left) == 0
        assert len(work) == 0
        # assemble the placed tiles into a larger image grid
        image = []
        min_row, min_col = min(placed_tiles.keys())
        max_row, max_col = max(placed_tiles.keys())
        rows = len(starting_tile)
        for tile_row in range(min_row, max_row + 1):
            for row in range(1, rows - 1):
                line = []
                for tile_col in range(min_col, max_col + 1):
                    tile = placed_tiles[(tile_row, tile_col)]
                    line.append(''.join(tile[row][1:-1]))
                image.append(''.join(''.join(line)))
        # Calculate water roughness
        SEA_SERPENT = [
            '                  # ',
            '#    ##    ##    ###',
            ' #  #  #  #  #  #   ',
            ]
        SERPENT_HEIGHT = len(SEA_SERPENT)
        SERPENT_WIDTH = len(SEA_SERPENT[0])
        MONSTER_WEIGHT = 15
        findings = []
        # analyze the image in 8 different orientations
        for rotated_image in self.get_rotations(image):
            findings.append([0, 0])
            rows = len(rotated_image)
            cols = len(rotated_image[0])
            for row in range(rows):
                for col in range(cols):
                    if rotated_image[row][col] == '#':
                        findings[-1][0] += 1
                    if (
                        row >= rows - SERPENT_HEIGHT or
                        col >= cols - SERPENT_WIDTH
                        ):
                        continue
                    falsified_ind = False
                    for srow in range(SERPENT_HEIGHT):
                        for scol in range(SERPENT_WIDTH):
                            if (
                                SEA_SERPENT[srow][scol] == '#' and
                                rotated_image[row + srow][col + scol] != '#'
                                ):
                                falsified_ind = True
                                break
                        if falsified_ind:
                            break
                    if not falsified_ind:
                        findings[-1][1] += 1
        finding = [0, 0]
        for i in range(len(findings)):
            if findings[i][1] > 0:
                finding = findings[i]
                break
        water_roughness = finding[0] - MONSTER_WEIGHT * finding[1]
        result = water_roughness
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        tiles = self.get_tiles(raw_input_lines)
        solutions = (
            self.solve(tiles),
            self.solve2(tiles),
            )
        result = solutions
        return result

class Day19: # Monster Messages
    '''
    Monster Messages
    https://adventofcode.com/2020/day/19
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        rules = {}
        messages = []
        for raw_input_line in raw_input_lines:
            if ':' in raw_input_line:
                rule_id, raw_rule = raw_input_line.split(': ')
                rule_id = int(rule_id)
                if '"' in raw_rule:
                    rules[rule_id] = [raw_rule[1]]
                else:
                    rule = []
                    for raw_elem in raw_rule.split(' | '):
                        rule.append(list(map(int, raw_elem.split(' '))))
                    rules[rule_id] = rule
            elif len(raw_input_line) > 0:
                messages.append(raw_input_line)
        result = rules, messages
        return result
    
    def get_matches(self, message, rules, rule_index, start=0):
        matches = []
        for attempt in rules[rule_index]:
            cursors = [start]
            for val in attempt:
                next_cursors = []
                for cursor in cursors:
                    if type(val) == str:
                        if start < len(message) and message[start] == val:
                            next_cursors.append(cursor + 1)
                    else:
                        next_matches = self.get_matches(
                            message,
                            rules,
                            val,
                            cursor,
                            )
                        for match in next_matches:
                            next_cursors.append(match)
                cursors = next_cursors
            for cursor in cursors:
                matches.append(cursor)
        result = matches
        return result
    
    def solve(self, rules, messages):
        valid_message_count = 0
        for message in messages:
            matches = self.get_matches(message, rules, 0, 0)
            if len(message) in matches:
                valid_message_count += 1
        result = valid_message_count
        return result
    
    def solve2(self, rules, messages):
        rules[8] = [[42], [42, 8]]
        rules[11] = [[42, 31], [42, 11, 31]]
        valid_message_count = 0
        for message in messages:
            matches = self.get_matches(message, rules, 0, 0)
            if len(message) in matches:
                valid_message_count += 1
        result = valid_message_count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        rules, messages = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(rules, messages),
            self.solve2(rules, messages),
            )
        result = solutions
        return result

class Day18: # Operation Order
    '''
    Operation Order
    https://adventofcode.com/2020/day/18
    '''
    def tokenize(self, chars):
        tokens = []
        num = 0
        for i, char in enumerate(chars + ' '):
            if char not in '0123456789':
                if i > 0 and chars[i - 1] in '0123456789':
                    tokens.append(num)
                    num = 0
            if char == ' ':
                continue
            elif char in '0123456789':
                num = 10 * num + int(char)
            else:
                tokens.append(char)
        result = tokens
        return result

    def get_parsed_input(self, raw_input_lines):
        result = []
        for raw_input_line in raw_input_lines:
            result.append(self.tokenize(raw_input_line))
        return result
    
    def solve(self, parsed_input):
        # Use Shunting-Yard Algorithm
        # https://en.wikipedia.org/wiki/Shunting-yard_algorithm
        totals = []
        for tokens in parsed_input:
            output = []
            operators = []
            for token in tokens:
                if type(token) is int:
                    output.append(token)
                elif token in '+*':
                    while (
                        len(operators) > 0 and
                        operators[-1] + token in ('++', '**', '*+', '+*')
                        ):
                        output.append(operators.pop())
                    operators.append(token)
                elif token == '(':
                    operators.append(token)
                elif token == ')':
                    while operators[-1] != '(':
                        output.append(operators.pop())
                    if operators[-1] == '(':
                        operators.pop()
            while len(operators) > 0:
                output.append(operators.pop())
            stack = []
            for el in output:
                stack.append(el)
                if type(stack[-1]) is str and stack[-1] in '+*':
                    operator = stack.pop()
                    a = stack.pop()
                    b = stack.pop()
                    if operator == '+':
                        stack.append(a + b)
                    elif operator == '*':
                        stack.append(a * b)
            totals.append(sum(stack))
        result = sum(totals)
        return result
    
    def solve2(self, parsed_input):
        # Use Shunting-Yard Algorithm
        # https://en.wikipedia.org/wiki/Shunting-yard_algorithm
        totals = []
        for tokens in parsed_input:
            output = []
            operators = []
            for token in tokens:
                if type(token) is int:
                    output.append(token)
                elif token in '+*':
                    while (
                        len(operators) > 0 and
                        operators[-1] + token in ('++', '**', '+*')
                        ):
                        output.append(operators.pop())
                    operators.append(token)
                elif token == '(':
                    operators.append(token)
                elif token == ')':
                    while operators[-1] != '(':
                        output.append(operators.pop())
                    if operators[-1] == '(':
                        operators.pop()
            while len(operators) > 0:
                output.append(operators.pop())
            stack = []
            for el in output:
                stack.append(el)
                if type(stack[-1]) is str and stack[-1] in '+*':
                    operator = stack.pop()
                    a = stack.pop()
                    b = stack.pop()
                    if operator == '+':
                        stack.append(a + b)
                    elif operator == '*':
                        stack.append(a * b)
            totals.append(sum(stack))
        result = sum(totals)
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

class Day17: # Conway Cubes
    '''
    Conway Cubes
    https://adventofcode.com/2020/day/17
    '''
    def get_initial_state(self, raw_input_lines: List[str]):
        initial_state = set()
        for y, raw_input_line in enumerate(raw_input_lines):
            for x, cell in enumerate(raw_input_line):
                if cell == '#':
                    initial_state.add((x, y, 0))
        result = initial_state
        return result
    
    def solve(self, initial_state):
        curr_state = set(initial_state)
        for _ in range(6):
            neighbors = collections.defaultdict(int)
            for x, y, z in curr_state:
                for (dx, dy, dz) in (
                    (-1, -1, -1),
                    (-1, -1,  0),
                    (-1, -1,  1),
                    (-1,  0, -1),
                    (-1,  0,  0),
                    (-1,  0,  1),
                    (-1,  1, -1),
                    (-1,  1,  0),
                    (-1,  1,  1),
                    ( 0, -1, -1),
                    ( 0, -1,  0),
                    ( 0, -1,  1),
                    ( 0,  0, -1),
                    # ( 0,  0,  0),
                    ( 0,  0,  1),
                    ( 0,  1, -1),
                    ( 0,  1,  0),
                    ( 0,  1,  1),
                    ( 1, -1, -1),
                    ( 1, -1,  0),
                    ( 1, -1,  1),
                    ( 1,  0, -1),
                    ( 1,  0,  0),
                    ( 1,  0,  1),
                    ( 1,  1, -1),
                    ( 1,  1,  0),
                    ( 1,  1,  1),
                    ):
                    neighbors[(x + dx, y + dy, z + dz)] += 1
            next_state = set()
            for x, y, z in curr_state:
                if 2 <= neighbors[(x, y, z)] <= 3:
                    next_state.add((x, y, z))
            for x, y, z in neighbors:
                if neighbors[(x, y, z)] == 3 and (x, y, z) not in curr_state:
                    next_state.add((x, y, z))
            curr_state = next_state
        result = len(curr_state)
        return result
    
    def solve2(self, initial_state):
        curr_state = set()
        for (x, y, z) in initial_state:
            curr_state.add((x, y, z, 0))
        for _ in range(6):
            neighbors = collections.defaultdict(int)
            for x, y, z, w in curr_state:
                for (dx, dy, dz, dw) in (
                    (-1, -1, -1, -1),
                    (-1, -1,  0, -1),
                    (-1, -1,  1, -1),
                    (-1,  0, -1, -1),
                    (-1,  0,  0, -1),
                    (-1,  0,  1, -1),
                    (-1,  1, -1, -1),
                    (-1,  1,  0, -1),
                    (-1,  1,  1, -1),
                    ( 0, -1, -1, -1),
                    ( 0, -1,  0, -1),
                    ( 0, -1,  1, -1),
                    ( 0,  0, -1, -1),
                    ( 0,  0,  0, -1),
                    ( 0,  0,  1, -1),
                    ( 0,  1, -1, -1),
                    ( 0,  1,  0, -1),
                    ( 0,  1,  1, -1),
                    ( 1, -1, -1, -1),
                    ( 1, -1,  0, -1),
                    ( 1, -1,  1, -1),
                    ( 1,  0, -1, -1),
                    ( 1,  0,  0, -1),
                    ( 1,  0,  1, -1),
                    ( 1,  1, -1, -1),
                    ( 1,  1,  0, -1),
                    ( 1,  1,  1, -1),
                    
                    (-1, -1, -1,  0),
                    (-1, -1,  0,  0),
                    (-1, -1,  1,  0),
                    (-1,  0, -1,  0),
                    (-1,  0,  0,  0),
                    (-1,  0,  1,  0),
                    (-1,  1, -1,  0),
                    (-1,  1,  0,  0),
                    (-1,  1,  1,  0),
                    ( 0, -1, -1,  0),
                    ( 0, -1,  0,  0),
                    ( 0, -1,  1,  0),
                    ( 0,  0, -1,  0),
                    # ( 0,  0,  0,  0),
                    ( 0,  0,  1,  0),
                    ( 0,  1, -1,  0),
                    ( 0,  1,  0,  0),
                    ( 0,  1,  1,  0),
                    ( 1, -1, -1,  0),
                    ( 1, -1,  0,  0),
                    ( 1, -1,  1,  0),
                    ( 1,  0, -1,  0),
                    ( 1,  0,  0,  0),
                    ( 1,  0,  1,  0),
                    ( 1,  1, -1,  0),
                    ( 1,  1,  0,  0),
                    ( 1,  1,  1,  0),
                    
                    (-1, -1, -1,  1),
                    (-1, -1,  0,  1),
                    (-1, -1,  1,  1),
                    (-1,  0, -1,  1),
                    (-1,  0,  0,  1),
                    (-1,  0,  1,  1),
                    (-1,  1, -1,  1),
                    (-1,  1,  0,  1),
                    (-1,  1,  1,  1),
                    ( 0, -1, -1,  1),
                    ( 0, -1,  0,  1),
                    ( 0, -1,  1,  1),
                    ( 0,  0, -1,  1),
                    ( 0,  0,  0,  1),
                    ( 0,  0,  1,  1),
                    ( 0,  1, -1,  1),
                    ( 0,  1,  0,  1),
                    ( 0,  1,  1,  1),
                    ( 1, -1, -1,  1),
                    ( 1, -1,  0,  1),
                    ( 1, -1,  1,  1),
                    ( 1,  0, -1,  1),
                    ( 1,  0,  0,  1),
                    ( 1,  0,  1,  1),
                    ( 1,  1, -1,  1),
                    ( 1,  1,  0,  1),
                    ( 1,  1,  1,  1),
                    ):
                    neighbors[(x + dx, y + dy, z + dz, w + dw)] += 1
            next_state = set()
            for x, y, z, w in curr_state:
                if 2 <= neighbors[(x, y, z, w)] <= 3:
                    next_state.add((x, y, z, w))
            for x, y, z, w in neighbors:
                if (
                    neighbors[(x, y, z, w)] == 3 and
                    (x, y, z, w) not in curr_state
                ):
                    next_state.add((x, y, z, w))
            curr_state = next_state
        result = len(curr_state)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        initial_state = self.get_initial_state(raw_input_lines)
        solutions = (
            self.solve(initial_state),
            self.solve2(initial_state),
            )
        result = solutions
        return result

class Day16: # Ticket Translation
    '''
    Ticket Translation
    https://adventofcode.com/2020/day/16
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        rules = {}
        your_ticket = []
        nearby_tickets = []
        mode = 'rules'
        for raw_input_line in raw_input_lines:
            if ': ' in raw_input_line:
                field, raw_suffix = raw_input_line.split(': ')
                rules[field] = []
                valid_ranges = raw_suffix.split(' or ')
                for valid_range in valid_ranges:
                    rule = tuple(map(int, valid_range.split('-')))
                    rules[field].append(rule)
            else:
                if len(raw_input_line) == 0:
                    continue
                if raw_input_line in (
                    'your ticket:',
                    'nearby tickets:',
                    ):
                    mode = raw_input_line[:-1]
                elif mode == 'your ticket':
                    your_ticket = list(map(int, raw_input_line.split(',')))
                elif mode == 'nearby tickets':
                    nearby_tickets.append(
                        list(map(int, raw_input_line.split(',')))
                        )
        result = (
            rules,
            your_ticket,
            nearby_tickets,
        )
        return result
    
    def ticket_errors(self,
        ticket: List[int],
        rules: Dict[str, List[Tuple[int]]],
        ):
        errors = []
        for value in ticket:
            valid_ind = False
            for valid_ranges in rules.values():
                for valid_range in valid_ranges:
                    min_val, max_val = valid_range
                    if min_val <= value <= max_val:
                        valid_ind = True
                        break
                if valid_ind:
                    break
            if not valid_ind:
                errors.append(value)
        result = errors
        return result
    
    def solve(self,
        rules: Dict[str, List[Tuple[int]]],
        nearby_tickets: List[List[int]],
        ):
        errors = []
        for ticket in nearby_tickets:
            errors.extend(self.ticket_errors(ticket, rules))
        result = sum(errors)
        return result
    
    def solve2(self,
        rules: Dict[str, List[Tuple[int]]],
        your_ticket: List[int],
        nearby_tickets: List[List[int]],
        ):
        valid_tickets = list(
            ticket for
            ticket in nearby_tickets if
            len(self.ticket_errors(ticket, rules)) == 0
            )
        fields = {}
        for field in rules:
            fields[field] = set(range(len(your_ticket)))
        for field in fields:
            valid_ranges = rules[field]
            for field_id in range(len(your_ticket)):
                possible_ind = True
                for ticket in valid_tickets:
                    valid_ind = False
                    for valid_range in valid_ranges:
                        min_val, max_val = valid_range
                        if min_val <= ticket[field_id] <= max_val:
                            valid_ind = True
                            break
                    if not valid_ind:
                        possible_ind = False
                        break
                if not possible_ind:
                    fields[field].remove(field_id)
        fixed_fields = {}
        while len(fixed_fields) < len(your_ticket):
            fixed_field_id = -1
            for field, possible_ids in fields.items():
                if len(possible_ids) == 1:
                    fixed_field_id = next(iter(possible_ids))
                    break
            if fixed_field_id >= 0:
                fixed_fields[field] = fixed_field_id
                for field in fields:
                    if fixed_field_id in fields[field]:
                        fields[field].remove(fixed_field_id)
        result = functools.reduce(operator.mul,(
            your_ticket[fixed_fields['departure location']],
            your_ticket[fixed_fields['departure station']],
            your_ticket[fixed_fields['departure platform']],
            your_ticket[fixed_fields['departure track']],
            your_ticket[fixed_fields['departure date']],
            your_ticket[fixed_fields['departure time']],
            ), 1)
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        parsed_input = self.get_parsed_input(raw_input_lines)
        rules, your_ticket, nearby_tickets = parsed_input
        solutions = (
            self.solve(rules, nearby_tickets),
            self.solve2(rules, your_ticket, nearby_tickets),
            )
        result = solutions
        return result

class Day15: # Rambunctious Recitation
    '''
    Rambunctious Recitation
    https://adventofcode.com/2020/day/15
    '''
    def get_starting_numbers(self, raw_input_lines: List[str]):
        result = list(map(int, raw_input_lines[0].split(',')))
        return result
    
    def bruteforce(self, starting_numbers, turns):
        spoken = starting_numbers[0]
        prev_spoken = None
        last_spoken = collections.defaultdict(collections.deque)
        for i in range(turns):
            if i < len(starting_numbers):
                spoken = starting_numbers[i]
            else:
                if len(last_spoken[prev_spoken]) == 1:
                    spoken = 0
                else:
                    times = last_spoken[prev_spoken]
                    spoken = times[-1] - times[-2]
            last_spoken[spoken].append(i)
            while len(last_spoken[spoken]) > 2:
                last_spoken[spoken].popleft()
            prev_spoken = spoken
        result = spoken
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        starting_numbers = self.get_starting_numbers(raw_input_lines)
        solutions = (
            self.bruteforce(starting_numbers, 2_020),
            self.bruteforce(starting_numbers, 30_000_000),
            )
        result = solutions
        return result

class Day14: # Docking Data
    '''
    Docking Data
    https://adventofcode.com/2020/day/14
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        result = []
        for raw_input_line in raw_input_lines:
            a, b = raw_input_line.split(' = ')
            if a == 'mask':
                result.append((a, b))
            else:
                c = a.split('[')[1][:-1]
                result.append(('mem', int(c), int(b)))
        return result
    
    def solve(self, parsed_input):
        mem = collections.defaultdict(int)
        mask = 'X' * 36
        for row in parsed_input:
            if row[0] == 'mask':
                mask = row[1]
            elif row[0] == 'mem':
                num = row[2]
                value = 0
                for i in range(36):
                    power = 2 ** (36 - i - 1)
                    if mask[i] == '1' or mask[i] == 'X' and num & power > 0:
                        value += power
                mem[row[1]] = value
        result = sum(mem.values())
        return result

    def gen_addresses(self, mask):
        if 'X' in mask:
            for char in ('0', '1'):
                yield from self.gen_addresses(mask.replace('X', char, 1))
        else:
            yield mask
    
    def solve2(self, parsed_input):
        mem = collections.defaultdict(int)
        mask = 'X' * 36
        for row in parsed_input:
            if row[0] == 'mask':
                mask = row[1]
            elif row[0] == 'mem':
                masked_address = []
                for i in range(len(mask)):
                    if mask[i] == '0':
                        power = 2 ** (35 - i)
                        bit = 1 if row[1] & power > 0 else 0
                        masked_address.append(str(bit))
                    else:
                        masked_address.append(mask[i])
                masked_address = ''.join(masked_address)
                for address_str in self.gen_addresses(masked_address):
                    address = int(address_str, 2)
                    mem[address] = row[2]
        result = sum(mem.values())
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

class Day13: # Shuttle Search
    '''
    Shuttle Search
    https://adventofcode.com/2020/day/13
    '''
    def get_parsed_input(self, raw_input_lines: List[str]):
        earliest_departure_time = int(raw_input_lines[0])
        buses = set()
        for i, bus_id in enumerate(raw_input_lines[1].split(',')):
            if bus_id != 'x':
                bus_id = int(bus_id)
                buses.add((bus_id, (bus_id - (i % bus_id)) % bus_id))
        result = (earliest_departure_time, buses)
        return result
    
    def solve(self, earliest_departure_time, buses):
        min_wait_time = float('inf')
        earliest_bus_id = 0
        for bus_id, _ in buses:
            wait_time = bus_id - (earliest_departure_time % bus_id)
            if wait_time < min_wait_time:
                min_wait_time = wait_time
                earliest_bus_id = bus_id
        result = earliest_bus_id * min_wait_time
        return result
    
    def solve2(self, earliest_departure_time, buses):
        modulo, remainder = buses.pop()
        while len(buses) > 0:
            next_modulo, next_remainder = buses.pop()
            while remainder % next_modulo != next_remainder:
                remainder += modulo
            modulo *= next_modulo
        result = remainder
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        earliest_departure_time, buses = self.get_parsed_input(raw_input_lines)
        solutions = (
            self.solve(earliest_departure_time, buses),
            self.solve2(earliest_departure_time, buses),
            )
        result = solutions
        return result

class Day12: # Rain Risk
    '''
    Rain Risk
    https://adventofcode.com/2020/day/12
    '''
    def get_instructions(self, raw_input_lines: List[str]):
        instructions = []
        for raw_input_line in raw_input_lines:
            instruction = (raw_input_line[0], int(raw_input_line[1:]))
            instructions.append(instruction)
        result = instructions
        return result
    
    def solve(self, instructions):
        facings = ['N', 'E', 'S', 'W']
        facing = 1 # ship
        offsets = {
            'N': (-1, 0),
            'S': ( 1, 0),
            'W': ( 0,-1),
            'E': ( 0, 1),
        }
        position = (0, 0) # NS, WE
        for instruction, amount in instructions:
            offset = None
            if instruction == 'F':
                offset = offsets[facings[facing]]
            elif instruction == 'L':
                facing = (facing - amount // 90) % len(facings)
            elif instruction == 'R':
                facing = (facing + amount // 90) % len(facings)
            elif instruction in offsets:
                offset = offsets[instruction]
            if offset is not None:
                position = (
                    position[0] + amount * offset[0],
                    position[1] + amount * offset[1],
                )
        result = sum(map(abs, position))
        return result
    
    def solve2(self, instructions):
        offsets = {
            'N': (-1, 0),
            'S': ( 1, 0),
            'W': ( 0,-1),
            'E': ( 0, 1),
        }
        waypoint = (-1, 10) # NS, WE
        ship = (0, 0) # NS, WE
        for instruction, amount in instructions:
            if instruction == 'F':
                ship = (
                    ship[0] + amount * waypoint[0],
                    ship[1] + amount * waypoint[1],
                )
            elif instruction == 'L':
                rotation_count = amount // 90
                for _ in range(rotation_count):
                    waypoint = (-waypoint[1], waypoint[0])
            elif instruction == 'R':
                rotation_count = amount // 90
                for _ in range(rotation_count):
                    waypoint = (waypoint[1], -waypoint[0])
            elif instruction in offsets:
                offset = offsets[instruction]
                waypoint = (
                    waypoint[0] + amount * offset[0],
                    waypoint[1] + amount * offset[1],
                )
        result = sum(map(abs, ship))
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

class Day11: # Seating System
    '''
    Seating System
    https://adventofcode.com/2020/day/11
    '''
    def get_seats(self, raw_input_lines: List[List[str]]):
        result = []
        for raw_input_line in raw_input_lines:
            rowdata = []
            for seat in raw_input_line:
                rowdata.append(seat)
            result.append(rowdata)
        return result
    
    def solve(self, seats):
        rows = len(seats)
        cols = len(seats[0])
        seats = [row[:] for row in seats]
        next_seats = [row[:] for row in seats]
        change_ind = True
        while change_ind:
            change_ind = False
            for row in range(rows):
                for col in range(cols):
                    occupied_count = 0
                    for (r, c) in (
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
                            0 <= r < rows and
                            0 <= c < cols and
                            seats[r][c] == '#'
                        ):
                            occupied_count += 1
                    if seats[row][col] == 'L':
                        if occupied_count == 0:
                            next_seats[row][col] = '#'
                            change_ind = True
                    elif seats[row][col] == '#':
                        if occupied_count >= 4:
                            next_seats[row][col] = 'L'
                            change_ind = True
            for row in range(rows):
                for col in range(cols):
                    seats[row][col] = next_seats[row][col]
        occupied_count = 0
        for row in range(rows):
            for col in range(cols):
                if seats[row][col] == '#':
                    occupied_count += 1
        result = occupied_count
        return result
    
    def solve2(self, seats):
        rows = len(seats)
        cols = len(seats[0])
        seats = [row[:] for row in seats]
        next_seats = [row[:] for row in seats]
        change_ind = True
        while change_ind:
            change_ind = False
            for row in range(rows):
                for col in range(cols):
                    occupied_count = 0
                    for (dr, dc) in (
                        (-1, -1),
                        (-1,  0),
                        (-1,  1),
                        ( 0, -1),
                        ( 0,  1),
                        ( 1, -1),
                        ( 1,  0),
                        ( 1,  1),
                    ):
                        dist = 1
                        while True:
                            if (
                                (row + dist * dr) < 0 or
                                (row + dist * dr) >= rows or
                                (col + dist * dc) < 0 or
                                (col + dist * dc) >= cols
                            ):
                                break
                            seat = seats[row + dist * dr][col + dist * dc]
                            if seat == '#':
                                occupied_count += 1
                                break
                            elif seat == 'L':
                                break
                            dist += 1
                    if seats[row][col] == 'L':
                        if occupied_count == 0:
                            next_seats[row][col] = '#'
                            change_ind = True
                    elif seats[row][col] == '#':
                        if occupied_count >= 5:
                            next_seats[row][col] = 'L'
                            change_ind = True
            for row in range(rows):
                for col in range(cols):
                    seats[row][col] = next_seats[row][col]
        occupied_count = 0
        for row in range(rows):
            for col in range(cols):
                if seats[row][col] == '#':
                    occupied_count += 1
        result = occupied_count
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        seats = self.get_seats(raw_input_lines)
        solutions = (
            self.solve(seats),
            self.solve2(seats),
            )
        result = solutions
        return result

class Day10: # Adapter Array
    '''
    https://adventofcode.com/2020/day/10
    '''
    def get_adapters(self, raw_input_lines: List[str]) -> Set[int]:
        result = set()
        for raw_input_line in raw_input_lines:
            result.add(int(raw_input_line))
        return result
    
    def solve(self, adapters: Set[int]) -> int:
        adapters_left = set(adapters)
        chain = [0]
        while len(adapters_left) > 0:
            adapter = min(adapters_left)
            adapters_left.remove(adapter)
            chain.append(adapter)
        chain.append(chain[-1] + 3)
        diffs = [0, 0, 0, 0]
        for i in range(1, len(chain)):
            diff = chain[i] - chain[i - 1]
            assert 1 <= diff <= 3
            diffs[diff] += 1
        # sum of 1-jolt differences multiplied by sum of 3-jolt differences
        result = diffs[1] * diffs[3]
        return result
    
    def solve2(self, adapters: Set[int]) -> int:
        dp = [1] + [0] * max(adapters)
        for adapter in sorted(adapters):
            if adapter >= 1:
                dp[adapter] += dp[adapter - 1]
            if adapter >= 2:
                dp[adapter] += dp[adapter - 2]
            if adapter >= 3:
                dp[adapter] += dp[adapter - 3]
        result = dp[-1]
        return result
    
    def main(self):
        raw_input_lines = get_raw_input_lines()
        adapters = self.get_adapters(raw_input_lines)
        solutions = (
            self.solve(adapters),
            self.solve2(adapters),
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
    python AdventOfCode2020.py 25 < 2020day25.in
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
       10: (Day10, 'Adapter Array'),
       11: (Day11, 'Seating System'),
       12: (Day12, 'Rain Risk'),
       13: (Day13, 'Shuttle Search'),
       14: (Day14, 'Docking Data'),
       15: (Day15, 'Rambunctious Recitation'),
       16: (Day16, 'Ticket Translation'),
       17: (Day17, 'Conway Cubes'),
       18: (Day18, 'Operation Order'),
       19: (Day19, 'Monster Messages'),
       20: (Day20, 'Jurassic Jigsaw'),
       21: (Day21, 'Allergen Assessment'),
       22: (Day22, 'Crab Combat'),
       23: (Day23, 'Crab Cups'),
       24: (Day24, 'Lobby Layout'),
       25: (Day25, 'Combo Breaker'),
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
