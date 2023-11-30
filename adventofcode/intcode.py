'''
Created on Nov 28, 2023

@author: Sestren
'''

import collections
import copy
from typing import Dict

class IntcodeVM:
    def __init__(self, program: Dict[int, int]):
        self.debug = False
        self.program = copy.deepcopy(program)
        self.pc = 0
        self.inputs = collections.deque()
        self.outputs = collections.deque()
        self.status = 'ACTIVE'
    
    def log(self, message):
        if self.debug:
            print(message)
    
    def get_next_output(self):
        result = None
        if len(self.outputs) > 0:
            result = self.outputs.popleft()
        return result
    
    def send_input(self, value):
        self.inputs.appendleft(value)

    def get_address(self, position: int, mode: int):
        assert mode in (0, 1)
        if mode == 0:
            return self.program[position]
        elif mode == 1:
            return position

    def run(self):
        while True:
            opcode = self.program[self.pc] % 100
            modes = [
                self.program[self.pc] // 100 % 10,
                self.program[self.pc] // 1000 % 10,
                self.program[self.pc] // 10000 % 10,
            ]
            if opcode == 1: # ADD
                addr_a = self.get_address(self.pc + 1, modes[0])
                addr_b = self.get_address(self.pc + 2, modes[1])
                addr_c = self.get_address(self.pc + 3, modes[2])
                a = self.program[addr_a]
                b = self.program[addr_b]
                self.program[addr_c] = a + b
                self.pc += 4
            elif opcode == 2: # MULTIPLY
                addr_a = self.get_address(self.pc + 1, modes[0])
                addr_b = self.get_address(self.pc + 2, modes[1])
                addr_c = self.get_address(self.pc + 3, modes[2])
                a = self.program[addr_a]
                b = self.program[addr_b]
                self.program[addr_c] = a * b
                self.pc += 4
            elif opcode == 3: # RECEIVE INPUT
                if len(self.inputs) > 0:
                    addr_a = self.get_address(self.pc + 1, modes[0])
                    curr_input = self.inputs.pop()
                    self.program[addr_a] = curr_input
                    self.last_input = curr_input
                    self.log(f'{self.pc}: RCV {curr_input} --> [{addr_a}]')
                    self.pc += 2
                    self.status = 'ACTIVE'
                else:
                    self.status = 'AWAITING_INPUT'
                    self.log(f'{self.pc}: RCV ...')
                    break
            elif opcode == 4: # SEND OUTPUT
                addr_a = self.get_address(self.pc + 1, modes[0])
                a = self.program[addr_a]
                self.outputs.append(a)
                self.log(f'{self.pc}: SND {a}]')
                self.pc += 2
            elif opcode == 99: # HALT
                self.status = 'HALTED'
                self.log(f'{self.pc}: HLT')
                break
            else:
                raise IndexError('Incorrect opcode: ' + str(opcode))

if __name__ == '__main__':
    vm = IntcodeVM()
