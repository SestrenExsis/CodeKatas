'''
Created on Nov 28, 2023

@author: Sestren
'''
from typing import Dict

class IntcodeVM:
    '''
    A simple 3-dimensional scalar
    '''
    def __init__(self, program: Dict[int, int]):
        self.program = program
        self.pc = 0

    def step(self):
        opcode = self.program[self.pc]
        if opcode == 1:
            addr_a = self.program[self.pc + 1]
            addr_b = self.program[self.pc + 2]
            addr_c = self.program[self.pc + 3]
            a = self.program[addr_a]
            b = self.program[addr_b]
            self.program[addr_c] = a + b
            self.pc += 4
        elif opcode == 2:
            addr_a = self.program[self.pc + 1]
            addr_b = self.program[self.pc + 2]
            addr_c = self.program[self.pc + 3]
            a = self.program[addr_a]
            b = self.program[addr_b]
            self.program[addr_c] = a * b
            self.pc += 4
        elif opcode == 99:
            raise StopIteration()
        else:
            raise IndexError('Incorrect opcode: ' + opcode)

    def run(self):
        while True:
            try:
                self.step()
            except StopIteration:
                return
            except IndexError:
                return

if __name__ == '__main__':
    vm = IntcodeVM()
