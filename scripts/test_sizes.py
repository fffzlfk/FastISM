#!/usr/bin/python3

import sys
import os

def run_command(cmd: str):
    os.system(cmd)


if __name__ == "__main__":
    func = sys.argv[1]
    for i in range(10, 16):
        run_command(f"./build/{ func } images/{ 1 << i }x{ 1 << i }.png")
