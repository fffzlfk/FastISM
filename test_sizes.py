#!/usr/bin/python3

import sys
import os

def run_command(cmd: str):
    os.system(cmd)


if __name__ == "__main__":
    func = sys.argv[1]
    for i in range(8, 14):
        run_command(f"./build/{ func } images/sizes/{ 1 << i }.png")
