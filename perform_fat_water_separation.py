from utils import *

import sys

def main():
    # Check if two arguments are provided
    if len(sys.argv) != 4:
        print("Usage: python perform_fat_water_separation.py <model_path>< ip_path> <op_path>")
        return

    model_path = sys.argv[1]
    ip_path = sys.argv[2]
    op_path = sys.argv[3]

    fat_water_separation_torch(model_path, ip_path, op_path, 1024.)


if __name__ == "__main__":
    main()
