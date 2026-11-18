"""Usage: python scripts/verify_results.py artifacts/search_RUN_ID"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval_utils import verify_result_directory

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_result_directory(args.directory), indent=2))
