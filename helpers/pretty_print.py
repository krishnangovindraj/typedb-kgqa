import json
import argparse

def pretty_print(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Wrote pretty-printed JSON to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretty-print a JSON file")
    parser.add_argument("input", help="Path to input JSON file")
    parser.add_argument("-o", "--output", default=None, help="Output file path (default: <input>_pretty.json)")
    args = parser.parse_args()
    output = args.output or args.input.replace(".json", "_pretty.json")
    pretty_print(args.input, output)
