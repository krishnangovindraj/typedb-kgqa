import json
import argparse
import random
import sys


def extract_questions(data, output_file):
    for entry in data:
        output_file.write(entry["question"] + "\n")
    print(f"Extracted {len(data)} questions", file=sys.stderr)


def extract_sources(data, output_file):
    for entry in data:
        titles = [title for title, _ in entry["context"]]
        output_file.write(json.dumps(titles) + "\n")
    print(f"Extracted sources for {len(data)} examples", file=sys.stderr)


def extract_answers(data, output_file):
    for entry in data:
        output_file.write(entry["answer"] + "\n")
    print(f"Extracted {len(data)} answers", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from 2WikiMultiHopQA dataset")
    parser.add_argument("input", help="Path to input JSON file")
    parser.add_argument("command", choices=["questions", "sources", "answers"], help="What to extract")
    parser.add_argument("-o", "--output", default=None, help="Output file path (default: stdout)")
    parser.add_argument("-n", "--num-examples", type=int, default=None, help="Number of examples to extract (default: all)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling. If set, randomly sample; otherwise take first n")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    if args.num_examples:
        if args.seed is not None:
            random.seed(args.seed)
            data = random.sample(data, min(args.num_examples, len(data)))
        else:
            data = data[:args.num_examples]

    output_file = open(args.output, "w") if args.output else sys.stdout
    try:
        if args.command == "questions":
            extract_questions(data, output_file)
        elif args.command == "sources":
            extract_sources(data, output_file)
        elif args.command == "answers":
            extract_answers(data, output_file)
    finally:
        if args.output:
            output_file.close()
