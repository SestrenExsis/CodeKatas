import argparse
import difflib

class FileDiffer:
    def __init__(self):
        self.line_diff_count = 0

    def filediff(self, source_file: str, target_file: str):
        self.line_diff_count = 0
        with open(source_file) as src, open(target_file) as trg:
            differ = difflib.Differ()
            for line in differ.compare(src.readlines(), trg.readlines()):
                if not line.startswith(' '):
                    self.line_diff_count += 1
                    print(line[2:], end='')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file1', help='First file', type=str)
    parser.add_argument('file2', help='Second file', type=str)
    args = parser.parse_args()
    file_differ = FileDiffer()
    file_differ.filediff(args.file1, args.file2)
    file1 = args.file1
    print(f'Line-diff count: {file_differ.line_diff_count}')