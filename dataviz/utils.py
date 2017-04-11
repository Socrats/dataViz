import glob
import os


def search_path(root_dir: str, expression: str) -> list:
    files = []
    for path, dirs, files in os.walk(root_dir):
        for filename in files:
            if expression in filename:
                files.append(filename)
    return files


def get_files(root_dir: str, expression: str) -> dict:
    files = {}
    for i in [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]:
        files[''.join(['f', str(i)])] = \
            glob.glob(''.join([root_dir, ''.join(['/f', str(i), '/data/*', expression, '-4999*'])]))
    return files


def get_current_path():
    return os.getcwd()
