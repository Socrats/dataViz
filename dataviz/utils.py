import glob
import os


def search_path(root_dir: str, expression: str) -> list:
    files = []
    for path, dirs, files in os.walk(root_dir):
        for filename in files:
            if expression in filename:
                files.append(filename)
    return files


def get_files(root_dir: str, expression: str, folders: list) -> dict:
    files = {}
    for folder in folders:
        files[folder] = \
            glob.glob(''.join([root_dir, ''.join(['/', folder, '/data/*', expression, '-4999*'])]))
    return files


def get_current_path():
    return os.getcwd()
