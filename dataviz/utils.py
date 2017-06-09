import glob
import os


def search_path(root_dir: str, expression: str) -> list:
    files = []
    for path, dirs, files in os.walk(root_dir):
        for filename in files:
            if expression in filename:
                files.append(filename)
    return files


def get_files(root_dir: str, expression: str, folders: list or str, size: int) -> dict:
    # TODO: change this so that it accepts the value
    files = {}
    if type(folders) is list:
        for folder in folders:
            files[folder] = \
                glob.glob("{root_dir}{folder}/data/*{expression}-{size}*".format(root_dir=root_dir, folder=folder,
                                                                                 expression=expression, size=size))
    else:
        files[folders] = \
            glob.glob(''.join([root_dir, ''.join(['/data/*', expression, '-9999*'])]))
    return files


def get_current_path():
    return os.getcwd()
