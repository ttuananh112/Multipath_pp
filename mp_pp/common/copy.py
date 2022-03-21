import os
from shutil import copyfile


def copy_file(
        src_file: str,
        dst_folder: str
) -> None:
    """
    Copy file from src_file to dst_folder
    File name is the same as src_file
    :param src_file: path to source file
    :param dst_folder: path to destination folder
    :return:
    """
    f_name = os.path.basename(src_file)
    dst_file = os.path.join(dst_folder, f_name)
    copyfile(src_file, dst_file)
