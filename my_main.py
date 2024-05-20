import os


def rename_files_in_directory(this_path, prefix):
    i = 1
    for filename in os.listdir(this_path):
        new_name = prefix + str(i) + ".jpg"
        source = os.path.join(this_path, filename)
        destination = os.path.join(this_path, new_name)
        os.rename(source, destination)
        i += 1


if __name__ == '__main__':
    path = 'train/pulgones/'
    rename_files_in_directory(path, "sheet-")
