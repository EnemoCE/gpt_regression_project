import os
import shutil

def clean_dirs_by_name(base_path, short_description):
    if not os.path.exists(base_path):
        print(f"No such directory: {base_path}")
        return

    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if dir_name == short_description:
                dir_path = os.path.join(root, dir_name)
                # Delete the directory
                shutil.rmtree(dir_path)
                print(f"Deleted directory: {dir_path}")
