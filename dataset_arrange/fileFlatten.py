import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict
import re


class FileFlattener:
    def __init__(self, root_dir: str, target_level: str):
        self.root_dir = Path(root_dir)
        self.target_level = target_level
        self.processed_files = 0
        self.skipped_files = 0
        self.error_files = 0

    def find_target_directories(self) -> List[Path]:
        target_dirs = []

        for root, dirs, files in os.walk(self.root_dir):
            for dir_name in dirs:
                if dir_name == self.target_level:
                    target_dirs.append(Path(root) / dir_name)

        return target_dirs

    def get_all_files_recursively(self, directory: Path) -> List[Path]:
        files = []

        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                file_path = Path(root) / filename
                files.append(file_path)

        return files

    def generate_unique_filename(self, target_dir: Path, original_file: Path) -> Path:
        rel_path = original_file.relative_to(target_dir)

        if len(rel_path.parts) == 1:
            return target_dir / original_file.name

        path_parts = rel_path.parts[:-1]
        file_stem = original_file.stem
        file_suffix = original_file.suffix

        path_prefix = "_".join(path_parts)
        new_filename = f"{path_prefix}_{file_stem}{file_suffix}"

        target_file = target_dir / new_filename
        counter = 1

        while target_file.exists():
            new_filename = f"{path_prefix}_{file_stem}_{counter}{file_suffix}"
            target_file = target_dir / new_filename
            counter += 1

        return target_file

    def move_file(self, src_file: Path, dst_file: Path) -> bool:
        try:
            # make sure the file exist
            dst_file.parent.mkdir(parents=True, exist_ok=True)

            # file exist
            shutil.move(str(src_file), str(dst_file))
            print(f"move: {src_file.name} -> {dst_file.name}")
            return True

        except Exception as e:
            print(f"error: move failed {src_file} -> {dst_file}: {e}")
            return False

    def copy_file(self, src_file: Path, dst_file: Path) -> bool:
        try:
            # make sure file exist
            dst_file.parent.mkdir(parents=True, exist_ok=True)

            # copy faile
            shutil.copy2(src_file, dst_file)
            print(f"copy: {src_file.name} -> {dst_file.name}")
            return True

        except Exception as e:
            print(f"error: copy failed {src_file} -> {dst_file}: {e}")
            return False

    def clean_empty_directories(self, directory: Path) -> None:
        for root, dirs, files in os.walk(directory, topdown=False):
            root_path = Path(root)

            if root_path.name == self.target_level:
                continue

            try:
                if not any(root_path.iterdir()):
                    root_path.rmdir()
                    print(f"delete {root_path}")
            except OSError:
                pass

    def flatten_files(self, copy_mode: bool = False, clean_empty: bool = True) -> None:

        print(f"start flatten folder '{self.target_level}' 层级...")
        print(f"root dir {self.root_dir}")
        print(f"mode {'copy' if copy_mode else 'remove'}")
        print("-" * 60)

        target_dirs = self.find_target_directories()

        if not target_dirs:
            print(f"error: can't find '{self.target_level}' ")
            return

        print(f"find {len(target_dirs)}  folder:")
        for target_dir in target_dirs:
            print(f"  - {target_dir}")

        self.processed_files = 0
        self.skipped_files = 0
        self.error_files = 0

        for target_dir in target_dirs:
            print(f"\n start {target_dir}")

            all_files = self.get_all_files_recursively(target_dir)

            files_to_move = []
            for file_path in all_files:
                rel_path = file_path.relative_to(target_dir)
                if len(rel_path.parts) > 1:
                    files_to_move.append(file_path)

            print(f"find {len(files_to_move)}  files need to move")

            # move file
            for file_path in files_to_move:
                target_file = self.generate_unique_filename(target_dir, file_path)

                if file_path.parent == target_dir:
                    self.skipped_files += 1
                    continue

                if copy_mode:
                    success = self.copy_file(file_path, target_file)
                else:
                    success = self.move_file(file_path, target_file)

                if success:
                    self.processed_files += 1
                else:
                    self.error_files += 1

            # clean dir
            if clean_empty and not copy_mode:
                print(f"empty dir")
                self.clean_empty_directories(target_dir)

        self.print_summary()



    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("complexity summary: ")
        print("=" * 60)
        print(f": {self.processed_files} files process successfully")
        print(f" {self.skipped_files}  files skipped ")
        print(f" {self.error_files}  files error")
        print(f"total {self.processed_files + self.skipped_files + self.error_files}  files")



if __name__ == "__main__":

    #fakeAV
    flattener_fakeav = FileFlattener(
        root_dir=r"E:\UCL\dissertation_project\dataset\FakeAVCeleb",
        target_level="true"
    )
    flattener_fakeav.flatten_files(copy_mode=False,clean_empty=True)

    fakeav_fake = FileFlattener(
        root_dir=r"E:\UCL\dissertation_project\dataset\FakeAVCeleb",
        target_level="fake"
    )
    fakeav_fake.flatten_files(copy_mode=False, clean_empty=True)