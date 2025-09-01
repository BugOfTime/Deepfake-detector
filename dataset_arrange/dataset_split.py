import random
import shutil
from pathlib import Path
from typing import Sequence, Union, Optional, Tuple,Dict
import csv
from tqdm import tqdm
from collections import defaultdict


#split dataset to train,train and val
class dataset_split:

    def __init__(self,
                 dataset_path: Union[str, Path],
                 output_path: Union[str, Path],
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_seed: Optional[int] = None): # it could use same seed to create the same splited dataset


        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # check the total sum
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
            raise ValueError("train,train and val ratios must sum to 1.0")

        if random_seed is not None:
            random.seed(random_seed)

# get the fiels by labels
    def get_files(self, label: str) -> list[Path]:
        label_path = self.dataset_path / label
        if not label_path.exists():
            raise FileNotFoundError(f"can't find the label address: {label_path}")


        # for file in label_path.iterdir():
        #     if file.is_file():
        #         if ext_filter is None or file.suffix.lstrip('.').lower() in ext_filter:
        #             files.append(file)

        files = [file for file in label_path.iterdir() if file.is_file()]
        return files


# create output directory structure
    def create_output_folder(self):

        splits = ['train', 'val', 'train']
        labels = ['fake', 'true']

        for split in splits:
            for label in labels:
                output_dir = self.output_path / split / label
                output_dir.mkdir(parents=True, exist_ok=True)


    # shuffle all the files
    def split_files(self, files: list[Path]) -> Tuple[list[Path], list[Path], list[Path]]:

        shuffled_files = files.copy()
        random.shuffle(shuffled_files)

        total_files = len(shuffled_files)
        train_count = int(total_files * self.train_ratio)
        val_count = int(total_files * self.val_ratio)

        train_files = shuffled_files[:train_count]
        val_files = shuffled_files[train_count:train_count + val_count]
        test_files = shuffled_files[train_count + val_count:]

        return train_files, val_files, test_files

    def copy_files(self, files: list[Path], dst_dir: Path, label: str,
                    remove_original: bool = False) -> list[Path]:

        copied_files = []
        label_dst = dst_dir / label

        for file in tqdm(files, desc=f"move {label} to {dst_dir.name}", unit="file"):
            dst_file = label_dst / file.name
            try:
                if remove_original:
                    shutil.move(str(file), str(dst_file))
                else:
                    shutil.copy2(str(file), str(dst_file))
                copied_files.append(dst_file)
            except Exception as e:
                print(f"copy file failed {file.name}: {e}")

        return copied_files

    def split_dataset(self,
                      remove_original: bool = False,
                      create_summary: bool = False) -> dict:


        self.create_output_folder()

        # get all the file
        fake_files = self.get_files('fake')
        true_files = self.get_files('true')


        # split dataset
        fake_train, fake_val, fake_test = self.split_files(fake_files)
        true_train, true_val, true_test = self.split_files(true_files)

        train_dir = self.output_path / 'train'
        val_dir = self.output_path / 'val'
        test_dir = self.output_path / 'train'

        train_fake = self.copy_files(fake_train, train_dir, 'fake', remove_original)
        train_true = self.copy_files(true_train, train_dir, 'true', remove_original)

        test_fake = self.copy_files(fake_test, test_dir, 'fake', remove_original)
        test_true = self.copy_files(true_test, test_dir, 'true', remove_original)

        val_fake = self.copy_files(fake_val, val_dir, 'fake', remove_original)
        val_true = self.copy_files(true_val, val_dir, 'true', remove_original)



        result = {
            'train': {
                'fake': len(train_fake),
                'true': len(train_true),
                'total': len(train_fake) + len(train_true)
            },
            'val': {
                'fake': len(val_fake),
                'true': len(val_true),
                'total': len(val_fake) + len(val_true)
            },
            'train': {
                'fake': len(test_fake),
                'true': len(test_true),
                'total': len(test_fake) + len(test_true)
            }
        }

        # create csv
        if create_summary:
            self.create_csv(result,self.output_path)

        # print the result
        self.print_summary(result)

        return result


    def create_csv(self,
            result: Dict[str, Dict[str, int]],
            dst_path: Union[str, Path],
            filename: str = 'splited_summary.csv'
    ) -> Path:

        dst = Path(dst_path)
        dst.mkdir(parents=True, exist_ok=True)
        csv_path = dst / filename

        # count all the split number
        total_files = sum(result.get(split, {}).get('total', 0) for split in ['train', 'val', 'train'])

        with csv_path.open('w', newline='', encoding='utf-8') as cf:
            writer = csv.writer(cf)
            writer.writerow(['split', 'label', 'count', 'percentage'])

            for split in ['train', 'val', 'train']:
                counts = result.get(split, {})
                for label in ['fake', 'true']:
                    count = counts.get(label, 0)
                    percentage = (count / total_files) * 100 if total_files > 0 else 0
                    writer.writerow([split, label, count, f"{percentage:.2f}%"])

        print(f"the scv file has been saved to: {csv_path}")
        return csv_path

    def print_summary(self, result: dict):
        # print the detail
        print("\n=== dataset split summary ===")
        print(f"{'split':<10} {'Fake':<10} {'True':<10} {'total':<10}")
        print("-" * 40)

        for split in ['train', 'val', 'train']:
            fake_count = result[split]['fake']
            true_count = result[split]['true']
            total_count = result[split]['total']
            print(f"{split:<10} {fake_count:<10} {true_count:<10} {total_count:<10}")

        # print total
        total_fake = sum(result[split]['fake'] for split in ['train', 'val', 'train'])
        total_true = sum(result[split]['true'] for split in ['train', 'val', 'train'])
        total_all = total_fake + total_true

        print("-" * 40)
        print(f"{'total':<10} {total_fake:<10} {total_true:<10} {total_all:<10}")

        # calculate radio
        if total_all > 0:
            train_ratio = result['train']['total'] / total_all
            val_ratio = result['val']['total'] / total_all
            test_ratio = result['train']['total'] / total_all

            print(f"\n actual ratio:")
            print(f"train: {train_ratio:.2%}")
            print(f"train: {test_ratio:.2%}")
            print(f"val: {val_ratio:.2%}")




if __name__ == "__main__":

    image_split = dataset_split(
        dataset_path = "F:/UCL/dissertation_project/cleaned_dataset/image_data",
        output_path = "F:/UCL/dissertation_project/prepared_dataset/image_data",
        train_ratio = 0.7,
        val_ratio = 0.15,
        test_ratio = 0.15,
        random_seed = 107
    )

    # split the dataset
    split_image = image_split.split_dataset(
        remove_original = False,  # keep the original file
        create_summary = True  # create the summary
    )



    # split audio dataset
    audio_split = dataset_split(
        dataset_path = "F:/UCL/dissertation_project/cleaned_dataset/audio_data",
        output_path = "F:/UCL/dissertation_project/prepared_dataset/audio_data",
        train_ratio = 0.7,
        val_ratio = 0.15,
        test_ratio = 0.15,
        random_seed = 107
    )

    split_audio = audio_split.split_dataset(
        remove_original = False,
        create_summary = True
    )

    # split video dataset
    video_split = dataset_split(
        dataset_path = "F:/UCL/dissertation_project/cleaned_dataset/video_data",
        output_path = "F:/UCL/dissertation_project/prepared_dataset/video_data",
        train_ratio = 0.7,
        val_ratio = 0.15,
        test_ratio = 0.15,
        random_seed = 107
    )

    split_video = video_split.split_dataset(
        remove_original=False,  # keep the original file
        create_summary=True  # create the summary
    )
