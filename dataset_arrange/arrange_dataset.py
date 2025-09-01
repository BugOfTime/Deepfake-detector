import random
import shutil
from pathlib import Path
from typing import  Sequence, Union
import csv
from tqdm import tqdm
from pydub import AudioSegment
from PIL import Image



# lower case the extension name and remove ".", normalise the extension
def normalise_ext(ext: str | None) -> set[str] | None:
    if ext is None:
        return None
    return {ext.lstrip(".").lower() for ext in ext}

#create a progress bar
def progress_bar(file_list, dst_path, remove=False,desc="Transferring"):
    transferred: list[Path] = []
    for selected_file in tqdm(file_list, desc=desc, unit="file"):
        target = dst_path / selected_file.name
        if remove:
            shutil.move(selected_file, target)
        else:
            shutil.copy2(selected_file, target)
        transferred.append(target)
    return transferred

def write_summary_csv(
    records: list[tuple[str, str, str]],
    dst_path: Union[str, Path],
    filename: str = 'dataset_summary.csv'
) -> Path:

    dst = Path(dst_path)
    csv_path = dst / filename
    dst.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(['filename', 'source', 'label'])
        writer.writerows(records)
    return csv_path

def convert_to_png(src: Path, dst: Path):
    with Image.open(src) as img:
        rgb = img.convert("RGB")
        rgb.save(dst.with_suffix('.png'), format="PNG")

def convert_to_wav(src: Path, dst: Path):
    audio = AudioSegment.from_file(src)
    audio.export(dst.with_suffix('.wav'), format="wav")




def arrange_random_dataset(
    *,
    src: str | Path | Sequence[str | Path],
    true_dst: str | Path,
    fake_dst: str | Path,
    csv_path: str | Path,
    rate: float = 0.5,
    ext: Sequence[str] | None = None,
    remove: bool = False,
) -> list[Path]:
    if not (0 < rate <= 1):
        raise ValueError("'rate' must be in the range (0, 1].")

    src_dirs = [Path(p) for p in (src if isinstance(src, (list, tuple)) else [src])]
    true_path = Path(true_dst)
    fake_path = Path(fake_dst)
    ext_filter = normalise_ext(ext)

    #  Gather all matching files.
    all_files: list[Path] = []
    for data in src_dirs:
        if not data.is_dir():
            raise FileNotFoundError(f"Source directory not found: {data}")
        for file in data.rglob('*'):
            if file.is_file() and (not ext_filter or file.suffix.lstrip('.').lower() in ext_filter):
                all_files.append(file)

    if not all_files:
        raise ValueError("No files matched the given extension filter.")

    # sample files.
    sample_count = max(1, int(len(all_files) * rate))
    selected = random.sample(all_files, sample_count)

    #  ensure destination dirs exist.
    true_path.mkdir(parents=True, exist_ok=True)
    fake_path.mkdir(parents=True, exist_ok=True)

    # fistribute & record
    records: list[tuple[str, str, str]] = []
    for src_file in tqdm(selected, desc="Transferring files", unit="file"):
        # determine true/fake label
        label = src_file.parent.name.lower()
        if label == 'true':
            dest_dir = true_path
        elif label == 'fake':
            dest_dir = fake_path
        else:
            continue

        # extract dataset name
        parts = list(src_file.parts)
        lower = [p.lower() for p in parts]
        if "dataset" in lower and lower.index("dataset") + 1 < len(parts):
            dataset_name = parts[lower.index("dataset") + 1]
        else:
            dataset_name = src_file.parent.parent.name

        # copy and remove
        tgt = dest_dir / src_file.name
        if remove:
            src_file.rename(tgt)
        else:
            shutil.copy2(src_file, tgt)

        records.append((src_file.name, dataset_name, label))

    #  write summary CSV
    csv_file = write_summary_csv(records, csv_path)
    print(f"Total files scanned: {len(all_files)}")
    print(f"Files selected: {len(selected)}")
    print(f"CSV summary written to: {csv_file}")

    return selected




def move_files_by_CSV_label(
        src: str | Path,
        csv_path: str | Path,
        true_folder: str | Path = "true",
        fake_folder: str | Path = "fake",
        ext: Sequence[str] | None = None,
        video_file: bool = False,
        remove: bool = False
):
    src = Path(src)
    true_folder = Path(true_folder)
    fake_folder = Path(fake_folder)
    csv_path = Path(csv_path)
    ext_filter = normalise_ext(ext)

    name2label = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        print("CSV headers:", header)
        if video_file:
            for row in reader:
                video_label = row['Video Ground Truth'].strip().lower()
                audio_label = row['Audio Ground Truth'].strip().lower()

                # only video and audio label is real it can be real
                if video_label == 'real' and audio_label == 'real':
                    name2label[row['Filename']] = 'real'
                else:
                    name2label[row['Filename']] = 'fake'
        else:
            for row in reader:
                name2label[row['Filename']] = row['Ground Truth'].strip().lower()

    true_folder.mkdir(parents=True, exist_ok=True)
    fake_folder.mkdir(parents=True, exist_ok=True)

    files = [file for file in src.iterdir()
             if file.is_file() and (not ext_filter or file.suffix.lstrip(".").lower() in ext_filter)]

    count_true, count_fake, count_notfound = 0, 0, 0
    for file in tqdm(files, desc="Moving files by label", unit="file"):
        label = name2label.get(file.name)
        suffix = file.suffix.lower()

        if label == 'real':
            target_folder = true_folder
            count_true += 1
        elif label == 'fake':
            target_folder = fake_folder
            count_fake += 1
        else:
            count_notfound += 1
            continue

        moved_successfully = False

        if suffix in (".jpg", ".webp",".jpeg"):

            new_name = file.stem + '.png'
            target_path = target_folder / new_name
            try:
                convert_to_png(file, target_path)
                moved_successfully = True
                if remove:
                    file.unlink()
            except Exception as e:
                print(f"convert failed {file.name}: {e}")

        elif suffix in (".mp4",".m4a",".mp3"):

            new_name = file.stem + '.wav'
            target_path = target_folder / new_name
            try:
                convert_to_wav(file, target_path)
                moved_successfully = True

                if remove:
                    file.unlink()
            except Exception as e:
                print(f"convert failed {file.name}: {e}")

        else:
            target_path = target_folder / file.name
            try:
                if remove:
                    shutil.move(str(file), str(target_path))
                else:
                    shutil.copy2(str(file), str(target_path))
                moved_successfully = True
            except Exception as e:
                print(f"move_failed {file.name}: {e}")

        if not moved_successfully:
            if label == 'real':
                count_true -= 1
            elif label == 'fake':
                count_fake -= 1
    print(
        f"Moved {count_true} to true folder, {count_fake} to fake folder, {count_notfound} files label not found in CSV.")




def prepare_image_dataset():

    sources_folder = [

        #deepfake-Eval-2024 dataset
        "F:/UCL/dissertation_project/dataset/Deepfake-Eval-2024/image_data/true",
        "F:/UCL/dissertation_project/dataset/Deepfake-Eval-2024/image_data/fake"


    ]

    arrange_random_dataset(
        src=sources_folder,
        true_dst="F:/UCL/dissertation_project/cleaned_dataset/image_data/true",
        fake_dst="F:/UCL/dissertation_project/cleaned_dataset/image_data/fake",
        csv_path="F:/UCL/dissertation_project/cleaned_dataset/image_data",
        rate=1,           # choose how many samples that you want
        ext=["png"],
        remove=False         # if you want to remove the original file
    )

def prepare_video_dataset():
    sources_folder = [
        # DeeperDeeperForensics
        "F:/UCL/dissertation_project/dataset/DeeperForensics/true",
        "F:/UCL/dissertation_project/dataset/DeeperForensics/fake",

        # deepfake-Eval-2024 dataset
        "F:/UCL/dissertation_project/dataset/Deepfake-Eval-2024/video_data/true",
        "F:/UCL/dissertation_project/dataset/Deepfake-Eval-2024/video_data/fake",

        # deepspeak
        "F:/UCL/dissertation_project/dataset/deepspeak/true",
        "F:/UCL/dissertation_project/dataset/deepspeak/fake",

        # FaceForensics
        "F:/UCL/dissertation_project/dataset/FaceForensics/true",
        "F:/UCL/dissertation_project/dataset/FaceForensics/fake",
        #
        # # FakeAVCeleb
        "F:/UCL/dissertation_project/dataset/FakeAVCeleb/true",
        "F:/UCL/dissertation_project/dataset/FakeAVCeleb/fake"

    ]

    arrange_random_dataset(
        src=sources_folder,
        true_dst="F:/UCL/dissertation_project/cleaned_dataset/video_data/true",
        fake_dst="F:/UCL/dissertation_project/cleaned_dataset/video_data/fake",
        csv_path="F:/UCL/dissertation_project/cleaned_dataset/video_data",
        rate=0.7,  # choose how many samples that you want
        ext=["mp4"],
        remove=False  # if you want to remove the original file
    )

def prepare_audio_dataset():
    sources_folder = [

        # deepfake-Eval-2024 dataset (True)
        "F:/UCL/dissertation_project/dataset/Deepfake-Eval-2024/audio_data/fake",
        "F:/UCL/dissertation_project/dataset/Deepfake-Eval-2024/audio_data/true"

    ]

    arrange_random_dataset(
        src=sources_folder,
        true_dst="F:/UCL/dissertation_project/cleaned_dataset/audio_data/true",
        fake_dst="F:/UCL/dissertation_project/cleaned_dataset/audio_data/fake",
        csv_path="F:/UCL/dissertation_project/cleaned_dataset/audio_data",
        rate=1,  # choose how many samples that you want
        ext=["wav"],
        remove=False  # if you want to remove the original file
    )

if __name__ == "__main__":

    # move_files_by_CSV_label(
    #     src="E:/UCL/dissertation_project/dataset/Deepfake-Eval-2024/image-data",
    #     csv_path="E:/UCL/dissertation_project/dataset/Deepfake-Eval-2024/image-metadata-publish.csv",
    #     true_folder="E:/UCL/dissertation_project/dataset/Deepfake-Eval-2024/image_data/true",
    #     fake_folder="E:/UCL/dissertation_project/dataset/Deepfake-Eval-2024/image_data/fake",
    #     ext=["jpg", "webp","jpeg","png"],
    #     video_file=False,
    #     remove=False
    # )

    # move_files_by_CSV_label(
    #     src="E:/UCL/dissertation_project/dataset/Deepfake-Eval-2024/video-data",
    #     csv_path="E:/UCL/dissertation_project/dataset/Deepfake-Eval-2024/video-metadata-publish.csv",
    #     true_folder="E:/UCL/dissertation_project/dataset/Deepfake-Eval-2024/video_data/true",
    #     fake_folder="E:/UCL/dissertation_project/dataset/Deepfake-Eval-2024/video_data/fake",
    #     ext=["mp4"],
    #     video_file=True,
    #     remove=False
    # )
    #
    # move_files_by_CSV_label(
    #     src="F:/UCL/dissertation_project/dataset/Deepfake-Eval-2024/audio-data",
    #     csv_path="F:/UCL/dissertation_project/dataset/Deepfake-Eval-2024/audio-metadata-publish.csv",
    #     true_folder="F:/UCL/dissertation_project/dataset/Deepfake-Eval-2024/audio_data/true",
    #     fake_folder="F:/UCL/dissertation_project/dataset/Deepfake-Eval-2024/audio_data/fake",
    #     ext=["mp4","m4a","mp3","wav"],
    #     video_file=False,
    #     remove=False
    # )


    # prepare_image_dataset()
    prepare_audio_dataset()
    prepare_video_dataset()


