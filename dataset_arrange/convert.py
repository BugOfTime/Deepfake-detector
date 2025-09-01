from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm


root_dir = Path(r"F:/UCL/dissertation_project/cleaned_dataset/audio_data")

def convert_and_replace(mp3_path: Path):
    wav_path = mp3_path.with_suffix('.wav')
    audio = AudioSegment.from_file(mp3_path)
    audio.export(wav_path, format="wav")
    mp3_path.unlink()


def batch_inplace_convert(root_dir: Path):
    mp3_files = list(root_dir.rglob("*.mp3"))
    if not mp3_files:
        print(f"No MP3 files found under {root_dir}")
        return

    for mp3 in tqdm(mp3_files, desc="Converting MP3 to WAV"):
        try:
            convert_and_replace(mp3)
        except Exception as e:
            tqdm.write(f" Failed converting {mp3}: {e}")

if __name__ == "__main__":
    batch_inplace_convert(root_dir)



