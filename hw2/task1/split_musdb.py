import musdb
import shutil
from pathlib import Path

mus_val = musdb.DB(
    root="/home/ndrw1221/nas/datasets/musdb18hq-dataset",
    is_wav=True,
    subsets="train",
    split="valid",
)

valid_tracks = mus_val.tracks
root = Path("/home/ndrw1221/nas/datasets/musdb18stft")
train_dir = root / "train"
valid_dir = root / "valid"

# Ensure the valid directory exists
valid_dir.mkdir(parents=True, exist_ok=True)

# Move each valid track from the train directory to the valid directory
for track in valid_tracks:
    track_name = track.name
    source_dir = train_dir / track_name  # Path of the track in the train directory
    destination_dir = (
        valid_dir / track_name
    )  # Path to move the track to the valid directory

    if source_dir.exists():
        print(f"Moving {track_name} from {source_dir} to {destination_dir}")
        shutil.move(str(source_dir), str(destination_dir))
    else:
        print(f"Track {track_name} not found in {train_dir}")
