import os
from typing import List

AUDIO_EXTENSIONS = [
    ".wav",
    ".aiff",
]  # Example extensions


def find_audio_files(
    start_path: str, extensions: List[str] = AUDIO_EXTENSIONS
) -> List[str]:
    """Finds all audio files in a directory recursively.
    Returns a list of file paths.
    """
    # Ensure extensions are in a consistent format
    extensions = [
        ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions
    ]

    matched_files = []

    # Walk through all directories and files starting from start_path
    for root, dirs, files in os.walk(start_path):
        for file in files:
            # Check if the file ends with any of the provided extensions
            if any(file.lower().endswith(ext) for ext in extensions):
                matched_files.append(os.path.join(root, file))

    # Sort the list of matched files in alphabetical order before returning
    return sorted(matched_files)


import soundfile as sf
from typing import List


def filter_audio_files_by_properties(
    audio_files: List[str], sample_rate: int, subtype: str
) -> List[str]:
    """Filters a list of audio files by sample rate and bit depth.

    Args:
        audio_files: A list of paths to audio files.
        target_sample_rate: The target sample rate to filter files by.
        target_bit_depth: The target bit depth to filter files by.

    Returns:
        A list of file paths that match the target sample rate and bit depth.
    """
    matched_files = []

    for file_path in audio_files:
        info = sf.info(file_path)
        # print(type(info.samplerate), type(info.subtype))
        print(info.samplerate, info.subtype)
        # Determining bit depth from subtype, assuming subtype contains bit depth information

        if info.samplerate == sample_rate and info.subtype == subtype:
            matched_files.append(file_path)

    return matched_files


if __name__ == "__main__":
    # Example usage
    start_path = "/Volumes/SD256"
    audio_files = find_audio_files(start_path)
    print(f"Found {len(audio_files)} audio files.")

    target_sample_rate = int(48000)
    target_bit_depth = str("PCM_24")
    filtered_files = filter_audio_files_by_properties(
        audio_files, target_sample_rate, target_bit_depth
    )
    print(f"Found {len(filtered_files)} files matching the criteria.")

    for f in filtered_files:
        print(f)
