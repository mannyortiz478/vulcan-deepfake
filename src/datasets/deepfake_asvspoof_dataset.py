import logging
from pathlib import Path

import pandas as pd

from src.datasets.base_dataset import SimpleAudioFakeDataset

DF_ASVSPOOF_SPLIT = {
    "partition_ratio": [0.7, 0.15],
    "seed": 45
}

LOGGER = logging.getLogger()

class DeepFakeASVSpoofDataset(SimpleAudioFakeDataset):

    protocol_file_name = "keys/CM/trial_metadata.txt"
    subset_dir_prefix = "ASVspoof2021_DF_eval"
    subset_parts = ("part00", "part01", "part02", "part03")

    def __init__(self, path, subset="train", transform=None):
        super().__init__(subset, transform)
        # normalize and expand user-provided path (handles ~)
        self.path = Path(path).expanduser()

        self.partition_ratio = DF_ASVSPOOF_SPLIT["partition_ratio"]
        self.seed = DF_ASVSPOOF_SPLIT["seed"]

        self.flac_paths = self.get_file_references()
        if not self.flac_paths:
            raise FileNotFoundError(
                f"No .flac files found under {self.path}. Ensure the dataset is extracted and you passed the correct path. "
                "If you used '~' ensure it was expanded or provide an absolute path." 
            )

        self.samples = self.read_protocol()

        self.transform = transform
        LOGGER.info(f"Spoof: {len(self.samples[self.samples['label'] == 'spoof'])}")
        LOGGER.info(f"Original: {len(self.samples[self.samples['label'] == 'bonafide'])}")

    def get_file_references(self):
        flac_paths = {}

        # Candidate directories to search (in order of preference)
        candidate_dirs = []
        for part in self.subset_parts:
            candidate_dirs.append(
                Path(self.path) / f"{self.subset_dir_prefix}_{part}" / self.subset_dir_prefix / "flac"
            )
        # also consider the top-level dataset layout (no part folders)
        candidate_dirs.append(Path(self.path) / self.subset_dir_prefix / "flac")
        # include any nested matches for robustness
        candidate_dirs.extend(list(Path(self.path).glob(f"**/{self.subset_dir_prefix}/flac")))

        # deduplicate while preserving order
        seen = set()
        unique_dirs = []
        for d in candidate_dirs:
            sd = str(d)
            if sd not in seen:
                unique_dirs.append(d)
                seen.add(sd)

        # Search candidate dirs first
        for d in unique_dirs:
            if d.exists():
                flac_list = list(d.glob("*.flac"))
                LOGGER.info(f"Searching flac dir: {d} ({len(flac_list)} files)")
                for f in flac_list:
                    flac_paths[f.stem] = f

        # Fallback: scan entire tree for .flac if nothing found yet
        if not flac_paths:
            all_flacs = list(Path(self.path).glob("**/*.flac"))
            LOGGER.info(f"Fallback: scanned entire tree, found {len(all_flacs)} flac files")
            for f in all_flacs:
                flac_paths[f.stem] = f

        LOGGER.info(f"Total flac files indexed: {len(flac_paths)}")
        if not flac_paths:
            tried = [str(d) for d in unique_dirs]
            raise FileNotFoundError(
                f"No .flac files found when searching under {self.path}. Searched candidate directories: {tried}. "
                "Please confirm the dataset is extracted and the path is correct (avoid unexpanded ~)."
            )
        return flac_paths

    def read_protocol(self):
        samples = {
            "sample_name": [],
            "label": [],
            "path": [],
            "attack_type": [],
        }

        real_samples = []
        fake_samples = []
        with open(Path(self.path) / self.protocol_file_name, "r") as file:
            for line in file:
                tokens = line.strip().split()
                # token positions may vary across versions; find label by token content
                label = next((t for t in tokens if t in ("bonafide", "spoof")), None)

                if label == "bonafide":
                    real_samples.append(line)
                elif label == "spoof":
                    fake_samples.append(line)

        fake_samples = self.split_samples(fake_samples)
        for line in fake_samples:
            samples = self.add_line_to_samples(samples, line)

        real_samples = self.split_samples(real_samples)
        for line in real_samples:
            samples = self.add_line_to_samples(samples, line)

        return pd.DataFrame(samples)

    def add_line_to_samples(self, samples, line):
        tokens = line.strip().split()
        # most protocols place filename in position 1
        sample_name = tokens[1]
        # normalize sample_name (strip, remove extension if present)
        sample_name = sample_name.strip()
        if sample_name.endswith('.flac'):
            sample_name = sample_name[:-5]

        # label present somewhere in tokens (we already filtered), find it
        label = next((t for t in tokens if t in ("bonafide", "spoof")), "spoof")
        samples["sample_name"].append(sample_name)
        samples["label"].append(label)
        samples["attack_type"].append(label)

        # Robust lookup: try exact match first, then basename, then substring matches
        sample_path = None
        if sample_name in self.flac_paths:
            sample_path = self.flac_paths[sample_name]
        else:
            # try basename (if sample_name contains directories)
            base = sample_name.split('/')[-1]
            if base in self.flac_paths:
                sample_path = self.flac_paths[base]

        if sample_path is None:
            # try substring/contains matching (e.g. keys may contain extra prefixes)
            matches = [v for k, v in self.flac_paths.items() if sample_name in k or k in sample_name]
            if matches:
                sample_path = matches[0]
                LOGGER.warning(
                    f"Protocol sample_name '{sample_name}' not found as exact stem; using best match '{sample_path.stem}'"
                )

        if sample_path is None:
            # helpful error listing a few available stems
            available = list(self.flac_paths.keys())[:10]
            raise KeyError(
                f"Sample '{sample_name}' not found in flac paths. Available stems (first 10): {available}"
            )

        assert sample_path.exists(), f"Referenced audio file does not exist: {sample_path}"
        samples["path"].append(sample_path)

        return samples

