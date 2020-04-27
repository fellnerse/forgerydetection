from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING

import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader

from forgery_detection.lightning.logging.const import AudioMode

logger = logging.getLogger(__file__)

if TYPE_CHECKING:
    from forgery_detection.data.file_lists import FileList
    from forgery_detection.data.file_lists import SimpleFileList


class SafeImageFolder(ImageFolder):
    def __init__(self, root, *args, **kwargs):
        # make the imagefolder follow symlinks during initialization
        _os_walk = os.walk

        def _walk(dir, *args, **kwargs):
            return _os_walk(dir, followlinks=True)

        setattr(os, "walk", _walk)
        super().__init__(root, *args, **kwargs)
        setattr(os, "walk", _os_walk)

    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except OSError as e:
            logger.info("Some error with an image: ", e)
            logger.info("With path:", self.samples[index][0])
            logger.info("called with index:", index)
            self.samples.pop(index)
            return self.__getitem__(index % len(self))


class FileListDataset(VisionDataset):
    """Almost the same as DatasetFolder by pyTorch.

    But this one does not build up a file list by walking a folder. Instead this file
    list has to be provided."""

    def __init__(
        self,
        file_list: FileList,
        split: str,
        sequence_length: int,
        should_align_faces=False,
        transform=None,
        target_transform=None,
        audio_file_list: Optional[SimpleFileList] = None,
        audio_mode: AudioMode = AudioMode.EXACT,
    ):
        super().__init__(
            file_list.root, transform=transform, target_transform=target_transform
        )
        self.audio_file_list = audio_file_list
        self.should_sample_audio = audio_file_list is not None
        self.audio_mode = audio_mode

        self.classes = file_list.classes
        self.class_to_idx = file_list.class_to_idx
        self._samples = file_list.samples[split]
        self.samples_idx = file_list.samples_idx[split]
        self.split = split
        self.targets = [s[1] for s in self._samples]
        self.sequence_length = sequence_length

        self.should_align_faces = should_align_faces
        if self.should_align_faces:
            self.relative_bbs = file_list.relative_bbs[split]
            if len(self.relative_bbs) == 0:
                raise ValueError("Trying to align faces without relative bbs.")

    def __getitem__(self, index: Tuple[Tuple[int, int], int]):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        (img_idx, align_idx), audio_idx = index

        path, target = self._samples[img_idx]
        vid = default_loader(f"{self.root}/{path}")

        if self.should_align_faces:
            relative_bb = self.relative_bbs[img_idx]
            vid = self.align_face(vid, relative_bb)

        if self.transform is not None:
            vid = self.transform(vid)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.should_sample_audio:
            aud_path, _ = self._samples[audio_idx]
            aud = self.audio_file_list(aud_path)
            sample = vid, aud

            # this indicates if the audio and the images are in sync
            target = (target, int(audio_idx == img_idx))
        else:
            sample = vid

        return sample, target

    def __len__(self):
        return len(self.samples_idx)

    def align_face(self, sample, relative_bb):
        x, y, w, h = self.calculate_relative_bb(
            sample.width, sample.height, relative_bb
        )
        return sample.crop((x, y, (x + w), (y + h)))
        # x, y, w, h = relative_bb
        # width, height = sample.size
        # return sample.crop((x * width, y * height, (x + w) * width, (y + h) * height))

    # todo do this in advance
    @staticmethod
    def calculate_relative_bb(
        total_width: int, total_height: int, relative_bb_values: List[int]
    ):
        x, y, w, h = relative_bb_values

        size_bb = int(max(w, h) * 1.3)
        center_x, center_y = x + int(0.5 * w), y + int(0.5 * h)

        # Check for out of bounds, x-y lower left corner
        x = max(int(center_x - size_bb // 2), 0)
        y = max(int(center_y - size_bb // 2), 0)

        # Check for too big size for given x, y
        size_bb = min(total_width - x, size_bb)
        size_bb = min(total_height - y, size_bb)

        return x, y, size_bb, size_bb

    def _get_possible_audio_shifts_with_min_distance(
        self, idx, min_offset=16, audio_length=8
    ):
        frame_number_in_video, video_length = self._get_video_length_and_frame_idx(idx)

        indices = np.array(
            list(
                range(
                    audio_length - 1,
                    max(audio_length, frame_number_in_video - min_offset + 1),
                )
            )
            + list(
                range(
                    min(
                        video_length - 1,
                        frame_number_in_video + min_offset + audio_length,
                    ),
                    video_length,
                )
            )
        )
        return indices - frame_number_in_video

    def _get_possible_audio_shifts_with_max_distance(
        self, idx, max_distance=50, audio_length=8
    ):
        frame_number_in_video, video_length = self._get_video_length_and_frame_idx(idx)
        indices = np.array(
            list(
                range(
                    max(audio_length, frame_number_in_video - max_distance),
                    frame_number_in_video,
                )
            )
            + list(
                range(
                    frame_number_in_video + 1,
                    min(frame_number_in_video + 1 + max_distance, video_length),
                )
            )
        )
        return indices - frame_number_in_video

    def _get_video_length_and_frame_idx(self, idx):
        path, _ = self._samples[idx]
        abs_path = Path(f"{self.root}/{path}")
        frame_number_in_video = int(abs_path.with_suffix("").name)
        video_length = len(sorted(abs_path.parent.glob("*" + abs_path.suffix)))
        return frame_number_in_video, video_length
