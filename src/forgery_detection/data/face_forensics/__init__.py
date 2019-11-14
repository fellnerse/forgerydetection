import itertools
from enum import auto
from enum import Enum
from pathlib import Path
from typing import Iterable
from typing import List
from typing import Union


class StrEnum(str, Enum):
    def __str__(self):
        return self.name


class Compression(StrEnum):
    raw = auto()
    c23 = auto()
    c40 = auto()


class DataType(StrEnum):
    videos = auto()
    face_images_tracked = auto()
    face_images = auto()
    masks = auto()
    full_images = auto()
    face_information = auto()


class FaceForensicsDataStructure:
    SUB_DIRECTORIES = sorted(
        ["original_sequences/youtube"]
        + [
            "manipulated_sequences/" + manipulated_sequence
            for manipulated_sequence in [
                "Deepfakes",
                "Face2Face",
                "FaceSwap",
                "NeuralTextures",
            ]
        ]
    )
    METHODS = list(map(lambda path: path.split("/")[1], SUB_DIRECTORIES))

    def __init__(
        self,
        root_dir: str,
        compressions: Iterable[Union[str, Compression]] = (Compression.raw,),
        data_types: List[Union[str, DataType]] = (DataType.face_images,),
    ):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"{self.root_dir} does not exist!")
        self.data_types = data_types if isinstance(data_types, tuple) else (data_types,)
        self.compressions = (
            compressions if isinstance(compressions, tuple) else (compressions,)
        )

    def get_subdirs(self) -> List[Path]:
        """Returns subdirectories containing datatype """
        return [
            self.root_dir / subdir / compression / data_type
            for subdir, compression, data_type in itertools.product(
                self.SUB_DIRECTORIES, self.compressions, self.data_types
            )
        ]
