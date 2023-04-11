from pathlib import Path
import lmdb
from typing import List
import numpy as np
import cv2
import io

class Ego4DHLMDB():
    def __init__(self, path_to_root: Path, readonly=False, lock=False, frame_template="{video_id:s}_{frame_number:010d}", map_size=1099511627776) -> None:
        self.environments = {}
        self.path_to_root = path_to_root
        if isinstance(self.path_to_root, str):
            self.path_to_root = Path(self.path_to_root)
        self.path_to_root.mkdir(parents=True, exist_ok=True)
        self.readonly = readonly
        self.lock = lock
        self.map_size = map_size
        self.frame_template = frame_template

    def _get_parent(self, parent: str) -> lmdb.Environment:
        return lmdb.open(str(self.path_to_root / parent), map_size=self.map_size, readonly=self.readonly, lock=self.lock)

    def put_batch(self, video_id: str, frames: List[int], data: List[np.ndarray]) -> None:
        with self._get_parent(video_id) as env:
            with env.begin(write=True) as txn:
                for frame, value in zip(frames, data):
                    if value is not None:
                        txn.put(self.frame_template.format(video_id=video_id,frame_number=frame).encode(), cv2.imencode('.jpg', value)[1])

    def put(self, video_id: str, frame: int, data: np.ndarray) -> None:
        if data is not None:
            with self._get_parent(video_id) as env:
                with env.begin(write=True) as txn:
                    txn.put(self.frame_template.format(video_id=video_id,frame_number=frame).encode(), cv2.imencode('.jpg', data)[1])

    def get(self, video_id: str, frame: int) -> np.ndarray:
        with self._get_parent(video_id) as env:
            with env.begin(write=False) as txn:
                data = txn.get(self.frame_template.format(video_id=video_id,frame_number=frame).encode())

                file_bytes = np.asarray(
                    bytearray(io.BytesIO(data).read()), dtype=np.uint8
                )
                return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    def get_batch(self, video_id: str, frames: List[int]) -> List[np.ndarray]:
        out = []
        with self._get_parent(video_id) as env:
            with env.begin() as txn:
                for frame in frames:
                    data = txn.get(self.frame_template.format(video_id=video_id,frame_number=frame).encode())
                    file_bytes = np.asarray(
                        bytearray(io.BytesIO(data).read()), dtype=np.uint8
                    )
                    out.append(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR))
            return out

    def get_existing_keys(self):
        keys = []
        for parent in self.path_to_root.iterdir():
            with self._get_parent(parent.name) as env:
                with env.begin() as txn:
                    keys += list(txn.cursor().iternext(values=False))
        return keys