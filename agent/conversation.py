import base64
import io
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Literal, Tuple

from livekit import rtc
from livekit.agents.tokenize.basic import (
    SentenceTokenizer,
    WordTokenizer,
    hyphenate_word,
)
from livekit.agents.utils import images
from PIL import Image


class EntryType(Enum):
    USER_SPEECH = "user_speech"
    CAMERA_FRAME = "camera_frame"
    SCREENSHARE_FRAME = "screenshare_frame"
    FOUR_CAMERA_FRAMES = "four_camera_frames"
    FOUR_SCREENSHARE_FRAMES = "four_screenshare_frames"
    SIXTEEN_CAMERA_FRAMES = "sixteen_camera_frames"
    SIXTEEN_SCREENSHARE_FRAMES = "sixteen_screenshare_frames"
    ASSISTANT_SPEECH = "assistant_speech"


@dataclass
class TimelineEntry:
    entry_type: EntryType
    timestamp: float
    content: str
    duration: float = 0.0


@dataclass
class FrameRate:
    speaking: float
    not_speaking: float


SPEAKING_FRAME_RATE = 2
NOT_SPEAKING_FRAME_RATE = 0.5
FOUR_FRAMES_COMPRESSION_CUTOFF = 10.0
SIXTEEN_FRAMES_COMPRESSION_CUTOFF = 60.0

HYPHEN_SPEECH_RATE = 3.83  # hyphens per second


class ConversationTimeline:
    def __init__(self):
        self.entries: List[TimelineEntry] = []
        self.sentence_tokenizer = SentenceTokenizer()
        self.word_tokenizer = WordTokenizer()
        self._is_speaking = False

    @property
    def is_speaking(self):
        return self._is_speaking

    @is_speaking.setter
    def is_speaking(self, value: bool):
        self._is_speaking = value

    def _insert_entry_chronologically(self, entry: TimelineEntry):
        """Insert an entry into the timeline in chronological order"""
        # Find the correct position using binary search
        left, right = 0, len(self.entries)
        while left < right:
            mid = (left + right) // 2
            if self.entries[mid].timestamp < entry.timestamp:
                left = mid + 1
            else:
                right = mid
        self.entries.insert(left, entry)

    def compress_entries(self):
        now = time.time()

        # First pass: compress frames older than 10 seconds into 2x2 grids
        resampled = []
        four_frames_buffer = []

        for entry in reversed(self.entries):
            age = now - entry.timestamp

            if entry.entry_type not in [
                EntryType.CAMERA_FRAME,
                EntryType.SCREENSHARE_FRAME,
            ]:
                resampled.insert(0, entry)
                continue

            if age > FOUR_FRAMES_COMPRESSION_CUTOFF:
                four_frames_buffer.append(entry)
                if len(four_frames_buffer) == 4:
                    merged_entry = self._merge_four_frames(four_frames_buffer)
                    resampled.insert(0, merged_entry)
                    four_frames_buffer = []
            else:
                resampled.insert(0, entry)

        # Handle remaining frames in first pass
        if len(four_frames_buffer) > 0:
            merged_entry = self._merge_four_frames(four_frames_buffer)
            resampled.insert(0, merged_entry)

        # Second pass: compress 2x2 grids older than 60 seconds into 4x4 grids
        final_resampled = []
        sixteen_frames_buffer = []

        for entry in reversed(resampled):
            age = now - entry.timestamp

            if entry.entry_type not in [
                EntryType.FOUR_CAMERA_FRAMES,
                EntryType.FOUR_SCREENSHARE_FRAMES,
            ]:
                final_resampled.insert(0, entry)
                continue

            if age > SIXTEEN_FRAMES_COMPRESSION_CUTOFF:
                sixteen_frames_buffer.append(entry)
                if len(sixteen_frames_buffer) == 4:
                    merged_entry = self._merge_sixteen_frames(sixteen_frames_buffer)
                    final_resampled.insert(0, merged_entry)
                    sixteen_frames_buffer = []
            else:
                final_resampled.insert(0, entry)

        # Handle remaining frames in second pass
        if len(sixteen_frames_buffer) > 0:
            merged_entry = self._merge_sixteen_frames(sixteen_frames_buffer)
            final_resampled.insert(0, merged_entry)

        self.entries = final_resampled

    def _merge_four_frames(self, frames: List[TimelineEntry]) -> TimelineEntry:
        # Create a 2x2 grid from 4 frames
        merged_image = self._merge_grid([frame.content for frame in frames], 4)

        # Calculate duration from first to last frame, plus duration of oldest frame
        duration = (frames[-1].timestamp - frames[0].timestamp) + frames[-1].duration

        return TimelineEntry(
            entry_type=EntryType.FOUR_CAMERA_FRAMES
            if frames[0].entry_type == EntryType.CAMERA_FRAME
            else EntryType.FOUR_SCREENSHARE_FRAMES,
            timestamp=frames[0].timestamp,
            content=merged_image,
            duration=duration,
        )

    def _merge_sixteen_frames(self, frames: List[TimelineEntry]) -> TimelineEntry:
        # Create a 4x4 grid from up to four 2x2 grids
        images_data = [
            img for frame in frames for img in self._unmerge_grid(frame.content, 4)
        ]
        merged_image = self._merge_grid(images_data, 16)

        # Calculate duration from first to last frame, plus duration of oldest frame
        duration = (frames[-1].timestamp - frames[0].timestamp) + frames[-1].duration

        return TimelineEntry(
            entry_type=EntryType.SIXTEEN_CAMERA_FRAMES
            if frames[0].entry_type == EntryType.FOUR_CAMERA_FRAMES
            else EntryType.SIXTEEN_SCREENSHARE_FRAMES,
            timestamp=frames[0].timestamp,
            content=merged_image,
            duration=duration,
        )

    def _merge_grid(
        self, frames: List[str], count: Literal[4, 16], output_size: int = 512
    ) -> str:
        grid_size = 2 if count == 4 else 4
        cell_size = output_size // grid_size

        # Create blank canvas
        canvas = Image.new("RGB", (output_size, output_size), "white")

        # Place each frame in the grid
        for idx, frame_url in enumerate(frames):
            # Calculate position in grid
            row = idx // grid_size
            col = idx % grid_size

            # Decode and resize the frame
            frame_img = self._decode_image(frame_url)
            frame_img = frame_img.resize(
                (cell_size, cell_size), Image.Resampling.LANCZOS
            )

            # Calculate position to paste
            x = col * cell_size
            y = row * cell_size

            # Paste into canvas
            canvas.paste(frame_img, (x, y))

        # Convert back to data URL
        buffer = io.BytesIO()
        canvas.save(buffer, format="JPEG")
        image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{image_data}"

    def _unmerge_grid(self, data_url: str, count: Literal[4, 16]) -> List[str]:
        # Reverse of merge_grid - splits a merged grid back into individual frames
        grid_size = 2 if count == 4 else 4
        merged_img = self._decode_image(data_url)
        output_size = merged_img.size[0] // grid_size

        frames = []
        for row in range(grid_size):
            for col in range(grid_size):
                # Calculate coordinates to crop
                left = col * output_size
                top = row * output_size
                right = left + output_size
                bottom = top + output_size

                # Crop and encode individual frame
                frame = merged_img.crop((left, top, right, bottom))
                buffer = io.BytesIO()
                frame.save(buffer, format="JPEG")
                frame_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
                frames.append(f"data:image/jpeg;base64,{frame_data}")

        return frames

    def _decode_image(self, data_url: str) -> Image:
        base64_data = data_url.split(",")[1]
        image_data = base64.b64decode(base64_data)
        return Image.open(io.BytesIO(image_data))

    def _split_speech_with_timestamps(
        self, text: str, end_timestamp: float
    ) -> List[Tuple[str, float, float]]:
        sentences = self.sentence_tokenizer.tokenize(text)

        result = []
        current_time = end_timestamp

        for sentence in reversed(sentences):
            words = self.word_tokenizer.tokenize(sentence)
            hyphenated_words = [hyphenate_word(word) for word in words]
            total_hyphens = sum(len(parts) for parts in hyphenated_words)
            sentence_duration = total_hyphens / HYPHEN_SPEECH_RATE
            sentence_timestamp = current_time
            result.insert(0, (sentence, sentence_timestamp, sentence_duration))

            current_time -= sentence_duration

        return result

    def add_user_speech(self, text: str):
        ts = time.time()

        i = len(self.entries) - 1
        while i >= 0:
            entry = self.entries[i]
            if entry.entry_type == EntryType.ASSISTANT_SPEECH:
                break
            if entry.entry_type == EntryType.USER_SPEECH:
                self.entries.pop(i)
            i -= 1

        sentence_chunks = self._split_speech_with_timestamps(text, ts)

        for sentence, sentence_ts, sentence_duration in sentence_chunks:
            entry = TimelineEntry(
                entry_type=EntryType.USER_SPEECH,
                timestamp=sentence_ts,
                content=sentence,
                duration=sentence_duration,
            )
            self._insert_entry_chronologically(entry)

    def add_camera_frame(self, frame: rtc.VideoFrame):
        ts = time.time()
        if not self._should_add_frame():
            return

        frame_ref = self._encode_frame(frame)

        entry = TimelineEntry(
            entry_type=EntryType.CAMERA_FRAME, timestamp=ts, content=frame_ref
        )
        self._insert_entry_chronologically(entry)

    def add_screenshare_frame(self, frame: rtc.VideoFrame):
        ts = time.time()
        if not self._should_add_frame():
            return

        frame_ref = self._encode_frame(frame)

        entry = TimelineEntry(
            entry_type=EntryType.SCREENSHARE_FRAME, timestamp=ts, content=frame_ref
        )
        self._insert_entry_chronologically(entry)

    def add_assistant_speech(self, text: str):
        ts = time.time()

        sentence_chunks = self._split_speech_with_timestamps(text, ts)

        for sentence, sentence_ts, sentence_duration in sentence_chunks:
            entry = TimelineEntry(
                entry_type=EntryType.ASSISTANT_SPEECH,
                timestamp=sentence_ts,
                content=sentence,
                duration=sentence_duration,
            )
            self._insert_entry_chronologically(entry)

    def _should_add_frame(self) -> bool:
        timestamp = time.time()

        if self.is_speaking:
            interval = 1 / SPEAKING_FRAME_RATE
        else:
            interval = 1 / NOT_SPEAKING_FRAME_RATE

        for entry in reversed(self.entries):
            if entry.entry_type in [
                EntryType.CAMERA_FRAME,
                EntryType.SCREENSHARE_FRAME,
            ]:
                if timestamp - entry.timestamp <= interval:
                    return timestamp - entry.timestamp >= interval
                break

        return True

    def _encode_frame(self, frame: rtc.VideoFrame) -> str:
        encoded_data = images.encode(
            frame,
            images.EncodeOptions(
                format="JPEG",
                resize_options=images.ResizeOptions(
                    width=512,
                    height=512,
                    strategy="center_aspect_fit",
                ),
            ),
        )
        return (
            f"data:image/jpeg;base64,{base64.b64encode(encoded_data).decode('utf-8')}"
        )
