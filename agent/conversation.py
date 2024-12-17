import base64
import io
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Literal, Tuple
import os

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
    end_timestamp: float
    content: str
    duration: float = 0.0


@dataclass
class FrameRate:
    speaking: float
    not_speaking: float


SPEAKING_FRAME_RATE = 1.0
NOT_SPEAKING_FRAME_RATE = 0.5
FOUR_FRAME_PACKING_CUTOFF = 8.0
SIXTEEN_FRAME_PACKING_CUTOFF = 45.0

HYPHEN_SPEECH_RATE = 3.83  # hyphens per second

JPEG_QUALITY = 50


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

    def add_user_speech(self, text: str):
        ts = time.time()

        # Due to intteruption handling, we may need to remove existing trailing user speech entries
        # as they will be duplicated in the new entry
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
                end_timestamp=sentence_ts,
                content=sentence,
                duration=sentence_duration,
            )
            self._insert_entry_chronologically(entry)

    def add_assistant_speech(self, text: str):
        ts = time.time()

        sentence_chunks = self._split_speech_with_timestamps(text, ts)

        for sentence, sentence_ts, sentence_duration in sentence_chunks:
            entry = TimelineEntry(
                entry_type=EntryType.ASSISTANT_SPEECH,
                end_timestamp=sentence_ts,
                content=sentence,
                duration=sentence_duration,
            )
            self._insert_entry_chronologically(entry)

    def add_camera_frame(self, frame: rtc.VideoFrame):
        ts = time.time()
        if not self._sample_frame():
            return

        frame_ref = self._encode_frame(frame)

        entry = TimelineEntry(
            entry_type=EntryType.CAMERA_FRAME, end_timestamp=ts, content=frame_ref
        )
        self._insert_entry_chronologically(entry)

    def add_screenshare_frame(self, frame: rtc.VideoFrame):
        ts = time.time()
        if not self._sample_frame():
            return

        frame_ref = self._encode_frame(frame)

        entry = TimelineEntry(
            entry_type=EntryType.SCREENSHARE_FRAME, end_timestamp=ts, content=frame_ref
        )
        self._insert_entry_chronologically(entry)

    # Packs older frames into grids, to reduce token usage while maintaining visual context
    def repack(self):
        now = time.time()

        # First unpack any existing grids back into individual frames
        unpacked = []
        for entry in self.entries:
            if entry.entry_type in [
                EntryType.SIXTEEN_CAMERA_FRAMES,
                EntryType.SIXTEEN_SCREENSHARE_FRAMES,
            ]:
                # Unpack 4x4 grid back into individual frames
                unpacked.extend(self._unpack_sixteen_frames(entry))
            elif entry.entry_type in [
                EntryType.FOUR_CAMERA_FRAMES,
                EntryType.FOUR_SCREENSHARE_FRAMES,
            ]:
                # Unpack 2x2 grid back into individual frames
                unpacked.extend(self._unpack_four_frames(entry))
            else:
                unpacked.append(entry)

        four_packed = []
        four_frames_buffer = []

        # First pass will pack raw frames into 2x2 grids
        for entry in reversed(unpacked):
            age = now - entry.end_timestamp

            if entry.entry_type not in [
                EntryType.CAMERA_FRAME,
                EntryType.SCREENSHARE_FRAME,
            ]:
                four_packed.insert(0, entry)
                continue

            if age > FOUR_FRAME_PACKING_CUTOFF:
                four_frames_buffer.append(entry)
                if len(four_frames_buffer) == 4:
                    packed_entry = self._pack_four_frames(
                        list(reversed(four_frames_buffer))
                    )
                    four_packed.insert(0, packed_entry)
                    four_frames_buffer = []
            else:
                four_packed.insert(0, entry)

        # Handle anything left in the buffer
        if len(four_frames_buffer) > 0:
            packed_entry = self._pack_four_frames(list(reversed(four_frames_buffer)))
            four_packed.insert(0, packed_entry)

        # Second pass will pack 2x2 grids into 4x4 grids
        sixteen_packed = []
        sixteen_frames_buffer = []

        for entry in reversed(four_packed):
            age = now - entry.end_timestamp

            if entry.entry_type not in [
                EntryType.FOUR_CAMERA_FRAMES,
                EntryType.FOUR_SCREENSHARE_FRAMES,
            ]:
                sixteen_packed.insert(0, entry)
                continue

            if age > SIXTEEN_FRAME_PACKING_CUTOFF:
                sixteen_frames_buffer.append(entry)
                if len(sixteen_frames_buffer) == 4:
                    packed_entry = self._pack_sixteen_frames(
                        list(reversed(sixteen_frames_buffer))
                    )
                    sixteen_packed.insert(0, packed_entry)
                    sixteen_frames_buffer = []
            else:
                sixteen_packed.insert(0, entry)

        # Handle anything left in the buffer
        if len(sixteen_frames_buffer) > 0:
            packed_entry = self._pack_sixteen_frames(
                list(reversed(sixteen_frames_buffer))
            )
            sixteen_packed.insert(0, packed_entry)

        self.entries = sixteen_packed

        if os.getenv('DEBUG'):
            end_time = time.time()
            entry_count = 0
            frame_count = 0
            for entry in self.entries:
                if entry.entry_type in [
                    EntryType.CAMERA_FRAME,
                    EntryType.SCREENSHARE_FRAME,
                ]:
                    entry_count += 1
                    frame_count += 1
                elif entry.entry_type in [
                    EntryType.FOUR_CAMERA_FRAMES,
                    EntryType.FOUR_SCREENSHARE_FRAMES,
                ]:
                    entry_count += 1
                    frame_count += 4
                elif entry.entry_type in [
                    EntryType.SIXTEEN_CAMERA_FRAMES,
                    EntryType.SIXTEEN_SCREENSHARE_FRAMES,
                ]:
                    entry_count += 1
                    frame_count += 16
            print(f"Repack took {(end_time - now):.3f}s. {frame_count} frames packed into {entry_count} entries")

    def _insert_entry_chronologically(self, entry: TimelineEntry):
        left, right = 0, len(self.entries)
        while left < right:
            mid = (left + right) // 2
            if self.entries[mid].end_timestamp < entry.end_timestamp:
                left = mid + 1
            else:
                right = mid
        self.entries.insert(left, entry)

    def _sample_frame(self) -> bool:
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
                if timestamp - entry.end_timestamp <= interval:
                    return timestamp - entry.end_timestamp >= interval
                break

        return True

    def _encode_frame(self, frame: rtc.VideoFrame) -> str:
        encoded_data = images.encode(
            frame,
            images.EncodeOptions(
                format="JPEG",
                quality=JPEG_QUALITY,
                resize_options=images.ResizeOptions(
                    width=512,
                    height=512,
                    strategy="scale_aspect_fit",
                ),
            ),
        )
        frame_data = base64.b64encode(encoded_data).decode("utf-8")
        return f"data:image/jpeg;base64,{frame_data}"

    def _unpack_four_frames(self, entry: TimelineEntry) -> List[TimelineEntry]:
        image_urls = self._unpack_grid(entry.content, 4)

        # Calculate timestamps that evenly distribute across the total duration
        frame_duration = entry.duration / len(image_urls)
        start_time = entry.end_timestamp - entry.duration

        entries = []
        for i, image_url in enumerate(image_urls):
            frame_end_timestamp = start_time + ((i + 1) * frame_duration)
            entries.append(
                TimelineEntry(
                    entry_type=EntryType.CAMERA_FRAME
                    if entry.entry_type == EntryType.FOUR_CAMERA_FRAMES
                    else EntryType.SCREENSHARE_FRAME,
                    end_timestamp=frame_end_timestamp,
                    content=image_url,
                    duration=frame_duration,
                )
            )
        return entries

    def _unpack_sixteen_frames(self, entry: TimelineEntry) -> List[TimelineEntry]:
        image_urls = self._unpack_grid(entry.content, 16)

        # Use same timestamp logic as _unpack_four_frames
        frame_duration = entry.duration / len(image_urls)
        start_time = entry.end_timestamp - entry.duration

        entries = []
        for i, image_url in enumerate(image_urls):
            frame_end_timestamp = start_time + ((i + 1) * frame_duration)
            entries.append(
                TimelineEntry(
                    entry_type=EntryType.CAMERA_FRAME
                    if entry.entry_type == EntryType.SIXTEEN_CAMERA_FRAMES
                    else EntryType.SCREENSHARE_FRAME,
                    end_timestamp=frame_end_timestamp,
                    content=image_url,
                    duration=frame_duration,
                )
            )
        return entries

    def _pack_four_frames(self, frames: List[TimelineEntry]) -> TimelineEntry:
        # Create a 2x2 grid from 4 frames
        packed_image = self._pack_grid([frame.content for frame in frames], 4)

        # Use the end timestamp of the newest frame
        end_timestamp = frames[-1].end_timestamp
        # Get the start time of the oldest frame
        start_of_earliest = frames[0].end_timestamp - frames[0].duration
        # Calculate total duration
        duration = end_timestamp - start_of_earliest

        return TimelineEntry(
            entry_type=EntryType.FOUR_CAMERA_FRAMES
            if frames[0].entry_type == EntryType.CAMERA_FRAME
            else EntryType.FOUR_SCREENSHARE_FRAMES,
            end_timestamp=end_timestamp,
            content=packed_image,
            duration=duration,
        )

    def _pack_sixteen_frames(self, frames: List[TimelineEntry]) -> TimelineEntry:
        images_data = [
            img for frame in frames for img in self._unpack_grid(frame.content, 4)
        ]
        packed_image = self._pack_grid(images_data, 16)

        # Use the end timestamp of the newest frame
        end_timestamp = frames[-1].end_timestamp
        # Get the start time of the oldest frame
        start_of_earliest = frames[0].end_timestamp - frames[0].duration
        # Calculate total duration
        duration = end_timestamp - start_of_earliest

        return TimelineEntry(
            entry_type=EntryType.SIXTEEN_CAMERA_FRAMES
            if frames[0].entry_type == EntryType.FOUR_CAMERA_FRAMES
            else EntryType.SIXTEEN_SCREENSHARE_FRAMES,
            end_timestamp=end_timestamp,
            content=packed_image,
            duration=duration,
        )

    def _pack_grid(
        self, frames: List[str], count: Literal[4, 16], output_size: int = 512
    ) -> str:
        grid_size = 2 if count == 4 else 4
        cell_size = output_size // grid_size

        canvas = Image.new("RGB", (output_size, output_size), "white")

        for idx, frame_url in enumerate(frames):
            row = idx // grid_size
            col = idx % grid_size

            frame_img = self._decode_image(frame_url)
            frame_img = frame_img.resize(
                (cell_size, cell_size), Image.Resampling.LANCZOS
            )

            x = col * cell_size
            y = row * cell_size

            canvas.paste(frame_img, (x, y))

        buffer = io.BytesIO()
        canvas.save(buffer, format="JPEG", quality=JPEG_QUALITY)
        image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{image_data}"

    def _unpack_grid(self, data_url: str, count: Literal[4, 16]) -> List[str]:
        grid_size = 2 if count == 4 else 4
        packed_img = self._decode_image(data_url)
        output_size = packed_img.size[0] // grid_size

        frames = []
        for row in range(grid_size):
            for col in range(grid_size):
                left = col * output_size
                top = row * output_size
                right = left + output_size
                bottom = top + output_size

                frame = packed_img.crop((left, top, right, bottom))

                if not self._is_frame_empty(frame):
                    buffer = io.BytesIO()
                    frame.save(buffer, format="JPEG", quality=JPEG_QUALITY)
                    frame_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    frames.append(f"data:image/jpeg;base64,{frame_data}")

        return frames

    def _is_frame_empty(self, frame: Image) -> bool:
        # Check center portion of frame (middle 60%) to control for edge artifacts
        output_size = frame.size[0]
        inset = int(output_size * 0.2)
        center = frame.crop((inset, inset, output_size - inset, output_size - inset))
        extrema = center.convert("L").getextrema()
        return extrema[0] >= 245 and extrema[1] >= 245

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
