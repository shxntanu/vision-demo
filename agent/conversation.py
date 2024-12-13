import base64
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

from livekit import rtc
from livekit.agents.tokenize.basic import (
    SentenceTokenizer,
    WordTokenizer,
    hyphenate_word,
)
from livekit.agents.utils import images


class EntryType(Enum):
    USER_SPEECH = "user_speech"
    CAMERA_FRAME = "camera_frame"
    SCREENSHARE_FRAME = "screenshare_frame"
    ASSISTANT_SPEECH = "assistant_speech"


@dataclass
class TimelineEntry:
    entry_type: EntryType
    timestamp: float  # absolute timestamp in seconds since epoch
    content: str  # text content or frame reference
    duration: float = 0.0


@dataclass
class FrameRate:
    speaking: float
    not_speaking: float


# Ordered from most recent to oldest
FRAME_RATE_WINDOWS = [
    (10.0, FrameRate(2, 0.5)),
    (60.0, FrameRate(0.5, 0.1)),
    (float("inf"), FrameRate(0.2, 0.01)),
]

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

    def resample_history(self):
        now = time.time()

        resampled = []

        last_kept = [None] * len(FRAME_RATE_WINDOWS)

        speaking_after = float("inf")

        for entry in reversed(self.entries):
            age = now - entry.timestamp

            if entry.entry_type not in [
                EntryType.CAMERA_FRAME,
                EntryType.SCREENSHARE_FRAME,
            ]:
                resampled.insert(0, entry)
                if entry.entry_type == EntryType.USER_SPEECH:
                    speaking_after = entry.timestamp - entry.duration
                continue

            for i, (horizon, frame_rate) in enumerate(FRAME_RATE_WINDOWS):
                if age <= horizon:
                    if speaking_after <= entry.timestamp:
                        interval = 1 / frame_rate.speaking
                    else:
                        interval = 1 / frame_rate.not_speaking

                    if (
                        last_kept[i] is None
                        or last_kept[i] - entry.timestamp >= interval
                    ):
                        resampled.insert(0, entry)
                        last_kept[i] = entry.timestamp
                    break

        self.entries = resampled

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

    def _is_speaking_at_timestamp(self, timestamp: float) -> bool:
        for entry in self.entries:
            if entry.entry_type == EntryType.USER_SPEECH:
                speech_start = entry.timestamp - entry.duration
                if speech_start <= timestamp <= entry.timestamp:
                    return True
        return False

    def _should_add_frame(self) -> bool:
        frame_rate = FRAME_RATE_WINDOWS[0][1]
        timestamp = time.time()

        if self.is_speaking:
            interval = 1 / frame_rate.speaking
        else:
            interval = 1 / frame_rate.not_speaking

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
                    strategy="scale_aspect_fit",
                ),
            ),
        )
        return (
            f"data:image/jpeg;base64,{base64.b64encode(encoded_data).decode('utf-8')}"
        )
