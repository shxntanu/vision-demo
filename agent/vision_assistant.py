import asyncio
import os
import re
import time
from typing import AsyncIterable

import httpx
from livekit import agents, rtc
from livekit.plugins import cartesia, deepgram, openai, silero
from openai import AsyncClient as OpenAIAsyncClient

from conversation import ConversationTimeline, EntryType
from debug import dump_chat_context_to_html

_SYSTEM_PROMPT = """
You are a powerful assistant with the ability to see, hear, and speak. You were created by LiveKit as a technology demonstration.

# Your Capabilities

## Sight

You can receive live video frames interlaced with other content. Interpret these images as a sequential video feed, rather than as static individual images.

Some video frames may be packed into grids. These should still be understood as time-ordered thumbnails from a live video feed.

The user may choose to publish their camera or share their device's screen in this way, as a live video feed.

## Speech

Your responses are played to the user via Text-to-Speech technology. Your output is concise and does not use unpronounceable punctuation or other formatting.

## Hearing

The user is speaking into a Speech-to-Text system. You receive a transcript of their speech as input, alongside their live video feed.

## Memory

You experience the conversation with a sense of time, and each item is tagged with a timestamp relative to the start of the conversation.

This allows you to understand the live video feed in context, and refer to and remember previous items from the conversation as well as previous video frames.

You will pay close attention to the historical video feed, as you may be asked questions about events depicted in video frames that are several seconds old, or more.
"""


class VisionAssistant:
    def __init__(self):
        self._agent = None
        self._conversation = ConversationTimeline()
        self._openai_client = OpenAIAsyncClient(
            max_retries=0,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=60.0, write=30.0, pool=60.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=50,
                    keepalive_expiry=120,
                ),
            ),
        )

    def start(self, room: rtc.Room):
        room.on("track_subscribed", self._on_track_subscribed)

        self._agent = agents.pipeline.VoicePipelineAgent(
            vad=silero.VAD.load(),
            stt=deepgram.STT(model="nova-2-general"),
            llm=openai.LLM(model="gpt-4o-mini", client=self._openai_client),
            tts=cartesia.TTS(voice="248be419-c632-4f23-adf1-5324ed7dbf1d"),
            before_llm_cb=self._on_before_llm,
            before_tts_cb=self._on_before_tts,
        )

        self._agent.on("user_started_speaking", self._on_user_started_speaking)
        self._agent.on("user_stopped_speaking", self._on_user_stopped_speaking)
        self._agent.on("agent_speech_committed", self._on_agent_speech_committed)
        self._agent.on("agent_speech_interrupted", self._on_agent_speech_interrupted)

        self._agent.start(room)

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            asyncio.create_task(
                self._handle_video_track(
                    track, publication.source == rtc.TrackSource.SOURCE_SCREENSHARE
                )
            )

    def _on_user_started_speaking(self):
        self._conversation.is_speaking = True

    def _on_user_stopped_speaking(self):
        self._conversation.is_speaking = False

    def _on_agent_speech_committed(self, message: agents.llm.ChatMessage):
        self._conversation.add_assistant_speech(
            self._remove_timestamps(message.content)
        )

    def _on_agent_speech_interrupted(self, message: agents.llm.ChatMessage):
        self._conversation.add_assistant_speech(
            self._remove_timestamps(message.content)
        )

    async def _on_before_llm(
        self,
        agent: agents.pipeline.VoicePipelineAgent,
        chat_ctx: agents.llm.ChatContext,
    ):
        self._inject_conversation(chat_ctx)

        if os.getenv("DEBUG"):
            dump_chat_context_to_html(chat_ctx)

    async def _on_before_tts(
        self, agent: agents.pipeline.VoicePipelineAgent, text: str | AsyncIterable[str]
    ):
        return self._remove_timestamps(text)

    def _inject_conversation(self, chat_ctx: agents.llm.ChatContext):
        if chat_ctx.messages and chat_ctx.messages[-1].role == "user":
            self._conversation.add_user_speech(chat_ctx.messages[-1].content)

        chat_ctx.messages.clear()

        chat_ctx.append(
            text=_SYSTEM_PROMPT,
            role="system",
        )

        first_entry_time = (
            self._conversation.entries[0].end_timestamp
            if self._conversation.entries
            else time.time()
        )

        self._conversation.repack()

        for entry in self._conversation.entries:
            relative_time = round(entry.end_timestamp - first_entry_time, 1)
            time_prefix = f"[{relative_time}s] "

            if entry.entry_type == EntryType.USER_SPEECH:
                chat_ctx.append(text=time_prefix + entry.content, role="user")
            elif entry.entry_type == EntryType.ASSISTANT_SPEECH:
                chat_ctx.append(text=time_prefix + entry.content, role="assistant")
            elif (
                entry.entry_type == EntryType.CAMERA_FRAME
                or entry.entry_type == EntryType.SCREENSHARE_FRAME
            ):
                type = (
                    "Camera"
                    if entry.entry_type == EntryType.CAMERA_FRAME
                    else "Screenshare"
                )
                chat_ctx.append(
                    text=time_prefix + f"New Video Frame ({type}): ",
                    images=[
                        agents.llm.ChatImage(
                            image=entry.content, inference_detail="low"
                        )
                    ],
                    role="user",
                )
            elif (
                entry.entry_type == EntryType.FOUR_CAMERA_FRAMES
                or entry.entry_type == EntryType.FOUR_SCREENSHARE_FRAMES
            ):
                type = (
                    "Camera"
                    if entry.entry_type == EntryType.FOUR_CAMERA_FRAMES
                    else "Screenshare"
                )
                chat_ctx.append(
                    text=time_prefix
                    + f"four video frames covering {entry.duration:.1f} seconds ({type}) (ascending left to right, top to bottom): ",
                    images=[
                        agents.llm.ChatImage(
                            image=entry.content, inference_detail="low"
                        )
                    ],
                    role="user",
                )
            elif (
                entry.entry_type == EntryType.SIXTEEN_CAMERA_FRAMES
                or entry.entry_type == EntryType.SIXTEEN_SCREENSHARE_FRAMES
            ):
                type = (
                    "Camera"
                    if entry.entry_type == EntryType.SIXTEEN_CAMERA_FRAMES
                    else "Screenshare"
                )
                chat_ctx.append(
                    text=time_prefix
                    + f"sixteen video frames covering {entry.duration:.1f} seconds ({type}) (ascending left to right, top to bottom): ",
                    images=[
                        agents.llm.ChatImage(
                            image=entry.content, inference_detail="low"
                        )
                    ],
                    role="user",
                )
            else:
                raise ValueError(f"Unknown entry type: {entry.entry_type}")

    # Sometimes the LLM will respond with its own timestamps. We need to strip them.
    def _remove_timestamps(self, text: str | AsyncIterable[str]):
        if isinstance(text, str):
            return re.sub(r"^\s*\[[^\]]+\]", "", text)

        async def process_stream():
            can_remove_timestamp = True
            is_timestamp = False
            async for chunk in text:
                if not can_remove_timestamp:
                    yield chunk
                    continue

                for i, char in enumerate(chunk):
                    if is_timestamp:
                        if char == "]":
                            is_timestamp = False
                            can_remove_timestamp = False
                            remaining = chunk[i + 1 :]
                            yield remaining
                            break
                    elif char == "[":
                        is_timestamp = True
                    elif char != " ":
                        can_remove_timestamp = False
                        yield chunk
                        break

        return process_stream()

    async def _handle_video_track(self, track: rtc.Track, is_screenshare: bool):
        video_stream = rtc.VideoStream(track)
        async for event in video_stream:
            if is_screenshare:
                self._conversation.add_screenshare_frame(event.frame)
            else:
                self._conversation.add_camera_frame(event.frame)

        await video_stream.aclose()
