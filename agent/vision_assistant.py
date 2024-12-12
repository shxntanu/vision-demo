from livekit import rtc, agents
from openai import AsyncClient as OpenAIAsyncClient
from livekit.plugins import silero, openai, deepgram, cartesia
import asyncio
import time
import httpx
from conversation import ConversationTimeline, EntryType
import re
from typing import AsyncIterable
from debug import debug_chat_context
import os

class VisionAssistant:
    def __init__(self):
        self._agent = None
        self._conversation = ConversationTimeline()
        self._openai_client = OpenAIAsyncClient(
            max_retries=0,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=15.0,
                    read=60.0,
                    write=30.0,
                    pool=60.0
                ),
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
            before_tts_cb=self._on_before_tts
            
        )

        # Add event listeners for speech events
        self._agent.on("user_started_speaking", self._on_user_started_speaking)
        self._agent.on("user_stopped_speaking", self._on_user_stopped_speaking)
        self._agent.on("agent_speech_committed", self._on_agent_speech_committed)
        self._agent.on("agent_speech_interrupted", self._on_agent_speech_interrupted)

        self._agent.start(room)
        
    async def _on_before_tts(self, agent: agents.pipeline.VoicePipelineAgent, text: str | AsyncIterable[str]):
        return self._remove_timestamps(text)

    async def _on_before_llm(self, agent: agents.pipeline.VoicePipelineAgent, chat_ctx: agents.llm.ChatContext):
        if chat_ctx.messages and chat_ctx.messages[-1].role == "user":
            self._conversation.add_user_speech(chat_ctx.messages[-1].content)
            
        self._conversation.resample_history()
        
        # Clear existing messages but keep the chat context instance
        chat_ctx.messages.clear()
        
        # Add the system prompt
        chat_ctx.append(
            text="""You are a helpful assistant with the ability to see, hear, and speak, detailed below.
            
            # Capabilities
            
            ## Speaking
            
            Your output will be played to the user via Text-to-Speech technology. They will not see the text version of your output, they will only hear it. Ensure that your output is concise, formatted for speech, and does not use unpronounceable punctuation or other formatting.
            
            ## Hearing
            
            The user is speaking into a Speech-to-Text system. You will receive a transcript of their speech. They have not seen this transcript, and it may not be perfectly accurate.
            
            ## Seeing
            
            The user may enable their camera or share their device's screen. You will receive video frames interlaced with the conversation.  Interpret images as a sequential video feed, rather than as static individual images.
            
            During the course of this conversation, the video feed will be dynamically resampled. You will receive more frames when the user is speaking or for more recent parts of the conversation. Older parts of the conversation will include fewer frames, to optimize the context window as you are a large language model.
            
            # Conversation Format
            
            The conversation is provided to you with timestamps on every item, relative to the start of the conversation. This is to help you understand the video feed in context.
            
            You do not need to respond with timestamps of your own.
            
            # Your Role
            
            You are a helpful assistant and will assist the user with whatever they require while maintaing a friendly, and cheering demeanor. You were created by LiveKit as a technology demonstration.
            """,
            role="system"
        )

        first_entry_time = self._conversation.entries[0].timestamp if self._conversation.entries else time.time()
        for entry in self._conversation.entries:
            relative_time = round(entry.timestamp - first_entry_time, 1)
            time_prefix = f"[{relative_time}s] "
            
            if entry.entry_type == EntryType.USER_SPEECH:
                chat_ctx.append(text=time_prefix + entry.content, role="user")
            elif entry.entry_type == EntryType.ASSISTANT_SPEECH:
                chat_ctx.append(text=time_prefix + entry.content, role="assistant")
            elif entry.entry_type == EntryType.CAMERA_FRAME:
                chat_ctx.append(
                    text=time_prefix + "New Video Frame (Camera): ",
                    images=[agents.llm.ChatImage(image=entry.content)],
                    role="user"
                )
            elif entry.entry_type == EntryType.SCREENSHARE_FRAME:
                chat_ctx.append(
                    text=time_prefix + "New Video Frame (Screenshare): ",
                    images=[agents.llm.ChatImage(image=entry.content)],
                    role="user"
                )
                
            
        if os.getenv("DEBUG"):
            debug_chat_context(chat_ctx)

    def _on_user_started_speaking(self):
        self._conversation.is_speaking = True
        
    def _on_user_stopped_speaking(self):
        self._conversation.is_speaking = False

    def _on_agent_speech_committed(self, message: agents.llm.ChatMessage):
        self._conversation.add_assistant_speech(self._remove_timestamps(message.content))

    def _on_agent_speech_interrupted(self, message: agents.llm.ChatMessage):
        self._conversation.add_assistant_speech(self._remove_timestamps(message.content))

    def _on_track_subscribed(self, track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            asyncio.create_task(self._handle_video_track(track, publication.source == rtc.TrackSource.SOURCE_SCREENSHARE))
            
    # Sometimes the LLM will respond with its own timestamps. We need to strip them.
    def _remove_timestamps(self, text: str | AsyncIterable[str]):
        if isinstance(text, str):
            return re.sub(r'^\s*\[[^\]]+\]', '', text)
        
        async def process_stream():
            can_remove_timestamp = True
            is_timestamp = False
            async for chunk in text:
                if not can_remove_timestamp:
                    yield chunk
                    continue
                
                for i, char in enumerate(chunk):
                    if is_timestamp:
                        if char == ']':
                            is_timestamp = False
                            can_remove_timestamp = False
                            remaining = chunk[i+1:]
                            yield remaining
                            break
                    elif char == '[':
                        is_timestamp = True
                    elif char != ' ':
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
