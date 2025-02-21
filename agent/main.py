import asyncio
import logging

from dotenv import load_dotenv
from google.genai.types import (
    Blob,
    LiveClientRealtimeInput,
)
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
    multimodal,
)
from livekit.agents.utils import images
from livekit.plugins import google
from livekit.rtc import Track, TrackKind, VideoStream

load_dotenv()

logger = logging.getLogger("vision-assistant")
logger.setLevel(logging.INFO)

SPEAKING_FRAME_RATE = 1.0  # frames per second when speaking
NOT_SPEAKING_FRAME_RATE = 0.5  # frames per second when not speaking
JPEG_QUALITY = 50

_SYSTEM_PROMPT = """
You are a powerful assistant with the ability to see, hear, and speak. You were created by LiveKit as a technology demonstration.

# Your Capabilities

## Sight
You can receive live video frames from the user's camera or screen share.

## Speech
Your responses should be concise and natural, avoiding unpronounceable punctuation or formatting since they will be spoken to the user.

## Hearing  
You receive live audio from the user's microphone.

## Memory
You experience the conversation with a sense of time and context. Pay close attention to the video feed as you may be asked about events from several minutes ago.
"""

async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    participant = await ctx.wait_for_participant()
    chat_ctx = llm.ChatContext()
    model = google.beta.realtime.RealtimeModel(
        voice="Puck",
        temperature=0.8,
        instructions=_SYSTEM_PROMPT,
    )   
    agent = multimodal.MultimodalAgent(
        model=model,            
        chat_ctx=chat_ctx,
    )
    agent.start(ctx.room, participant)
    
    async def handle_video_track(track: Track):
        logger.info("Handling video track")
        video_stream = VideoStream(track)
        last_frame_time = 0  # Track the last time we processed a frame
        frame_counter = 0  # Add counter for unique filenames
        
        async for event in video_stream:
            current_time = asyncio.get_event_loop().time()
            
            if current_time - last_frame_time < _get_frame_interval():
                continue
                
            last_frame_time = current_time
            frame = event.frame
            
            encoded_data = images.encode(
                frame,
                images.EncodeOptions(
                    format="JPEG",
                    quality=JPEG_QUALITY,
                    resize_options=images.ResizeOptions(
                        width=1024,
                        height=1024,
                        strategy="scale_aspect_fit",
                    ),
                ),
            )
            
            frame_counter += 1
            
            realtime_input = LiveClientRealtimeInput(
                media_chunks=[Blob(data=encoded_data, mime_type="image/jpeg")],
            )
            
            try:
                model.sessions[0]._queue_msg(realtime_input)
                logger.info(f"Queued frame {frame_counter}")
            except Exception as e:
                logger.error(f"Error queuing frame {frame_counter}: {e}")
        await video_stream.aclose()

    ctx.room.on("track_subscribed", lambda track, pub, participant: 
        asyncio.create_task(handle_video_track(track)) if track.kind == TrackKind.KIND_VIDEO else None
    )
    
    def _get_frame_interval(self) -> float:
        return 1.0 / (SPEAKING_FRAME_RATE if self.is_speaking else NOT_SPEAKING_FRAME_RATE)
    

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
