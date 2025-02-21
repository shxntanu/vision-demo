import logging
import asyncio
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm
)
from livekit.agents import multimodal
from livekit.plugins import google
from livekit.rtc import Track, TrackKind, VideoStream
from google.genai.types import (
    Blob,
    LiveClientRealtimeInput,
)
from livekit.agents.utils import images
import base64

load_dotenv()

logger = logging.getLogger("vision-assistant")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    # agent = PipelineAgent(
    #     instructions="You are a helpful assistant that can see, hear, and speak.",
    #     # llm=openai.realtime.RealtimeModel(),
    #     llm=google.beta.realtime.RealtimeModel(
    #         model="gemini-2.0-flash-exp",  # Default model
    #         modalities="AUDIO",  # Enable audio responses
    #         voice="Puck",  # Default voice
    #         instructions="You are a helpful assistant that can see, hear, and speak.",
    #         temperature=0.7,  # Controls randomness
    #     )
    # )

    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    participant = await ctx.wait_for_participant()
    chat_ctx = llm.ChatContext()
    model = google.beta.realtime.RealtimeModel(
        voice="Puck",
        temperature=0.8,
        instructions="You are a helpful assistant that can see, hear, and speak.",
    )   
    # await agent.start(room=ctx.room)
    agent = multimodal.MultimodalAgent(
        model=model,            
        # fnc_ctx=fnc_ctx,
        chat_ctx=chat_ctx,
    )
    agent.start(ctx.room, participant)
    
    # Add video track handling
    async def handle_video_track(track: Track):
        logger.info("Handling video track")
        video_stream = VideoStream(track)
        last_frame_time = 0  # Track the last time we processed a frame
        frame_counter = 0  # Add counter for unique filenames
        
        async for event in video_stream:
            current_time = asyncio.get_event_loop().time()
            
            # Skip if less than 1 second has passed since last frame
            if current_time - last_frame_time < 1.0:
                continue
                
            last_frame_time = current_time
            frame = event.frame
            
            # Encode frame as JPEG
            encoded_data = images.encode(
                frame,
                images.EncodeOptions(
                    format="JPEG",
                    quality=50,
                    resize_options=images.ResizeOptions(
                        width=1024,
                        height=1024,
                        strategy="scale_aspect_fit",
                    ),
                ),
            )
            
            frame_counter += 1
            
            # Base64 encode the JPEG data
            # base64_data = base64.b64encode(encoded_data).decode()
            
            realtime_input = LiveClientRealtimeInput(
                media_chunks=[Blob(data=encoded_data, mime_type="image/jpeg")],
            )
            
            # Push to first session's queue
            try:
                model.sessions[0]._queue_msg(realtime_input)
                logger.info(f"Queued frame {frame_counter}")
            except Exception as e:
                logger.error(f"Error queuing frame {frame_counter}: {e}")
        await video_stream.aclose()

    # Subscribe to video tracks
    ctx.room.on("track_subscribed", lambda track, pub, participant: 
        asyncio.create_task(handle_video_track(track)) if track.kind == TrackKind.KIND_VIDEO else None
    )
    
    # agent.generate_reply()
    
    

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
