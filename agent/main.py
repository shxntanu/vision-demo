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

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()
    chat_ctx = llm.ChatContext()
    # await agent.start(room=ctx.room)
    agent = multimodal.MultimodalAgent(
        model=google.beta.realtime.RealtimeModel(
            voice="Puck",
            temperature=0.8,
            instructions="You are a helpful assistant that can see, hear, and speak.",
        ),
        # fnc_ctx=fnc_ctx,
        chat_ctx=chat_ctx,
    )
    agent.start(ctx.room, participant)
    
    # Add video track handling
    async def handle_video_track(track: Track):
        video_stream = VideoStream(track)
        async for event in video_stream:
            frame = event.frame
            realtime_input = LiveClientRealtimeInput(
                media_chunks=[Blob(data=frame.data.tobytes(), mime_type="image/raw")],
            )
            # Push to first session's queue
            agent.model.sessions[0]._queue_msg(realtime_input)
        await video_stream.aclose()

    # Subscribe to video tracks
    ctx.room.on("track_subscribed", lambda track, pub, participant: 
        asyncio.create_task(handle_video_track(track)) if track.kind == TrackKind.KIND_VIDEO else None
    )
    
    agent.generate_reply()
    
    

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
