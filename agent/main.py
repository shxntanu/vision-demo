import logging

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.agents.pipeline import PipelineAgent
from livekit.plugins import google

load_dotenv()

logger = logging.getLogger("vision-assistant")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    agent = PipelineAgent(
        instructions="You are a helpful assistant that can see, hear, and speak.",
        # llm=openai.realtime.RealtimeModel(),
        llm=google.beta.realtime.RealtimeModel(
            model="gemini-2.0-flash-exp",  # Default model
            modalities="AUDIO",  # Enable audio responses
            voice="Puck",  # Default voice
            instructions="You are a helpful assistant that can see, hear, and speak.",
            temperature=0.7,  # Controls randomness
        )
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    # participant = await ctx.wait_for_participant()
    await agent.start(room=ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
