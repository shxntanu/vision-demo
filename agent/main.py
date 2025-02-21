import logging

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.agents.pipeline import PipelineAgent
from livekit.plugins import openai

load_dotenv()

logger = logging.getLogger("vision-assistant")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    agent = PipelineAgent(
        instructions="You are a helpful assistant that can see, hear, and speak.",
        llm=openai.realtime.RealtimeModel(),
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    # participant = await ctx.wait_for_participant()
    await agent.start(room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
