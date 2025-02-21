import logging

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
    agent.generate_reply()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
