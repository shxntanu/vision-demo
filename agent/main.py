from livekit.agents import JobContext, WorkerOptions, AutoSubscribe, cli
from vision_assistant import VisionAssistant

from dotenv import load_dotenv

load_dotenv()


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)

    assistant = VisionAssistant()
    assistant.start(ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
