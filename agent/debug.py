import os
from datetime import datetime
from typing import List

from livekit.agents.llm import ChatImage, ChatMessage


def debug_chat_context(chat_context: List[ChatMessage]):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = os.path.join(os.path.dirname(__file__), "debug")
    os.makedirs(debug_dir, exist_ok=True)
    html_path = os.path.join(debug_dir, f"{timestamp}.html")

    # Create HTML content with same styling as before
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chat Context Debug</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 20px auto;
                padding: 0 20px;
                background: #f5f5f5;
            }
            .entry {
                background: white;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .header {
                color: #444;
                font-size: 0.9em;
                margin-bottom: 8px;
            }
            .timestamp {
                color: #666;
                font-size: 0.8em;
            }
            .content {
                margin-top: 10px;
            }
            .content img {
                max-width: 300px;
                height: auto;
                display: block;
                margin: 10px 0;
            }
            .error {
                color: red;
                font-size: 0.9em;
            }
        </style>
    </head>
    <body>
    """

    html_content += f"""
        <h1>Chat Context Debug - {timestamp}</h1>
    """

    for msg in chat_context.messages:
        html_content += f"""
        <div class="entry">
            <div class="header">
                <strong>{msg.role}</strong>
            </div>
            <div class="content">
        """

        content_list = msg.content if isinstance(msg.content, list) else [msg.content]

        for content in content_list:
            if isinstance(content, ChatImage):
                try:
                    if isinstance(content.image, str) and content.image.startswith(
                        "data:image/jpeg;base64,"
                    ):
                        html_content += f'<img src="{content.image}" alt="Image">'
                    else:
                        html_content += "<p>[Other image data]</p>"
                except Exception as e:
                    html_content += (
                        f'<p class="error">Error processing image: {str(e)}</p>'
                    )
            else:
                html_content += f"<p>{content}</p>"

        html_content += """
            </div>
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\nDebug HTML saved to: {html_path}")
