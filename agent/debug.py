import os
from datetime import datetime
from typing import List

from livekit.agents.llm import ChatImage, ChatMessage


# Dumps the ChatContext to an HTML file for debugging purposes
def dump_chat_context_to_html(chat_context: List[ChatMessage]):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = os.path.join(os.path.dirname(__file__), "debug")
    os.makedirs(debug_dir, exist_ok=True)
    html_path = os.path.join(debug_dir, f"{timestamp}.html")

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chat Context Debug</title>
        <style>
            body {
                max-width: 1024px;
                margin: 24px auto;
                padding: 0 24px;
                background: #f5f5f5;
            }
            
            .messages {
                display: flex;
                flex-direction: column;
            }
            
            body.reversed .messages {
                flex-direction: column-reverse;
            }
            
            .message {
                background: white;
                padding: 16px;
                margin: 16px 0;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            
            .sort-button {
                padding: 8px 16px;
                background: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                margin: 16px 0;
            }
            
            body:not(.reversed) .sort-button::before {
                content: "Sort Descending";
            }
            
            body.reversed .sort-button::before {
                content: "Sort Ascending";
            }
            
            .sort-button:hover {
                background: #0056b3;
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
            
            pre {
                white-space: pre;
                overflow-x: auto;
                max-width: 100%;
                border-radius: 4px;
            }
        </style>
    </head>
    <body>
    """

    html_content += f"""
    <h1>ChatContext - {timestamp}</h1>
    <button class="sort-button" onclick="document.body.classList.toggle('reversed');"></button>
    <div class="messages">
    """

    for msg in chat_context.messages:
        html_content += f"""
        <div class="message">
            <strong>{msg.role}</strong>
        """

        content_list = msg.content if isinstance(msg.content, list) else [msg.content]

        for content in content_list:
            if isinstance(content, ChatImage):
                html_content += f'<img src="{content.image}" alt="{content.image}" />'
            else:
                html_content += f"<pre>{content}</pre>"

        html_content += """
        </div>
        """

    html_content += """
    </div>
    </body>
    </html>
    """

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\nDebug HTML saved to: {html_path}")
