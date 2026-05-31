import json

from flask import Flask, Response, render_template, request, jsonify, stream_with_context
from dotenv import load_dotenv
import os
from uuid import uuid4
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.memory import ConversationBufferMemory
from shoppinggpt.router.lib_semantic_router import (
    SemanticRouter,
    PRODUCT_ROUTE_NAME,
    CHITCHAT_ROUTE_NAME
)
from shoppinggpt.chain import create_chitchat_chain
from shoppinggpt.agent import ShoppingAgent
from shoppinggpt.logging_utils import (
    clear_request_context,
    configure_logging,
    get_logger,
    set_request_context,
)
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# List models
# for model in genai.list_models():
#     print(model.name)

configure_logging()
logger = get_logger(__name__)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# LLM and Embedding setup
LLM = ChatGoogleGenerativeAI(temperature=0, model="gemini-3.1-flash-lite")

# Memory setup
SHARED_MEMORY = ConversationBufferMemory(return_messages=True)

# Initialize SemanticRouter
SEMANTIC_ROUTER = SemanticRouter()

app = Flask(__name__)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _chunk_text(text: str, size: int = 24):
    for index in range(0, len(text), size):
        yield text[index:index + size]

def handle_query(query: str) -> dict:
    """Handle user query and return response."""
    request_id = uuid4().hex[:8]
    set_request_context(request_id=request_id)
    logger.info("Received query: %s", query)

    try:
        guided_route = SEMANTIC_ROUTER.guide(query)
        set_request_context(request_id=request_id, route=guided_route)
        logger.info("Route selected: %s", guided_route)

        if guided_route == CHITCHAT_ROUTE_NAME:
            chitchat_chain = create_chitchat_chain(LLM, SHARED_MEMORY)
            response = chitchat_chain.invoke({"input": query})
        elif guided_route == PRODUCT_ROUTE_NAME:
            agent = ShoppingAgent(LLM, SHARED_MEMORY)
            response = agent.invoke(query)
        else:
            response = "Unknown query type"

        if hasattr(response, 'content'):
            content = response.content
        elif isinstance(response, dict) and 'output' in response:
            content = response['output']
        else:
            content = str(response)

        # Update shared memory
        SHARED_MEMORY.chat_memory.add_user_message(query)
        SHARED_MEMORY.chat_memory.add_ai_message(content)

        logger.info("Completed query with response length=%s", len(content))
        return {
            'response': content,
            'type': guided_route
        }
    except Exception:
        logger.exception("Failed to handle query")
        raise
    finally:
        clear_request_context()


def stream_query(query: str):
    request_id = uuid4().hex[:8]
    set_request_context(request_id=request_id)
    logger.info("Received streaming query: %s", query)

    try:
        yield _sse("status", {"message": "Đang phân tích yêu cầu..."})
        guided_route = SEMANTIC_ROUTER.guide(query)
        set_request_context(request_id=request_id, route=guided_route)
        logger.info("Route selected: %s", guided_route)

        content = ""
        if guided_route == CHITCHAT_ROUTE_NAME:
            chitchat_chain = create_chitchat_chain(LLM, SHARED_MEMORY)
            yield _sse("status", {"message": "Đang tạo câu trả lời..."})
            for chunk in chitchat_chain.stream({"input": query}):
                text = getattr(chunk, "content", "") or ""
                if text:
                    content += text
                    yield _sse("delta", {"text": text})
        elif guided_route == PRODUCT_ROUTE_NAME:
            agent = ShoppingAgent(LLM, SHARED_MEMORY)
            yield _sse("status", {"message": "Đang tìm sản phẩm..."})
            response = agent.invoke(query)
            content = response if isinstance(response, str) else str(response)
            yield _sse("status", {"message": "Đang xuất kết quả..."})
            for chunk in _chunk_text(content):
                yield _sse("delta", {"text": chunk})
        else:
            content = "Unknown query type"
            yield _sse("delta", {"text": content})

        SHARED_MEMORY.chat_memory.add_user_message(query)
        SHARED_MEMORY.chat_memory.add_ai_message(content)

        yield _sse("done", {"type": guided_route})
        logger.info("Completed streaming query with response length=%s", len(content))
    except Exception:
        logger.exception("Failed to stream query")
        yield _sse("error", {"message": "Xin lỗi, đã có lỗi xảy ra khi xử lý yêu cầu."})
    finally:
        clear_request_context()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['GET'])
def get_bot_response():
    user_message = request.args.get('msg')
    response = handle_query(user_message)
    logger.info("HTTP response generated for message")
    return jsonify(response)


@app.route('/stream', methods=['GET'])
def stream_bot_response():
    user_message = request.args.get('msg', '')
    response = Response(
        stream_with_context(stream_query(user_message)),
        mimetype='text/event-stream',
    )
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
