from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_classic.memory import ConversationBufferMemory
from shoppinggpt.tool.product_search import product_search_tool
from shoppinggpt.tool.policy_search import policy_search_tool
from langchain_classic.prompts import ChatPromptTemplate
from shoppinggpt.logging_utils import get_logger


logger = get_logger(__name__)
LARGE_PRODUCT_LIST_THRESHOLD = 20


def _format_agent_output(output) -> str:
    logger.info(f"Formatting agent output: {output}")
    if isinstance(output, list):
        parts = []
        for item in output:
            if isinstance(item, dict):
                if item.get("type") == "thinking":
                    continue
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]).strip())
            elif isinstance(item, str):
                parts.append(item.strip())
        text = "\n".join(part for part in parts if part)
    else:
        text = str(output).strip()

    if not text:
        return "Xin lỗi, tôi chưa có câu trả lời phù hợp."

    lines = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
    if len(lines) == 1:
        return lines[0]

    return "\n".join(f"- {line}" for line in lines)


def _format_product_rows(rows) -> str:
    if not rows:
        return "Xin lỗi, tôi chưa có câu trả lời phù hợp."

    lines = [
        "Chào bạn, dưới đây là danh sách sản phẩm:",
        "",
        "| STT | Mã SP | Tên sản phẩm | Giá (VNĐ) |",
        "|---:|---|---|---:|",
    ]

    for index, row in enumerate(rows, start=1):
        product_code = row.get("product_code", "-")
        product_name = row.get("product_name", "-")
        price = row.get("price", "-")
        if isinstance(price, (int, float)):
            price = f"{price:,.0f}".replace(",", ".")
        lines.append(f"| {index} | {product_code} | {product_name} | {price} |")

    return "\n".join(lines)


def _extract_product_rows(intermediate_steps):
    for action, observation in intermediate_steps or []:
        if getattr(action, "tool", "") == "product_search_tool" and isinstance(observation, list):
            return observation
    return None


class ShoppingAgent:
    def __init__(self, llm, shared_memory: ConversationBufferMemory):
        self.llm = llm
        self.verbose = False
        self.memory = shared_memory
        self.tools = [product_search_tool, policy_search_tool]
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent and helpful AI assistant for an online fashion store.
            Your task is to answer customer questions about products and store policies.
            Use the available tools to search for accurate information and provide appropriate answers.
                      
            Always use Vietnamese to communicate with customers."""),
            ("human", "{input}"),
            ("ai", "{agent_scratchpad}")
        ])

    def invoke(self, query: str) -> str:
        logger.info("Invoking shopping agent")
        inputs = {
            "input": query,
        }
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            handle_parsing_errors=True,
            memory=self.memory,
            return_intermediate_steps=True,
        )
        ai_message = agent_executor.invoke(inputs)
        logger.info(f"Raw agent output: {ai_message}")
        agent_output = ai_message.get('output')
        product_rows = _extract_product_rows(ai_message.get('intermediate_steps'))

        if isinstance(product_rows, list) and len(product_rows) >= LARGE_PRODUCT_LIST_THRESHOLD:
            logger.info("Bypassing LLM rewrite for large product list rows=%s", len(product_rows))
            return _format_product_rows(product_rows)

        logger.info("Shopping agent completed with output length=%s", len(agent_output) if agent_output is not None else 0)
        return _format_agent_output(agent_output)
