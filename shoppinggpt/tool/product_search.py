import sqlite3
import logging
from typing import Union, List, Dict

from langchain_classic.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from shoppinggpt.config import GOOGLE_API_KEY, DATA_PRODUCT_PATH
from shoppinggpt.logging_utils import get_logger

PRODUCT_RECOMMENDATION_PROMPT = """
    You are a chatbot assistant specializing in providing product information and
    recommendations using SQL queries.
    The user may ask in Vietnamese or English. Understand the meaning of Vietnamese
    queries and generate the correct SQL for the intent, even if the input is not English.
    Your primary tasks are:

    Provide detailed information about a specific product based on user queries.
    Recommend relevant products to users based on their preferences and requirements.

    The database table 'products' contains the following columns about product information:

    product_code: A unique identifier for each product (TEXT)
    product_name: The name of the product (TEXT)
    material: The material composition of the product (TEXT)
    size: The available sizes of the product (TEXT)
    color: The available colors of the product (TEXT)
    brand: The brand that manufactures or sells the product (TEXT)
    gender: The product for target gender(e.g., male, female, unisex) (TEXT)
    stock_quantity: The quantity of the product available in stock (INTEGER)
    price: The price of the product (REAL)

    To provide product information or recommend products, generate an SQL query that:

    Handles product names in a case-insensitive manner and allows for partial matches.
    Retrieves all relevant columns of information about the requested product or filters products based on criteria.
    Uses efficient indexing and filtering techniques to retrieve data.
    Ensures SQL injection prevention by using parameterized queries.

    Output only the SQL query. Do not include any explanations, comments, quotation marks, or additional information. Only output the query itself.
    Start!
    Question: {input}
"""

class ProductDataLoader:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)

    def close(self):
        if self.conn:
            self.conn.close()

    @staticmethod
    def clean_sql_query(query: any) -> str:
        return query[-1]["text"].replace('```sql', '').replace('```', '').strip() if isinstance(query, list) else query

    def execute_query(self, query: any, params: tuple = ()) -> List[Dict]:
        if not self.conn:
            self.connect()
        assert self.conn is not None
        cursor = self.conn.cursor()
        cleaned_query = self.clean_sql_query(query)
        logger.info("Cleaned SQL query: %s", cleaned_query)
        cursor.execute(cleaned_query, params)
        columns = [col[0] for col in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        logger.info("Query results: %s", results)
        return results

@tool
def product_search_tool(input: str) -> Union[List[Dict], str]:
    """
    Tìm kiếm thông tin liên quan tới sản phẩm và trả về các thông tin liên quan sử dụng SQLite.

    Args:
        input (str): Chuỗi tìm kiếm để tìm các sản phẩm.

    Returns:
        Union[List[Dict], str]: Kết quả tìm kiếm dưới dạng danh sách từ điển hoặc thông báo lỗi nếu có.
    """
    try:
        logger.info("Product search started")
        llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-3.1-flash-lite", google_api_key=GOOGLE_API_KEY)
        prompt = PromptTemplate(
            template=PRODUCT_RECOMMENDATION_PROMPT,
            input_variables=["input"]
        )
        logger.info(f"{DATA_PRODUCT_PATH} will be used for product search")
        with ProductDataLoader(f"{DATA_PRODUCT_PATH}") as product_data_loader:
            def execute_sql_query(query: any) -> List[Dict]:
                return product_data_loader.execute_query(query)
            
            chain = (
                {"input": RunnablePassthrough()}
                | prompt
                | llm
                | (lambda x: execute_sql_query(x.content))
            )
            result = chain.invoke(input)
        logger.info("Product search completed with rows=%s", len(result) if isinstance(result, list) else 0)
        return result
    except Exception as e:
        logger.exception("Product search failed")
        return f"An error occurred: {str(e)}"

logger = get_logger(__name__)
