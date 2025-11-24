import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")


def get_connection():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )


def create_response_record():
    """Insert a new row with placeholder values and return the response ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO llm_results (prompt, response) VALUES (%s, %s)",
        ("PROCESSING", None)
    )
    response_id = cursor.lastrowid
    conn.commit()
    cursor.close()
    conn.close()
    return response_id


def save_response(response_id, prompt, response):
    """Update the database row when processing is done."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE llm_results SET prompt=%s, response=%s WHERE id=%s",
        (prompt, response, response_id)
    )
    conn.commit()
    cursor.close()
    conn.close()
