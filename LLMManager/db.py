import os
import pymysql
from dotenv import load_dotenv

load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DB = os.getenv("MYSQL_DB")
MYSQL_PORT = int(os.getenv("MYSQL_PORT"))


def get_connection():
    return pymysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        port=MYSQL_PORT
    )


def create_response_record(story_session_id, new_prompt):
    """Insert a new row with placeholder values and return the response ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO story_beats (story_session_id, prompt, response) VALUES (%s, %s, %s)",
        (story_session_id, new_prompt, None)
    )
    response_id = cursor.lastrowid
    conn.commit()
    cursor.close()
    conn.close()
    return response_id


def save_response(response_id, response):
    """Update the database row when processing is done."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE story_beats SET response=%s WHERE id=%s",
        (response, response_id)
    )
    conn.commit()
    cursor.close()
    conn.close()
