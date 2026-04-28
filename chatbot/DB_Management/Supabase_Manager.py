from supabase import Client, create_client
from typing import List, Tuple, Dict, Any
import uuid
from datetime import datetime

class DatabaseManager:


    def __init__(self, url: str, key: str):
        self.url = url
        self.key = key
        self.client: Client = None
        self._connect()

    def _connect(self) -> None:

        self.client = create_client(self.url, self.key)
        print("Connected to Supabase")

    def execute_query(self, query: str):
        if not query.lower().strip().startswith("select"):
            return []

        try:
            response = self.client.rpc(
                "execute_select",
                {"query": query}
            ).execute()

            if response.data:
                result = []
                for row in response.data:
                    result.append(tuple(row.values()))
                return result

            return []

        except Exception as e:
            print(f" Supabase Query Error: {e}")
            return [{"error": str(e)}]

    async def insert_message(self, user_id: str, role: str, content: str) -> Dict[str, Any]:
        """
        Insert a new message into the messages table.

        Args:
            user_id: UUID of the user
            role: Either "human" or "ai"
            content: The message content

        Returns:
            The inserted message record
        """
        try:
            message_data = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "role": role,
                "content": content,
                "created_at": datetime.utcnow().isoformat()
            }

            response = self.client.table("messages").insert(message_data).execute()

            if response.data:
                return response.data[0]
            return message_data

        except Exception as e:
            print(f" Insert Message Error: {e}")
            raise

    async def get_last_messages(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the last N messages for a user, ordered by created_at DESC.

        Args:
            user_id: UUID of the user
            limit: Number of messages to retrieve (default 10)

        Returns:
            List of message records, newest first
        """
        try:
            response = (
                self.client.table("messages")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )

            if response.data:
                return response.data
            return []

        except Exception as e:
            print(f" Get Messages Error: {e}")
            return []

    def close(self) -> None:

        print(" Supabase connection closed")