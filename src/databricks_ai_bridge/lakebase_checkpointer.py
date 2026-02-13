from pydantic import BaseModel
from datetime import datetime
from .lakebase import LakebaseClient
from typing import Tuple
from psycopg.types.json import Json
from abc import ABC, abstractmethod
from typing import Any, Union, Tuple
from uuid import uuid4

class GenericCheckpoint(BaseModel):
    # Define the fields for your checkpoint
    id : str = str(uuid4())
    state : dict = {}
    creation_timestamp : datetime = datetime.now()
    update_timestamp : datetime = datetime.now()
    # Implement abstract methods
    def generate_insert_sql(self, table_name : str) -> Tuple[str, tuple]:
        sql = f"""
            INSERT INTO {table_name}
            (id, state, creation_timestamp, update_timestamp)
            VALUES (%s, %s, %s, %s)
        """
        return sql, (self.id, Json(self.state), self.creation_timestamp, self.update_timestamp)
    
    def generate_update_sql(self, table_name : str) -> Tuple[str, tuple]:
        sql = f"""
            UPDATE {table_name}
            SET state = %s, update_timestamp = %s
            WHERE id = %s AND creation_timestamp = %s
        """
        return sql, (Json(self.state), self.update_timestamp, self.id, self.creation_timestamp)
    
    def generate_init_sql(self, table_name : str) -> Tuple[str, tuple]:
        sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            lb_id bigserial PRIMARY KEY,
            id text NOT NULL,
            state jsonb NOT NULL default '{{}}',
            creation_timestamp timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
            update_timestamp timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(id,creation_timestamp)
        )
        """
        return sql, None
    
    def generate_retrieve_checkpoint_sql(self, table_name : str) -> Tuple[str, tuple]:
        sql = f"""
            SELECT id, state, creation_timestamp, update_timestamp
            FROM {table_name}
            WHERE id = %s
            ORDER BY update_timestamp DESC
            LIMIT 1
        """
        return sql, (self.id,)

class LakebaseCheckpointer:
    def __init__(self, 
                 lakebase_client : LakebaseClient, 
                 sessions_table_name : str, 
                 checkpoint_class : GenericCheckpoint = GenericCheckpoint):
        self.checkpoint_class = checkpoint_class
        self.client = lakebase_client
        self.table_name = sessions_table_name
        self.init_schema()
    
    def init_schema(self) -> None:
        _checkpoint = self.checkpoint_class()
        sql, params = _checkpoint.generate_init_sql(table_name = self.table_name)
        response = self.client.execute(sql=sql, params = params)
    
    def get_most_recent_checkpoint(self, id : str) -> GenericCheckpoint | None:
        _checkpoint = self.checkpoint_class(id = id)
        # Get the most recently updated checkpoint for this id
        sql, params = _checkpoint.generate_retrieve_checkpoint_sql(table_name = self.table_name)
        response = self.client.execute(sql=sql, params = params)[0]
        if response:
            response = self.checkpoint_class(**response)
        return response

    def checkpoint_exists(self, id : str) -> bool:
        sql = f"""
            SELECT COUNT(*) AS count
            FROM {self.table_name}
            WHERE id = %s
        """
        response = self.client.execute(sql=sql, params=(id,))
        count = response[0]["count"]
        return True if count > 0 else False
    
    def update_checkpoint(self, id : str, state : dict) -> None:
        # Get the most recent checkpoint
        _checkpoint = self.get_most_recent_checkpoint(id = id)
        # Set the new state value locally
        _checkpoint.state = state
        _checkpoint.update_timestamp = datetime.now()
        sql, params = _checkpoint.generate_update_sql(table_name = self.table_name)
        self.client.execute(sql = sql, params = params)
        return

    def insert_checkpoint(self, id : str, state : dict) -> None:
        _checkpoint = self.checkpoint_class(
            id = id, state = state, creation_timestamp = datetime.now(), update_timestamp = datetime.now()
        )
        sql, params = _checkpoint.generate_insert_sql(table_name = self.table_name)
        response = self.client.execute(sql=sql, params=params)
        return

    def save_checkpoint(self, id : str, state : dict, overwrite : bool = False) -> None:
        if overwrite:
            checkpoint_exists = self.checkpoint_exists(id = id)
            if checkpoint_exists:
                self.update_checkpoint(id = id, state = state)
            else:
                self.insert_checkpoint(id = id, state = state)
        else:
            # Don't overwrite existing checkpoints, so just insert a new one
            self.insert_checkpoint(id = id, state = state)