from pydantic import BaseModel, Field
from datetime import datetime
from .lakebase import LakebaseClient
from typing import Tuple
from psycopg.types.json import Json
from abc import ABC, abstractmethod
from typing import Any, Union, Tuple
from uuid import uuid4

class GenericCheckpoint(BaseModel):
    ####################
    # CHECKPOINT COLUMNS
    ####################
    id : str = Field(default_factory = lambda: str(uuid4())) # `id` column must exist for locating the checkpoint, even in subclasses.
    state : dict = {}
    creation_timestamp : datetime = Field(default_factory=datetime.now)
    update_timestamp : datetime = Field(default_factory=datetime.now)
    ####################
    # UPDATE ATTRIBUTES
    ####################
    def update(self, **kwargs):
        for k,v in kwargs.items():
            # Check attribute exists
            assert hasattr(self, k), f"Attribute {k} does not exist in {self.__class__.__name__}"
            setattr(self, k, v)
        self.update_timestamp = datetime.now()
    #################################
    # SQL GENERATION IMPLEMENTATIONS
    #################################
    # If subclassing the `GenericCheckpoint` class, implement new methods to handle any changes in attributes.
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
        sql_all, params = self.generate_retrieval_all_checkpoints_sql(table_name = table_name)
        sql = f"""
            {sql_all} LIMIT 1
        """
        return sql, params
    
    def generate_retrieval_all_checkpoints_sql(self, table_name : str) -> Tuple[str, tuple]:
        sql = f"""
            SELECT id, state, creation_timestamp, update_timestamp
            FROM {table_name}
            WHERE id = %s
            ORDER BY update_timestamp DESC
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
        response = self.client.execute(sql=sql, params = params)
        if len(response) == 0:
            return None
        return self.checkpoint_class(**response[0])
    
    def get_all_checkpoints(self, id : str) -> list[GenericCheckpoint]:
        _checkpoint = self.checkpoint_class(id = id)
        sql, params = _checkpoint.generate_retrieval_all_checkpoints_sql(table_name = self.table_name)
        response = self.client.execute(sql=sql, params = params)
        return [self.checkpoint_class(**_resp) for _resp in response]

    def checkpoint_exists(self, id : str) -> bool:
        sql = f"""
            SELECT COUNT(*) AS count
            FROM {self.table_name}
            WHERE id = %s
        """
        response = self.client.execute(sql=sql, params=(id,))
        count = response[0]["count"]
        return True if count > 0 else False
    
    def update_most_recent_checkpoint(self, id : str, **checkpoint_kwargs) -> None:
        # Get the most recent checkpoint
        _checkpoint = self.get_most_recent_checkpoint(id = id)
        _checkpoint.update(**checkpoint_kwargs)
        sql, params = _checkpoint.generate_update_sql(table_name = self.table_name)
        self.client.execute(sql = sql, params = params)
        return

    def insert_checkpoint(self, id : str, **checkpoint_kwargs) -> None:
        _checkpoint = self.checkpoint_class(
            id = id, **checkpoint_kwargs
        )
        sql, params = _checkpoint.generate_insert_sql(table_name = self.table_name)
        response = self.client.execute(sql=sql, params=params)
        return

    def save_checkpoint(self, id : str, overwrite : bool = False, **checkpoint_kwargs) -> None:
        if overwrite:
            checkpoint_exists = self.checkpoint_exists(id = id)
            if checkpoint_exists:
                self.update_checkpoint(id = id, **checkpoint_kwargs)
            else:
                self.insert_checkpoint(id = id, **checkpoint_kwargs)
        else:
            # Don't overwrite the most recent checkpoint, so just insert a new one
            self.insert_checkpoint(id = id, **checkpoint_kwargs)