from pydantic import BaseModel
from datetime import datetime
from .lakebase import LakebaseClient
from typing import Tuple
from psycopg.types.json import Json

class Checkpoint(BaseModel):
    id : str
    state : dict
    creation_timestamp : datetime
    update_timestamp : datetime
    def get_insert_params(self) -> Tuple:
        return (
            self.id,
            Json(self.state),
            self.creation_timestamp,
            self.update_timestamp
        )
    def get_update_params(self) -> Tuple:
        return (
            Json(self.state),
            self.update_timestamp
        )

class LakebaseCheckpointer:
    def __init__(self, lakebase_client : LakebaseClient, sessions_table_name : str):
        self.client = lakebase_client
        self.table_name = sessions_table_name
    
    def init_schema(self) -> None:
        sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            lb_id bigserial PRIMARY_KEY,
            id text NOT NULL,
            state jsonb NOT NULL default '{{}}',
            creation_timestamp timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
            update_timestamp timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(id,creation_timestamp)
        )
        """
        response = self.client.execute(sql=sql)
    
    def get_most_recent_checkpoint(self, id : str) -> Checkpoint | None:
        # Get the most recently updated checkpoint for this id
        sql = f"""
            SELECT id, state, creation_timestamp, update_timestamp
            FROM {self.table_name}
            WHERE id = %s
            ORDER BY update_timestamp DESC
            LIMIT 1
        """
        response = self.client.execute(sql=sql, params = (id,))
        if response:
            response = Checkpoint(**response)
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
        sql = f"""
            UPDATE {self.table_name}
            SET state = %s, update_timestamp = %s
            WHERE id = %s AND creation_timestamp = %s
        """
        self.client.execute(sql=sql, params=_checkpoint.get_update_params + (_checkpoint.id, _checkpoint.creation_timestamp))
        return


    def insert_checkpoint(self, id : str, state : dict) -> None:
        _checkpoint = Checkpoint(id = id, state = state, creation_timestamp = datetime.now(), update_timestamp = datetime.now())
        sql = f"""
            INSERT INTO {self.table_name}
            (id, state, creation_timestamp, update_timestamp)
            VALUES (%s, %s, %s, %s)
        """
        response = self.client.execute(sql=sql, params=_checkpoint.get_insert_params())
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