from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ExecuteCommandRequest(_message.Message):
    __slots__ = ("cmd", "args", "pty")
    CMD_FIELD_NUMBER: _ClassVar[int]
    ARGS_FIELD_NUMBER: _ClassVar[int]
    PTY_FIELD_NUMBER: _ClassVar[int]
    cmd: str
    args: _containers.RepeatedScalarFieldContainer[str]
    pty: PtyConfig
    def __init__(self, cmd: _Optional[str] = ..., args: _Optional[_Iterable[str]] = ..., pty: _Optional[_Union[PtyConfig, _Mapping]] = ...) -> None: ...

class ExecuteCommandResponse(_message.Message):
    __slots__ = ("process_event", "sequence_number")
    PROCESS_EVENT_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    process_event: ProcessEvent
    sequence_number: int
    def __init__(self, process_event: _Optional[_Union[ProcessEvent, _Mapping]] = ..., sequence_number: _Optional[int] = ...) -> None: ...

class ProcessEvent(_message.Message):
    __slots__ = ("start", "data", "end")
    START_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    start: StartEvent
    data: DataEvent
    end: EndEvent
    def __init__(self, start: _Optional[_Union[StartEvent, _Mapping]] = ..., data: _Optional[_Union[DataEvent, _Mapping]] = ..., end: _Optional[_Union[EndEvent, _Mapping]] = ...) -> None: ...

class StartEvent(_message.Message):
    __slots__ = ("pid", "command_id")
    PID_FIELD_NUMBER: _ClassVar[int]
    COMMAND_ID_FIELD_NUMBER: _ClassVar[int]
    pid: int
    command_id: str
    def __init__(self, pid: _Optional[int] = ..., command_id: _Optional[str] = ...) -> None: ...

class DataEvent(_message.Message):
    __slots__ = ("stdout", "stderr", "pty")
    STDOUT_FIELD_NUMBER: _ClassVar[int]
    STDERR_FIELD_NUMBER: _ClassVar[int]
    PTY_FIELD_NUMBER: _ClassVar[int]
    stdout: bytes
    stderr: bytes
    pty: bytes
    def __init__(self, stdout: _Optional[bytes] = ..., stderr: _Optional[bytes] = ..., pty: _Optional[bytes] = ...) -> None: ...

class EndEvent(_message.Message):
    __slots__ = ("exit_code",)
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    exit_code: int
    def __init__(self, exit_code: _Optional[int] = ...) -> None: ...

class PtyConfig(_message.Message):
    __slots__ = ("cols", "rows")
    COLS_FIELD_NUMBER: _ClassVar[int]
    ROWS_FIELD_NUMBER: _ClassVar[int]
    cols: int
    rows: int
    def __init__(self, cols: _Optional[int] = ..., rows: _Optional[int] = ...) -> None: ...

class StreamInputRequest(_message.Message):
    __slots__ = ("start", "data", "close")
    START_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    CLOSE_FIELD_NUMBER: _ClassVar[int]
    start: StreamInputStart
    data: StreamInputData
    close: StreamInputClose
    def __init__(self, start: _Optional[_Union[StreamInputStart, _Mapping]] = ..., data: _Optional[_Union[StreamInputData, _Mapping]] = ..., close: _Optional[_Union[StreamInputClose, _Mapping]] = ...) -> None: ...

class StreamInputStart(_message.Message):
    __slots__ = ("command_id",)
    COMMAND_ID_FIELD_NUMBER: _ClassVar[int]
    command_id: str
    def __init__(self, command_id: _Optional[str] = ...) -> None: ...

class StreamInputData(_message.Message):
    __slots__ = ("data",)
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    def __init__(self, data: _Optional[bytes] = ...) -> None: ...

class StreamInputClose(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StreamInputResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class TerminateCommandRequest(_message.Message):
    __slots__ = ("command_id",)
    COMMAND_ID_FIELD_NUMBER: _ClassVar[int]
    command_id: str
    def __init__(self, command_id: _Optional[str] = ...) -> None: ...

class TerminateCommandResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
