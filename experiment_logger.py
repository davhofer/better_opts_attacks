import pickle
import json
import typing
import datetime
import sys
import shelve
import inspect
import time
import pathlib
import functools

class ExperimentLogger:
    """
    A stateful logger for arbitrary Python objects that maintains metadata
    across function calls.
    """
    def __init__(self, log_directory: str = 'experiment_logs'):
        """
        Initialize the logger with a specific log directory.
        """
        self.log_dir = pathlib.Path(log_directory)
        self.log_dir.mkdir(exist_ok=True)
        
        self.objects_path = str(self.log_dir / 'objects')
        self.metadata_path = self.log_dir / 'metadata.jsonl'
        
        # Stack to maintain metadata across function calls
        self._metadata_stack: typing.List[typing.Dict[str, typing.Any]] = []
    
    def push_metadata(self, metadata: typing.Dict[str, typing.Any]) -> None:
        """Push metadata onto the stack."""
        self._metadata_stack.append(metadata)
    
    def pop_metadata(self) -> typing.Dict[str, typing.Any]:
        """Pop metadata from the stack."""
        return self._metadata_stack.pop() if self._metadata_stack else {}
    
    def peek_metadata(self) -> typing.Dict[str, typing.Any]:
        """Get current metadata without removing it."""
        return self._metadata_stack[-1] if self._metadata_stack else {}

    def track_function(self, func=None, **metadata):
        """
        Decorator to track function calls and maintain metadata stack.
        Can be used as @logger.track_function or @logger.track_function(key=value)
        """
        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                # Prepare function metadata
                func_metadata = {
                    'function_name': fn.__name__,
                    'function_file': inspect.getfile(fn),
                    'function_line': inspect.getsourcelines(fn)[1],
                    **metadata
                }
                
                # Merge with current metadata if any
                if self._metadata_stack:
                    func_metadata['parent_metadata'] = self.peek_metadata()
                
                # Push new metadata to stack
                self.push_metadata(func_metadata)
                
                try:
                    return fn(*args, **kwargs)
                finally:
                    # Always pop metadata when function exits
                    self.pop_metadata()
            
            return wrapper
        
        # Handle both @track_function and @track_function(metadata)
        if func is None:
            return decorator
        return decorator(func)

    def log(self,
            obj: typing.Any, 
            custom_metadata: typing.Optional[typing.Dict[str, typing.Any]] = None, 
            variable_name: typing.Optional[str] = None
        ) -> None:
        """
        Log an arbitrary Python object with metadata.
        """
        # Get caller information
        caller_frame = inspect.currentframe().f_back
        filename = caller_frame.f_code.co_filename
        lineno = caller_frame.f_lineno
        
        # Generate unique ID using high-precision timestamp
        obj_id = f"{time.time():.7f}"
        
        # Start with basic metadata
        metadata = {
            'id': obj_id,
            'filename': filename,
            'lineno': lineno,
            'timestamp': obj_id,
        }
        
        # Add variable name if provided or try to extract it
        if variable_name:
            metadata['variable_name'] = variable_name
        else:
            frame_locals = caller_frame.f_locals
            obj_id_mem = id(obj)
            possible_names = [name for name, value in reversed(list(frame_locals.items())) if id(value) == obj_id_mem]
            if possible_names:
                metadata['variable_name'] = possible_names[0]
        
        # Add current function context if available
        if self._metadata_stack:
            metadata['call_stack'] = self._metadata_stack.copy()
        
        # Merge with custom metadata if provided
        if custom_metadata:
            metadata.update(custom_metadata)
        
        # Write metadata to JSONL file
        with open(self.metadata_path, 'a') as f:
            json.dump(metadata, f)
            f.write('\n')
        
        # Store object using shelve
        with shelve.open(self.objects_path) as shelf:
            shelf[obj_id] = obj
    
    def query(self, metadata_query: typing.Dict[str, typing.Any]) -> typing.Iterator[typing.Any]:
        """
        Query logged objects based on metadata.
        """
        def get_object(obj_id: str) -> typing.Any:
            with shelve.open(self.objects_path) as shelf:
                return shelf.get(obj_id)

        def matches_query(metadata: typing.Dict[str, typing.Any], query: typing.Dict[str, typing.Any]) -> bool:
            """
            Recursively check if metadata matches query, including the call stack.
            """
            # Check direct metadata matches
            direct_match = all(
                metadata.get(k) == v 
                for k, v in query.items()
            )
            if direct_match:
                return True
                
            # Check call stack if it exists
            if 'call_stack' in metadata:
                # Check each frame in the call stack
                for frame in metadata['call_stack']:
                    # Check direct frame metadata
                    if all(frame.get(k) == v for k, v in query.items()):
                        return True
                    
                    # Check parent metadata if it exists
                    if 'parent_metadata' in frame:
                        if matches_query(frame['parent_metadata'], query):
                            return True
            return False

        with open(self.metadata_path, 'r') as f:
            for line in f:
                metadata = json.loads(line.strip())
                
                # Use recursive matching function
                if matches_query(metadata, metadata_query):
                    obj_id = metadata['id']
                    obj = get_object(obj_id)
                    if obj is not None:
                        yield obj