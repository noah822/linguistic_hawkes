from typing import Union
import os
from pathlib import Path

class LazyFileReader:
    def __init__(self,
                 path: str,
                 buffer_size: int=1<<15,
                 start_pos: int=0,
                 readin_byte: int=None):
        
        self.path = path
        self.buffer_size = buffer_size

        self.start_pos = start_pos

        if readin_byte is not None:
            self.readin_byte = readin_byte
        else: # default to file size in byte number
            dummy_binary_file = open(path, 'rb')
            dummy_binary_file.seek(0, os.SEEK_END)
            end_pos = dummy_binary_file.tell()
            dummy_binary_file.close()
            self.readin_byte = end_pos - start_pos

    def __iter__(self):
        # lazily load file chunk by chunk, according to buffer size
        
        # whenever generates one-time iterator for file object
        # seek to specfied position

        handler = open(self.path, 'rb')
        
        # seek to starting position 
        handler.seek(self.start_pos)

        full_chunk_number = int(self.readin_byte / self.buffer_size)
        remain_bytes = self.readin_byte - full_chunk_number * self.buffer_size

        # do round-robin kind of operation
        for _ in range(full_chunk_number):
            yield handler.read(self.buffer_size).decode(encoding='utf-8')
        
        if remain_bytes > 0:
            yield handler.read(remain_bytes).decode(encoding='utf-8')
            


