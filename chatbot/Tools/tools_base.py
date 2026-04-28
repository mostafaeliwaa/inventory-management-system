from typing import Any, Dict, List, Optional, Tuple

class Tool:
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(self, *args, **kwargs) -> Any:
        
        raise NotImplementedError("Subclasses must implement execute()")
    
