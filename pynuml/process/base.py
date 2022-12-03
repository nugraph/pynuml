from abc import ABC
from typing import Any

class ProcessorBase(ABC):
	'''Base class for event processing'''

    def __call__(self, evt: Any) -> Any:
    	raise NotImplementedError