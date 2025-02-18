from typing import Dict, Type, TypeVar

T = TypeVar('T')


class Singleton:
    _instances: Dict[Type, object] = {}

    @classmethod
    def get_instance(cls, class_type: Type[T], *args, **kwargs) -> T:
        if class_type not in cls._instances:
            cls._instances[class_type] = class_type(*args, **kwargs)
        return cls._instances[class_type]

    @classmethod
    def clear_instance(cls, class_type: Type[T]) -> None:
        if class_type in cls._instances:
            del cls._instances[class_type]
