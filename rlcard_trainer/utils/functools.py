from typing import Union, get_origin, get_args
from functools import wraps
import inspect
import types

def type_check(func: types.FunctionType, *args, **kwargs):
    """
    A decorator that automatically verifies function arguments 
    using type hints from function annotations, including default values.
    
    Handles None values in Union types more flexibly.
    """
    @wraps(func)  # Preserve original function's metadata
    def wrapper(*args, **kwargs):
        # Get the function's signature
        sig = inspect.signature(func)
        
        # Apply defaults for missing arguments
        bound_arguments = sig.bind_partial(*args, **kwargs)
        bound_arguments.apply_defaults()
        
        # Get parameter type hints and default values
        param_type_hints = {}
        param_defaults = {}
        for param_name, param in sig.parameters.items():
            if param.annotation is not inspect.Parameter.empty:
                param_type_hints[param_name] = param.annotation
            
            # Store default values
            if param.default is not inspect.Parameter.empty:
                param_defaults[param_name] = param.default
        
        # Combine positional and keyword arguments
        all_args = bound_arguments.arguments
        
        # Check each argument (including default values) against its type specification
        for param_name, param_value in all_args.items():
            # Skip *args and **kwargs
            if param_name in ('args', 'kwargs'):
                continue
            
            # If no value provided, use default if available
            if param_value is None and param_name in param_defaults:
                param_value = param_defaults[param_name]
            
            # Check type if type hint exists
            if param_name in param_type_hints:
                _check_type(param_value, param_type_hints[param_name], param_name)
        
        return func(*args, **kwargs)
    
    return wrapper

def _check_type(value, type_spec, param_name=None):
    """
    Recursively validate a value against a type specification.
    
    Allows None for Union types that include None.
    """
    # Check if None is allowed in the type specification
    def is_none_allowed(type_spec):
        # Check for Union types
        origin = get_origin(type_spec)
        if origin is Union:
            return type(None) in get_args(type_spec)
        
        # Check for UnionType in Python 3.10+
        if isinstance(type_spec, types.UnionType):
            return type(None) in type_spec.__args__
        
        return False

    # If value is None, check if it's allowed
    if value is None:
        if not is_none_allowed(type_spec):
            raise TypeError(f"Value for {param_name or 'parameter'} cannot be None")
        return
    
    # Special handling for | (Union) types in Python 3.10+
    if isinstance(type_spec, types.UnionType):
        union_types = [t for t in type_spec.__args__ if t is not type(None)]
        for union_type in union_types:
            try:
                _check_type(value, union_type, param_name)
                return  # If any type matches, validation passes
            except TypeError:
                continue
        
        # If no type matches, raise an error
        raise TypeError(f"Value {value} does not match any type in Union {type_spec} for {param_name or 'parameter'}")
    
    # Get the origin type (e.g., list for list[int])
    origin = get_origin(type_spec)
    
    # If no origin, it's a basic type check
    if origin is None:
        if not isinstance(value, type_spec):
            raise TypeError(f"Expected {type_spec} for {param_name or 'parameter'}, got {type(value)}")
        return
    
    # Handle Union types from typing module
    if origin is Union:
        # Remove None from type arguments
        type_args = [t for t in get_args(type_spec) if t is not type(None)]
        
        # Check if the value matches any of the types in the Union
        for union_type in type_args:
            try:
                _check_type(value, union_type, param_name)
                return  # If any type matches, validation passes
            except TypeError:
                continue
        
        # If no type matches, raise an error
        raise TypeError(f"Value {value} does not match any type in Union {type_spec} for {param_name or 'parameter'}")
    
    # Handle List types
    if origin is list:
        if not isinstance(value, list):
            raise TypeError(f"Expected list for {param_name or 'parameter'}, got {type(value)}")
        
        # Check element types
        list_type_args = get_args(type_spec)
        if list_type_args:
            element_type = list_type_args[0]
            for item in value:
                _check_type(item, element_type, f"element in {param_name}")
        return
    
    # Handle Tuple types
    if origin is tuple:
        if not isinstance(value, tuple):
            raise TypeError(f"Expected tuple for {param_name or 'parameter'}, got {type(value)}")
        
        tuple_type_args = get_args(type_spec)
        
        # Check length if specific tuple length is specified
        if len(tuple_type_args) > 0 and tuple_type_args[-1] is not Ellipsis:
            if len(value) != len(tuple_type_args):
                raise TypeError(f"Expected tuple of length {len(tuple_type_args)} for {param_name or 'parameter'}, got length {len(value)}")
        
        # Validate each element
        for i, (item, item_type) in enumerate(zip(value, tuple_type_args)):
            if item_type is Ellipsis:
                break
            _check_type(item, item_type, f"element {i} in {param_name}")
        return
    
    # Add more type checking for other complex types as needed
    raise TypeError(f"Unsupported type specification: {type_spec}")


def reset_default_args(func):
    def wrapper(*args, **kwargs):
        # Get the function's default arguments
        defaults = func.__defaults__ or ()
        default_names = func.__code__.co_varnames[len(args):len(args)+len(defaults)]
        
        # Create a copy of default arguments for this call
        modified_kwargs = kwargs.copy()
        for name, default in zip(default_names, defaults):
            if name not in modified_kwargs:
                # Create a new instance if the argument is mutable
                if isinstance(default, (list, dict, set)):
                    modified_kwargs[name] = type(default)()
                else:
                    modified_kwargs[name] = default
        
        return func(*args, **modified_kwargs)
    return wrapper


if __name__ == "__main__":
    # ====================================
    # | Demonstrate type_check decorator |
    # ====================================
    @type_check
    def example_func(numbers: list[int | float], name: str, details: tuple[int, str] = (42, 'info')) -> None:
        """
        An example function to demonstrate type verification with default values.
        
        Args:
            numbers: A list of numbers (int or float)
            name: A string name
            details: A tuple containing an int and a string
        """
        print(f"Numbers: {numbers}")
        print(f"Name: {name}")
        print(f"Details: {details}")
    
    # Demonstrate preserved metadata
    print(f"Function name: {example_func.__name__}")
    print(f"Function docstring: {example_func.__doc__}")
    
    # Valid calls with and without default value
    example_func([1, 2.5, 3], "Test")  # Uses default details
    example_func([1, 2, 3], "Another test", (100, "data"))  # Provides custom details
    
    # These will raise TypeError
    try:
        example_func([1, "not a number"], "Test")
    except TypeError as e:
        print(f"Caught error: {e}")
    
    try:
        example_func([1, 2], 123)
    except TypeError as e:
        print(f"Caught error: {e}")


    # ============================================
    # | Demonstrate reset_default_args decorator |
    # ============================================
    @reset_default_args
    def example_func_with_defaults(numbers: list[int | float], name: str, details: tuple[int, str] = (42, 'info')) -> None:
        """
        An example function to demonstrate resetting default arguments.
        
        Args:
            numbers: A list of numbers (int or float)
            name: A string name
            details: A tuple containing an int and a string
        """
        print(f"Numbers: {numbers}")
        print(f"Name: {name}")
        print(f"Details: {details}")
    
    # Valid calls with and without default value
    example_func_with_defaults([1, 2.5, 3], "Test")  # Uses default details
    example_func_with_defaults([1, 2, 3], "Another test", (100, "data"))  # Provides custom details
    
    # These will raise TypeError
    try:
        example_func_with_defaults([1, "not a number"], "Test")
    except TypeError as e:
        print(f"Caught error: {e}")
    
    try:
        example_func_with_defaults([1, 2], 123)
    except TypeError as e:
        print(f"Caught error: {e}")
