import pytest
from hello_world import print_hello_world  # Assuming the function is named correctly in your code

def test_print_output():
    captured_out = StringIO()
    sys.stdout = captured_out
    print_hello_world()  # Replace with actual function call if different
    sys.stdout = sys.__stdout__
    assert captured_out.getvalue().strip() == "Hello, World!"

def test_function_exists():
    from hello_world import print_hello_world  # Assuming the function is named correctly in your code
    assert callable(print_hello_world)