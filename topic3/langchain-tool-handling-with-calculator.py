
"""
Tool Calling with LangChain
Shows how LangChain abstracts tool calling.
"""

import math
import json
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# ============================================
# PART 1: Define Your Tools
# ============================================

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location"""
    # Simulated weather data
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 55°F",
        "London": "Rainy, 48°F",
        "Tokyo": "Clear, 65°F"
    }
    return weather_data.get(location, f"Weather data not available for {location}")


@tool
def calculate(expression: str) -> str:
    """
    Calculate mathematical expressions including geometric and trigonometric functions.

    Supports:
    - Basic arithmetic: +, -, *, /, ** (power), % (modulo)
    - Trigonometric: sin(x), cos(x), tan(x), asin(x), acos(x), atan(x), atan2(y,x)
    - Hyperbolic: sinh(x), cosh(x), tanh(x)
    - Exponential/Log: exp(x), log(x), log10(x), log2(x), sqrt(x)
    - Other: abs(x), ceil(x), floor(x), round(x), degrees(x), radians(x)
    - Constants: pi, e

    Examples:
    - sin(pi/2) - sine of 90 degrees (in radians)
    - sqrt(16) - square root of 16
    - 2 ** 3 - 2 to the power of 3
    - degrees(pi) - convert pi radians to degrees

    Note: Trigonometric functions expect angles in radians.

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        JSON string with result or error message
    """
    try:
        # Define safe namespace with math functions
        safe_dict = {
            # Trigonometric functions
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'atan2': math.atan2,

            # Hyperbolic functions
            'sinh': math.sinh,
            'cosh': math.cosh,
            'tanh': math.tanh,

            # Exponential and logarithmic
            'exp': math.exp,
            'log': math.log,
            'log10': math.log10,
            'log2': math.log2,
            'sqrt': math.sqrt,

            # Power and roots
            'pow': math.pow,

            # Rounding
            'ceil': math.ceil,
            'floor': math.floor,
            'round': round,

            # Absolute value
            'abs': abs,

            # Angle conversion
            'degrees': math.degrees,
            'radians': math.radians,

            # Constants
            'pi': math.pi,
            'e': math.e,
        }

        # Parse and evaluate the expression safely
        result = eval(expression, {"__builtins__": {}}, safe_dict)

        # Format the result
        output = {
            "expression": expression,
            "result": result,
            "success": True
        }

        return json.dumps(output)

    except ZeroDivisionError:
        error_output = {
            "expression": expression,
            "error": "Division by zero",
            "success": False
        }
        return json.dumps(error_output)

    except Exception as e:
        error_output = {
            "expression": expression,
            "error": str(e),
            "success": False
        }
        return json.dumps(error_output)


# ============================================
# PART 2: Create LLM with Tools
# ============================================

# Create LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Bind BOTH tools to LLM
llm_with_tools = llm.bind_tools([get_weather, calculate])


# ============================================
# PART 3: The Agent Loop
# ============================================

def run_agent(user_query: str):
    """
    Simple agent that can use tools.
    Shows the manual loop that LangGraph automates.
    """

    # Start conversation with user query
    messages = [
        SystemMessage(content="You are a helpful assistant. Use the provided tools when needed. For trigonometric calculations, remember angles should be in radians unless converting."),
        HumanMessage(content=user_query)
    ]

    print(f"User: {user_query}\n")

    # Agent loop - can iterate up to 5 times
    for iteration in range(5):
        print(f"--- Iteration {iteration + 1} ---")

        # Call the LLM
        response = llm_with_tools.invoke(messages)

        # Check if the LLM wants to call a tool
        if response.tool_calls:
            print(f"LLM wants to call {len(response.tool_calls)} tool(s)")

            # Add the assistant's response to messages
            messages.append(response)

            # Execute each tool call
            for tool_call in response.tool_calls:
                function_name = tool_call["name"]
                function_args = tool_call["args"]

                print(f"  Tool: {function_name}")
                print(f"  Args: {function_args}")

                # Execute the tool
                if function_name == "get_weather":
                    result = get_weather.invoke(function_args)
                elif function_name == "calculate":
                    result = calculate.invoke(function_args)
                else:
                    result = f"Error: Unknown function {function_name}"

                print(f"  Result: {result}")

                # Add the tool result back to the conversation
                messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]
                ))

            print()
            # Loop continues - LLM will see the tool results

        else:
            # No tool calls - LLM provided a final answer
            print(f"Assistant: {response.content}\n")
            return response.content

    return "Max iterations reached"


# ============================================
# PART 4: Test It
# ============================================

if __name__ == "__main__":
    # Test query that requires weather tool
    print("="*60)
    print("TEST 1: Query requiring weather tool")
    print("="*60)
    run_agent("What's the weather like in San Francisco?")

    print("\n" + "="*60)
    print("TEST 2: Query not requiring tool")
    print("="*60)
    run_agent("Say hello!")

    print("\n" + "="*60)
    print("TEST 3: Multiple weather tool calls")
    print("="*60)
    run_agent("What's the weather in New York and London?")

    print("\n" + "="*60)
    print("TEST 4: Basic calculator")
    print("="*60)
    run_agent("What is 25 * 4 + 17?")

    print("\n" + "="*60)
    print("TEST 5: Trigonometric calculation")
    print("="*60)
    run_agent("What is the sine of 30 degrees?")

    print("\n" + "="*60)
    print("TEST 6: Geometric calculation")
    print("="*60)
    run_agent("Calculate the hypotenuse of a right triangle with sides 3 and 4")

    print("\n" + "="*60)
    print("TEST 7: Multiple calculations")
    print("="*60)
    run_agent("What is sqrt(144) and what is cos(0)?")

    print("\n" + "="*60)
    print("TEST 8: Complex expression")
    print("="*60)
    run_agent("Calculate 2*pi*5 (circumference of circle with radius 5)")

    print("\n" + "="*60)
    print("TEST 9: Mixed tools - weather and calculation")
    print("="*60)
    run_agent("What's the weather in Tokyo and what is 72 converted to Celsius? Use the formula: (F - 32) * 5/9")

    print("\n" + "="*60)
    print("TEST 10: Logarithmic calculation")
    print("="*60)
    run_agent("What is log base 10 of 1000?")
