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


@tool
def count_letter(text: str, letter: str) -> str:
    """
    Count the number of occurrences of a specific letter in a piece of text.

    The counting is case-insensitive, so 'A' and 'a' are treated as the same letter.
    Only counts letters, not other characters.

    Args:
        text: The text to search in
        letter: The letter to count (should be a single character)

    Returns:
        JSON string with the count and details

    Examples:
        - count_letter("Mississippi", "s") -> 4
        - count_letter("Hello World", "l") -> 3
        - count_letter("Programming", "g") -> 2
    """
    try:
        # Validate that letter is a single character
        if len(letter) != 1:
            return json.dumps({
                "error": "Letter parameter must be a single character",
                "success": False
            })

        # Convert both to lowercase for case-insensitive counting
        text_lower = text.lower()
        letter_lower = letter.lower()

        # Count occurrences
        count = text_lower.count(letter_lower)

        # Find positions (0-indexed) for additional context
        positions = [i for i, char in enumerate(text_lower) if char == letter_lower]

        output = {
            "text": text,
            "letter": letter,
            "count": count,
            "positions": positions,
            "success": True
        }

        return json.dumps(output)

    except Exception as e:
        error_output = {
            "text": text,
            "letter": letter,
            "error": str(e),
            "success": False
        }
        return json.dumps(error_output)


# ============================================
# PART 2: Create Tool Map and LLM with Tools
# ============================================

# Define all tools
tools = [get_weather, calculate, count_letter]

# Create tool map for dynamic dispatch
tool_map = {tool.name: tool for tool in tools}

# Create LLM and bind all tools
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)


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

                # IMPROVED: Use tool_map for dynamic dispatch
                if function_name in tool_map:
                    result = tool_map[function_name].invoke(function_args)
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
    print("\n" + "="*60)
    print("Mixed tools - weather and calculation")
    print("="*60)
    run_agent("What's the weather in Tokyo and what is it converted to Celsius? Use the formula: (F - 32) * 5/9")

    print("\n" + "="*60)
    print("Letter counting - Mississippi")
    print("="*60)
    run_agent("How many s are in Mississippi?")

    print("\n" + "="*60)
    print("Letter counting - riverboats")
    print("="*60)
    run_agent("How many s are in Mississippi riverboats?")

    print("\n" + "="*60)
    print("Letter counting - case insensitive")
    print("="*60)
    run_agent("How many letter 'e' appears in 'Tennessee'?")

    print("\n" + "="*60)
    print("Letter counting with calculation")
    print("="*60)
    run_agent("Count the letter 'i' in 'Mississippi' and then multiply that count by 3")

    print("\n" + "="*60)
    print("Letter counting with calculation (comparison)")
    print("="*60)
    run_agent("Are there more i's than s's in Mississippi riverboats?")

    print("\n" + "="*60)
    print("Letter counting with calculation")
    print("="*60)
    run_agent("What is the sin of the difference between the number of i's and the number of s's in Mississippi riverboats?")
