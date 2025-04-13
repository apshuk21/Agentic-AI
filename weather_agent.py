from dotenv import load_dotenv
from openai import OpenAI
import requests
import os
import json

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

client = OpenAI(api_key=openai_api_key)

def get_weather_info(location: str):
    url = f"https://wttr.in/{location}?format=%C+%t"
    response = requests.get(url)
    if response.status_code == 200:
        return f"The weather in {location} is: {response.text.strip()}"
    else:
        return f"Could not retrieve weather information for {location}. Please try again later."

def runCommand(command: str):
    print(f"Running command: {command}")
    try:
        result = os.system(command)
        if result == 0:
            return "Command executed successfully."
        else:
            return "Command execution failed."
    except Exception as e:
        return str(e)
    

available_tools = {
    "get_weather_info": {
        "function": get_weather_info,
        "description": "Takes a city name as an input and returns the weather information for that city.",
    },
    "runCommand": {
        "function": runCommand,
        "description": "Takes a command as an input and executes it.",
    }
}

system_prompt = """
    You're a helpful AI assistant that can help resolve user queries.
    You work through a systematic process where you analyse the query, identify or observe the user's intent, plan the action using available tools and execute it.
    You follow a step-by-step approach to answer the users's query using available set of tools.
    Based on your analysis, select the relevant tool. Based on tool selection you perform an action and wait for the response.
    Based on the response you get from the tool, you will provide a final answer to the user.

    Rules:
    1. Follow the output JSON format strictly.
    2. Always perform one step at a time and wait for the next input.
    3. Carefully analyse the user query and identify the intent.

    Available tools:
    - get_weather_info(location: str): Returns the weather information for the given location.
    - runCommand(command: str): Executes the given command and returns the result.

    Output format:
    {{  step: "This should be the output step.",
        "content": "The final output to the user.",
        "function": "The function name to be called",
        "args": "The arguments to be passed to the function",
    }}

    Example:
    User: What is the weather like today in Gurugram?
    Output: {{"step": "analysis", "content": "The user is interested in the weather information for Gurugram.}}
    Output: {{"step": "plan", "content": "From the available tools I should call get_weather_info function to get the weather information."}}
    Output: {{"step": "action", "function": "get_weather_info", "args": "Gurugram"}}
    Output: {{"step": "observe", "output": "12 degrees celsius, sunny"}}}
    Output: {{"step": "output", "content": "The weather in Gurugram today is 12 degrees celsius and sunny."}}
    
"""

messages = [
    {
        "role": "system",
        "content": system_prompt,
    },
]

user_query = input("User: ")
messages.append({"role": "user", "content": user_query})

# response = client.chat.completions.create(
#         model="gpt-4o",
#         temperature=0.5,
#         response_format={"type": "json_object"},
#         messages=[*messages,
#                   {"role": "assistant", "content": json.dumps({"step": "analysis", "content": "The user is interested in the weather information for New Delhi."})},
#                   {"role": "assistant", "content": json.dumps({"step": "plan", "content": "From the available tools I should call get_weather_info function to get the weather information."})},
#                   {"role": "assistant", "content": json.dumps({"step": "action", "function": "get_weather_info", "args": "Chicago"})},
#                 ],
#     )

# print(f"🤖: {response.choices[0].message.content}")

while True:
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.5,
        response_format={"type": "json_object"},
        messages=messages,
    )

    parsed_response =  json.loads(response.choices[0].message.content)

    if (parsed_response.get("step") == "analysis" or parsed_response.get("step") == "plan" or parsed_response.get("step") == "observe"):
        content = parsed_response.get("content") or parsed_response.get("output")
        print(f"🧠: {parsed_response.get('step')}: {content}")
        messages.append({ "role": "assistant", "content": json.dumps(parsed_response)}) # assistant prompt
        continue
    elif parsed_response.get("step") == "action":
        tool = available_tools.get(parsed_response.get("function"), False)
        messages.append({"role": "assistant", "content": json.dumps(parsed_response)})

        if tool:
            function = tool["function"]
            args = parsed_response.get("args")
            if args:
                print(f"🛠️: Tool used: {function.__name__}")
                response = function(args)
                messages.append({"role": "assistant", "content": json.dumps({"step": "observe", "output": response})})
                continue
            else:
                print("Error: No arguments provided for the function.")
                messages.append({"role": "assistant", "content": json.dumps({"step": "observe", "output": "Error: No arguments provided for the function."})})
                continue
    elif parsed_response.get("step") == "output":
        print(f"🤖: {parsed_response["content"]}")
        break


