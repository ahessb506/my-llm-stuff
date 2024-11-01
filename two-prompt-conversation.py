from langchain_ollama import OllamaLLM
import csv
import os

# Explanation of the script:
# This script simulates a conversation between two speakers.
# The speakers are defined by system prompts that dictate their behavior and personality traits.
# The conversation alternates between the two speakers until a termination condition is met.
# 
# Customization:
# 1. To customize the behavior of the speakers, modify the system prompts (SPEAKER1_SYSTEM_PROMPT and SPEAKER2_SYSTEM_PROMPT).
# 2. Adjust the number of conversation turns by changing the `num_turns` variable.
# 3. You can also modify the initial conversation lines to set a different starting context.
# 
# Triggering the Conversation:
# The conversation is triggered by the main function, which starts by initializing the Ollama LLM and establishing the first message.
# Each turn alternates between Speaker 1 and Speaker 2, collecting responses until one of them indicates that the conversation has ended.

# Define model name
OLLAMA_MODEL = "llama3.2"  # Ensure this matches the exact model name in your Ollama setup

# Define system prompts as global variables
SPEAKER_2_SYSTEM_PROMPT = """You are Speaker 2, a person..."""
SPEAKER_1_SYSTEM_PROMPT = """You are Speaker 2, a person..."""

def create_prompt(system_prompt, messages, speaker_role):
    """
    Constructs the full prompt by combining the system prompt with the conversation history.

    Args:
        system_prompt (str): The initial system prompt defining the assistant's behavior.
        messages (list of dict): The conversation history with 'role' and 'content'.
        speaker_role (str): The role of the speaker ('user' or 'assistant').

    Returns:
        str: The combined prompt string.
    """
    prompt = system_prompt + "\n\n"
    for message in messages:
        role = message['role']
        content = message['content']
        if role == 'user':
            prompt += f"User: {content}\n"
        elif role == 'assistant':
            prompt += f"Assistant: {content}\n"
    prompt += "Assistant:"  # Indicate that the assistant should respond next
    return prompt

def build_agent_conversation_history(agent_name, conversation_lines):
    """
    Builds the conversation history tailored for a specific agent.

    Args:
        agent_name (str): The name of the agent ('Speaker 2' or 'Speaker 1').
        conversation_lines (list of dict): The full conversation history.

    Returns:
        list of dict: Filtered conversation history relevant to the agent.
    """
    messages = []
    for line in conversation_lines:
        if line['Speaker'] == agent_name:
            role = 'assistant'
        else:
            role = 'user'
        messages.append({"role": role, "content": line['Content']})
    return messages

def main():
    # Initialize the Ollama LLM
    llm = OllamaLLM(model=OLLAMA_MODEL)

    conversation_lines = [{'Speaker': 'Speaker 1', 'Content': 'Hi'}]  # Start with Speaker 1 saying "Hi"
    num_turns = 10  # Total number of turns
    speaker = "Speaker 2"

    for turn in range(num_turns):
        if speaker == "Speaker 2":
            # Build conversation history for Speaker 2
            conversation_history = build_agent_conversation_history("Speaker 2", conversation_lines)
            # Create the full prompt
            prompt = create_prompt(SPEAKER_2_SYSTEM_PROMPT, conversation_history, 'user')

            # Invoke the model
            response = llm.invoke(prompt)
            response = response.strip()
            conversation_lines.append({'Speaker': speaker, 'Content': response})

            if "We have finished" in response:
                print("Conversation ended by Speaker 2.")
                break  # End the conversation when Speaker 2 includes the end condition
            speaker = "Speaker 1"
        else:
            # Build conversation history for Speaker 1
            conversation_history = build_agent_conversation_history("Speaker 1", conversation_lines)
            # Create the full prompt
            prompt = create_prompt(SPEAKER_1_SYSTEM_PROMPT, conversation_history, 'assistant')

            # Invoke the model
            response = llm.invoke(prompt)
            response = response.strip()
            conversation_lines.append({'Speaker': speaker, 'Content': response})
            speaker = "Speaker 2"

    # Create a unique filename with a numeric suffix to avoid overwriting
    file_number = 1
    while os.path.exists(f'conversation_{file_number:02d}.csv'):
        file_number += 1
    filename = f'conversation_{file_number:02d}.csv'

    # Write the conversation to a CSV file in the format 'Speaker: [Content]'
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Conversation'])  # Add a header to indicate the start of the conversation
        for line in conversation_lines:
            writer.writerow([f">{line['Speaker']}"])  # Add speaker with '>'
            writer.writerow([line['Content']])        # Add content on a new line
            writer.writerow([])                       # Add a break for better readability

    print(f"Conversation saved to '{filename}'.")

if __name__ == "__main__":
    main()
