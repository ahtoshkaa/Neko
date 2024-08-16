import os
import anthropic
import openai
import json
import requests
import time
import numpy as np
from numpy.linalg import norm
import re
from time import time,sleep
from uuid import uuid4
import datetime
from openai import OpenAI
from together import Together
from groq import Groq
import google.generativeai as genai
from google.generativeai import types
import google.generativeai.types.generation_types as gen_types
import PIL.Image
import base64

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "F:/AGI/Neko Project v8/service_account.json"

request_count = 0
current_mode = 'SFW'  # This can be 'SFW' or 'NSFW'
PROMPTS_DIR = 'prompts/'
INVENTORY_FILE = 'inventory.json'
PICTURES_DIR = 'pictures/'
SUBCONSCIOUS_DIR = 'subconscious/'


def handle_commands(user_input, anton_vector):
    global current_mode, user_input_counter
    if "NSFW" in user_input.upper():
        current_mode = 'NSFW'
        print("Switched to NSFW mode.")
        return (True, None, None)
    elif "SFW" in user_input.upper():
        current_mode = 'SFW'
        print("Switched to SFW mode.")
        return (True, None, None)
    elif "UNDO" in user_input.upper():
        if undo_last_entry():
            print("Last entry undone successfully.")
            return (True, None, None)
        else:
            print("Undo failed.")
            return (True, None, None)
    elif user_input.lower().startswith("/save"):
        try:
            num_messages = int(user_input.split()[1])
            if save_important_facts(num_messages, anton_vector):
                print("Important facts saved.")
                return (True, None, None)
            else:
                print("Failed to save important facts.")
                return (True, None, None)
        except (IndexError, ValueError):
            print("Invalid command format. Use /save <number>")
            return (True, None, None)
    elif user_input.lower().startswith("/image"):
        try:
            parts = user_input.split("//")
            image_name = parts[0].split()[1].strip()
            user_message = parts[1].strip() if len(parts) > 1 else ""
            print("Image command processed.")
            return (False, image_name, user_message)
        except IndexError:
            print("Invalid command format. Use /image <image_name> // <message>")
            return (False, None, None)
    return (False, None, None)  # No command detected

def undo_last_entry():
    # Undo for chat_logs
    if not delete_recent_files('chat_logs', 2):
        print("Unable to undo: not enough entries in chat_logs.")
        return False

    # Undo for subconscious
    if not delete_recent_files('subconscious', 2):
        print("Unable to undo: not enough entries in subconscious.")
        return False  # You might want to decide if this should also return False

    if not delete_recent_files('physical_state', 2):
        print("Unable to undo: not enough entries in chat_logs.")
        return False
    
    print("Last entry undone successfully (including subconscious).")

    conversation = load_json_files('chat_logs')
    recent_conversation = get_last_items(conversation, 'message', 2)
    subconscious = load_json_files('subconscious')
    recent_subconscious = get_last_items(subconscious, 'message', 1)
    physical_state = load_json_files('physical_state')
    recent_physical_state = get_last_items(physical_state, 'message', 1)
    
    print('\n\n%s' % recent_physical_state)
    print('\n\n%s' % recent_subconscious)
    print('\n\n%s' % recent_conversation)
    return True

def delete_recent_files(directory, file_count):
    try:
        files = sorted([f for f in os.listdir(directory) if f.endswith('.json')], key=lambda x: os.path.getmtime(os.path.join(directory, x)))
        if len(files) < file_count:
            return False  # Not enough files to delete
        for file in files[-file_count:]:
            os.remove(os.path.join(directory, file))
        return True
    except Exception as e:
        print(f"Error while deleting files: {e}")
        return False


def read_file(filepath, is_prompt=False):
    """Reads the content of a file."""
    if is_prompt:
        filepath = os.path.join(PROMPTS_DIR, filepath)
    try:
        with open(filepath, 'r', encoding='utf-8') as infile:
            return infile.read()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None

def save_important_facts(num_messages_to_include, anton_vector):
    # Load the last N messages
    conversation = load_json_files('chat_logs')
    recent_conversation = get_last_items(conversation, 'message', num_messages_to_include)
    conversation_messages = recent_conversation.split('\n\n')

    # Join the conversation messages into a single string
    conversation_string = '\n'.join(conversation_messages)
    #print(conversation_string)



    #### Pull facts and kinks
    all_facts = load_json_files('facts')
    relevant_facts = fetch_items(anton_vector, all_facts, 10)
    #all_kinks = load_json_files('kinks')
    #relevant_kinks = fetch_items(anton_vector, all_kinks, 2)        

    # Load the system prompt for fact extraction
    system = read_file('prompt_save.txt', is_prompt=True)    
    #### Generate response, vectorize, and save
    system = system.replace('<<FACTS>>', relevant_facts)
    #system = system.replace('<<KINKS>>', relevant_kinks)

    # Create the messages for the model
    messages = [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": conversation_string})
    #print(messages)
    while True:
        # Derive the important facts using the model
        important_facts = together_completion(messages)

        # Print the derived facts and ask for user feedback
        print("Derived Fact: ", important_facts)
        user_feedback = input("Is the derived fact satisfactory? (y/n/r): ").strip().lower()

        if user_feedback == 'y':
            # User accepts the fact, proceed to save
            break
        elif user_feedback == 'n':
            # User rejects the fact, exit the loop without saving
            print("Fact rejected. Exiting.")
            return True
        elif user_feedback == 'r':
            # User wants to regenerate the fact, continue the loop
            print("Regenerating the fact...")
            continue
        else:
            print("Invalid input. Please enter 'y' to accept, 'n' to reject, or 'r' to regenerate.")
    
    # Proceed with saving the fact after user accepts it
    
    # Create a vector embedding for the derived facts
    vector = gpt3_embedding(important_facts)

    # Save the derived facts
    timestamp = time()
    timestring = timestamp_to_datetime(timestamp)
    uuid_str = str(uuid4())
    info = {
        "content": "FACT",
        "message": important_facts,
        "time": timestamp,
        "timestring": timestring,
        "uuid": uuid_str,
        "vector": vector
    }
    filename = f'log_{timestamp}_FACT.json'
    
    save_json(f'facts/{filename}', info)

    print(f"Important facts saved to facts/{filename}")
    print(important_facts)

    return True


def write_file(filepath, content):
    """Writes content to a file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as outfile:
            outfile.write(content)
    except Exception as e:
        print(f"Error writing to file {filepath}: {e}")

def load_json(filepath):
    """Loads JSON data from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as infile:
            return json.load(infile)
    except Exception as e:
        print(f"Error loading JSON from {filepath}: {e}")
        return None

def save_json(filepath, payload):
    """Saves JSON data to a file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as outfile:
            json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)
    except Exception as e:
        print(f"Error saving JSON to {filepath}: {e}")



def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")


def gpt3_embedding(content, model='text-embedding-3-large'):
    content = content.encode(encoding='ASCII', errors='ignore').decode()
    response = client.embeddings.create(input=content, model=model)
    # Access the embedding from the response
    vector = response.data[0].embedding
    return vector


def similarity(v1, v2):
    # based upon https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
    return np.dot(v1, v2)/(norm(v1)*norm(v2))  # return cosine similarity


    
def fetch_items(vector, logs, count):
    scores = []
    for item in logs:
        if vector == item['vector']:
            continue  # Skip if it's the same message
        score = similarity(item['vector'], vector)
        item_scored = {'score': score, 'message': item.get('message', 'Default message or description here')}
        scores.append(item_scored)
    
    # Sort and trim the list based on count
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)[:count]

    # Concatenate the facts into a single string, each separated by a newline
    concatenated_facts = '\n'.join([item['message'] for item in ordered])

    return concatenated_facts




def load_json_files(directory, speaker_filter=None):
    files = os.listdir(directory)
    files = [f for f in files if f.endswith('.json')]  # Filter for JSON files
    result = []
    for file in files:
        data = load_json(os.path.join(directory, file))
        # Check if the last message's speaker matches the filter
        if speaker_filter and data.get('speaker') != speaker_filter:
            continue
        result.append(data)
    ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # Sort chronologically
    return ordered
   
   

def get_last_items(items_list, key, limit):
    # Ensure the limit does not exceed the list length
    limited_items = items_list[-limit:] if len(items_list) >= limit else items_list
    # Concatenate the specified field from each item
    output = '\n\n'.join([item[key] for item in limited_items])
    return output


def create_and_save_log(speaker, text, directory='chat_logs'):
    timestamp = time()
    vector = gpt3_embedding(text)
    timestring = timestamp_to_datetime(timestamp)
    message = f'{text}'
    message = text.replace('\n', ' ')
    info = {'speaker': speaker, 'time': timestamp, 'vector': vector, 'message': message, 'uuid': str(uuid4()), 'timestring': timestring}
    filename = f'log_{timestamp}_{speaker}.json'
    
    # Use the specified directory for saving the file
    save_json(f'{directory}/{filename}', info)
    return vector

def extract_facts_from_last_messages(anton_vector):
    # Load the last 4 messages (2 from user and 2 from AI)
    conversation = load_json_files('chat_logs')
    recent_conversation = get_last_items(conversation, 'message', 4)
    
    # Load facts
    all_facts = load_json_files('facts')
    relevant_facts = fetch_items(anton_vector, all_facts, 3)

    # Load the system prompt for fact extraction
    system = read_file('prompt_fact_extraction.txt', is_prompt=True)
    system = system.replace('<<FACTS>>', relevant_facts)

  
    
    # Create the messages for the model
    messages = [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": recent_conversation})

    # Derive the important facts using the model
    response = together_completion(messages)
    print("Extracted Fact: ", response)

    
    # Classifier
    episodic_memory = format_memory()
    system = read_file('prompt_classifier.txt', is_prompt=True)
    system = system.replace('<<EPISODIC_MEMORY>>', episodic_memory) 
    messages = [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": response})
    classifier_response = together_completion(messages)
    print("Classifier Response: ", classifier_response)    

    # Parse classifier response and update memory
    update_memory_file(response, classifier_response)
    return classifier_response

def update_memory_file(fact, classifier_response):
    memory_file = 'memory.json'
    if os.path.exists(memory_file):
        memory_data = load_json(memory_file)
    else:
        memory_data = []

    # Parse classifier response
    category_match = re.search(r'Category: (.+)', classifier_response)
    score_match = re.search(r'Score: (\d+)', classifier_response)

    if category_match and score_match:
        category = category_match.group(1)
        score = int(score_match.group(1))

        # Add new fact to memory
        memory_data.append({"fact": fact, "category": category, "score": score})

        # Save updated memory
        save_json(memory_file, memory_data)
        #print(f"Memory updated with new fact: {fact}")

def decrement_memory_scores():
    memory_file = 'memory.json'
    if os.path.exists(memory_file):
        memory_data = load_json(memory_file)
    else:
        return

    # Decrement scores and remove facts with a score of 0
    updated_memory = []
    for entry in memory_data:
        entry['score'] -= 1
        if entry['score'] > 0:
            updated_memory.append(entry)

    # Save updated memory
    save_json(memory_file, updated_memory)
    #print("Memory scores decremented.")

def load_memory():
    memory_file = 'memory.json'
    if os.path.exists(memory_file):
        return load_json(memory_file)
    return []

def format_memory():
    memory_data = load_memory()

    # Sort the facts based on the score in descending order
    sorted_memory = sorted(memory_data, key=lambda x: x['score'], reverse=True)

    # Extract the text of the facts
    formatted_memory = "\n".join([entry['fact'] for entry in sorted_memory])

    return formatted_memory



def anthropic_completion(messages2, system_main):
    message = client2.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0.0,
        system=system_main,
        messages=messages2
    )
    # Assuming the response is a list of ContentBlocks
    response_text = ""
    for content_block in message.content:
        response_text += content_block.text

    return response_text 

def chatgpt_completion(messages, model="gpt-4o-mini"):
    max_retry = 5
    retry = 0
    while retry < max_retry:
        try:
            response = client.chat.completions.create(model=model, messages=messages, temperature=1, max_tokens=1000, top_p=1,frequency_penalty=0, presence_penalty=0)
            text = response.choices[0].message.content
            return text
        except Exception as oops:
            retry += 1
            print('Error communicating with OpenAI:', oops)
            sleep(1)

    return "ChatGPT error: Unable to reach the API after several attempts."
    
def chatgpt_completion_stream(messages, model="gpt-4o"):
    response = client.chat.completions.create(model=model, messages=messages, temperature=1, max_tokens=1000, top_p=1,frequency_penalty=0, presence_penalty=1, stop=['ANTON:', 'Anton:',], stream=True)
    
    collected_messages = []
    print("\n\n\n\nNEKO: ", end='', flush=True)
    for chunk in response:
        chunk_message = chunk.choices[0].delta.content
        if chunk_message is not None:
            collected_messages.append(chunk_message)
            print(chunk_message, end='', flush=True)
    
    full_reply_content = ''.join(collected_messages)
    return full_reply_content


def together_completion(messages, model="meta-llama/Llama-3-70b-chat-hf", temperature="1.0"):
    max_retry = 5
    retry = 0
    while retry < max_retry:
        try:
            response = client3.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            text = response.choices[0].message.content
            return text
        except requests.exceptions.RequestException as e:
            retry += 1
            print(f"Connection error: {e}. Retrying {retry}/{max_retry}...")
            time.sleep(1)  # Wait a bit before retrying
    raise Exception("Failed to get response after several attempts")

def groq_completion(messages):
    max_retry = 5
    retry = 0
    while retry < max_retry:
        try:
            response = client4.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                temperature=1,
            )
            text = response.choices[0].message.content
            return text
        except requests.exceptions.RequestException as e:
            retry += 1
            print(f"Connection error: {e}. Retrying {retry}/{max_retry}...")
            time.sleep(1)  # Wait a bit before retrying
    raise Exception("Failed to get response after several attempts")

def gemini_completion(messages, image_path=None, model_name='gemini-1.5-pro-exp-0801', temperature=2, top_p=0.95, top_k=64, max_output_tokens=8192):
    global request_count
    
    api_key1 = ""
    api_key2 = ""
    
    # Alternate between API keys
    if request_count % 2 == 0:
        api_key = api_key1
    else:
        api_key = api_key2
    request_count += 1
    
    # api_key = ""
    genai.configure(api_key=api_key)

    # Model configuration
    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_output_tokens": max_output_tokens
    }

    # Safety settings
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ]

    model = genai.GenerativeModel(
        model_name=model_name, 
        generation_config=generation_config, 
        safety_settings=safety_settings,
    )

    chat_history = []

    # Add system message if present
    if messages[0]['role'] == 'system':
        chat_history.append({
            "role": "user",
            "parts": [messages[0]['content']],
        })
        chat_history.append({
            "role": "model",
            "parts": ["Understood, I will adhere to the system message."],
        })

    # Add other messages to chat history
    for message in messages[1:]:
        if message['role'] == 'user':
            chat_history.append({
                "role": "user",
                "parts": [message['content']],
            })
        elif message['role'] == 'assistant':
            chat_history.append({
                "role": "model",
                "parts": [message['content']],
            })

    user_input = messages[-1]['content']
    chat_session = model.start_chat(history=chat_history)
    # Handle image input
    if image_path:
        img = PIL.Image.open(image_path)
        img.show()
        response = chat_session.send_message([user_input, img])
    else:
        response = chat_session.send_message(user_input)

    # Return the response text
    return response.text


### INVENTORY

def detect_inventory_changes(user_input, neko_response, inventory_string):
    interaction_text = f"User: {user_input}\nNeko: {neko_response}"
    
    prompt = read_file('prompt_inventory.txt', is_prompt=True)
    prompt = prompt.replace('<<INVENTORY>>', inventory_string)
    prompt = prompt.replace('<<INTERACTION>>', interaction_text)
        
    # Call the GPT API to analyze the text
    response = together_completion([{"role": "system", "content": prompt}])
    #print(prompt)
    #print(response)
    # Parse the response to identify changes
    changes = parse_inventory_changes(response)
    return changes

def parse_inventory_changes(response_text):
    changes = []
    # Use regex or string manipulation to extract changes from the response
    pattern = r"Inventory Update:\s*- Action: (\w+)\s*- Category: (\w+)\s*- Item: (.+)\s*- Placement: (.+)"
    matches = re.findall(pattern, response_text)
    for match in matches:
        action, category, item, placement = match
        changes.append({"action": action, "category": category, "item": item, "placement": placement})
    return changes


def update_inventory(changes, inventory):
    for change in changes:
        action = change["action"]
        category = change["category"]
        item = change["item"]
        placement = change["placement"]
        
        if category == "clothing":
            if action == "add":
                inventory["clothing"][placement] = item
            elif action == "remove":
                if inventory["clothing"][placement] == item:
                    inventory["clothing"][placement] = None
        
        elif category == "accessories":
            if action == "add":
                inventory["accessories"].append({"item": item, "placement": placement})
            elif action == "remove":
                inventory["accessories"] = [accessory for accessory in inventory["accessories"] if accessory["item"] != item or accessory["placement"] != placement]
        
        elif category == "other":
            if action == "add":
                inventory["other"].append({"item": item, "placement": placement})
            elif action == "remove":
                inventory["other"] = [other_item for other_item in inventory["other"] if other_item["item"] != item or other_item["placement"] != placement]

    return inventory

def load_inventory():
    if os.path.exists(INVENTORY_FILE):
        return load_json(INVENTORY_FILE)
    return {
        "clothing": {
            "ass and pussy": "pink panties",
            "breasts": None,
            "feet": None,
            "legs": "lace stockings",
            "torso": "blouse",
            "waist": "skirt"
        },
        "accessories": [],
        "other": []
    }

def save_inventory(inventory):
    save_json(INVENTORY_FILE, inventory)

def format_inventory(inventory):
    formatted_inventory = "Inventory:\n"
    for category, items in inventory.items():
        if category == "clothing":
            for place, item in items.items():
                item_display = item if item else 'Naked'
                formatted_inventory += f"- {place.capitalize()}: {item_display}\n"
        elif category in ["accessories", "other"]:
            if items:
                formatted_inventory += f"{category.capitalize()}:\n"
                for accessory in items:
                    formatted_inventory += f"  - {accessory['item']} (Placement: {accessory['placement']})\n"
            else:
                formatted_inventory += f"{category.capitalize()}: None\n"
        else:
            formatted_inventory += f"- {category.capitalize()}: {items}\n"
    return formatted_inventory

### SUBCONSCIOUS

def process_subconscious(user_input, anton_vector, inventory_string, episodic_memory, recent_physical_state):
    """Processes the subconscious aspect of the AI's response."""
    conversation = load_json_files('chat_logs', speaker_filter='ANTON')
    recent_ANTON = get_last_items(conversation, 'message', 5)
    
    # Load necessary prompts
    system_subconscious = read_file('prompt_subconscious_system.txt', is_prompt=True)
    user_subconscious = read_file('prompt_subconscious_user.txt', is_prompt=True)

    # Prepare data for the subconscious prompt
    all_facts = load_json_files('facts')
    relevant_facts = fetch_items(anton_vector, all_facts, 5)
    all_profiles = load_json_files('profiles')
    relevant_profiles = fetch_items(anton_vector, all_profiles, 1)

    # Fill in the user prompt template
    user_subconscious = user_subconscious.replace('<<INPUT_RECENT>>', recent_ANTON)
    user_subconscious = user_subconscious.replace('<<INPUT_LAST>>', user_input)
    user_subconscious = user_subconscious.replace('<<FACTS>>', relevant_facts)
    user_subconscious = user_subconscious.replace('<<PROFILES>>', relevant_profiles)
    user_subconscious = user_subconscious.replace('<<INVENTORY>>', inventory_string)
    user_subconscious = user_subconscious.replace('<<EPISODIC_MEMORY>>', episodic_memory)
    user_subconscious = user_subconscious.replace('<<PHYSICAL_STATE>>', recent_physical_state)

    # Build the messages for the subconscious model
    subconscious = build_messages(system_subconscious, conversation_directory='subconscious', num_turns=1)
    subconscious.append({"role": "user", "content": user_subconscious})

    # Log the input to the subconscious
    create_and_save_log('SUBCONSCIOUS_INPUT', user_subconscious, directory='subconscious')

    # Generate the subconscious output
    subconscious_output = together_completion(subconscious)

    # Log the output from the subconscious
    create_and_save_log('SUBCONSCIOUS_OUTPUT', subconscious_output, directory='subconscious')

    # Return the subconscious output
    return subconscious_output

### PHYSICAL STATE

def process_physical_state(user_input, neko_response, recent_physical_state, inventory_string):
    """Processes the physical state of the AI based on the interaction."""
    
    # Load necessary prompts
    system_physical_state = read_file('prompt_physical_state_system.txt', is_prompt=True)
    user_physical_state = read_file('prompt_physical_state_user.txt', is_prompt=True)

    # Fill in the user prompt template
    user_physical_state = user_physical_state.replace('<<PREVIOUS_STATE>>', recent_physical_state)
    user_physical_state = user_physical_state.replace('<<USER_INPUT>>', user_input)
    user_physical_state = user_physical_state.replace('<<NEKO_RESPONSE>>', neko_response)
    user_physical_state = user_physical_state.replace('<<INVENTORY>>', inventory_string)

    # Log the input for physical state tracking
    create_and_save_log('PHYSICAL_STATE_INPUT', user_physical_state, directory='physical_state')

    # Build the messages for the physical state model
    physical_state = build_messages(system_physical_state, conversation_directory='physical_state', num_turns=1)
    physical_state.append({"role": "user", "content": user_physical_state})

    # Generate the physical state output
    physical_state_output = together_completion(physical_state)

    # Log the output of physical state tracking
    create_and_save_log('PHYSICAL_STATE_OUTPUT', physical_state_output, directory='physical_state')

    # Return the physical state output
    return physical_state_output

### MAIN RESPONSE

def generate_main_response(user_input, relevant_facts, relevant_profiles, inventory_string, episodic_memory, subconscious_output, recent_physical_state, image_path=None):
    """Generates the main response of the AI."""
    system_main = read_file('prompt_response_chat.txt', is_prompt=True)
    system_main = system_main.replace('<<FACTS>>', relevant_facts)
    system_main = system_main.replace('<<PROFILES>>', relevant_profiles)
    system_main = system_main.replace('<<INVENTORY>>', inventory_string)
    system_main = system_main.replace('<<EPISODIC_MEMORY>>', episodic_memory)
    system_main = system_main.replace('<<SUBCONSCIOUS>>', subconscious_output)
    system_main = system_main.replace('<<PHYSICAL_STATE>>', recent_physical_state)

    messages = build_messages(system_main, conversation_directory='chat_logs', num_turns=2)
    messages.append({"role": "user", "content": user_input})

    try:
        output = gemini_completion(messages, image_path=image_path)
    except gen_types.BlockedPromptException as e:
        print("BlockedPromptException encountered:", e)
        print("Explicit content detected, falling back to groq_completion.")
        output = groq_completion(messages)

    return output

### BUILD MESSAGES

def build_messages(system_message, conversation_directory='chat_logs', num_turns=1):
    """Builds messages for the language model, including system and past turns.

    Args:
        system_message: The system message for the model.
        conversation_directory: The directory to load past messages from.
        num_turns: The number of recent turns (input/output pairs) to include.

    Returns:
        A list of dictionaries, formatted as messages for the model.
    """

    messages = [{"role": "system", "content": system_message}]
    conversation = load_json_files(conversation_directory)
    recent_conversation = get_last_items(conversation, 'message', num_turns * 2)
    conversation_exchanges = recent_conversation.split('\n\n')

    for i, message in enumerate(conversation_exchanges):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": message})
    return messages


os.environ["OPENAI_API_KEY"] = read_file('key_openai.txt').strip()
client = OpenAI()
messages = []

client2 = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="",
)
messages2 = []

client3 = Together(api_key="")

client4 = Groq(api_key="")

#### load up latest conversation
conversation = load_json_files('chat_logs')
recent_2 = get_last_items(conversation, 'message', 2)
output = get_last_items(conversation, 'message', 1)

inventory = load_inventory()
inventory_string = format_inventory(inventory)
print('\n\n%s' % inventory_string)
print(recent_2)

if __name__ == '__main__':
    user_input_counter = 0  # Initialize the counter
    inventory = load_inventory()
    #print(inventory)
    while True:
        user_input = input('\n\nANTON: ')
        conversation = load_json_files('chat_logs', speaker_filter='ANTON')
        recent_ANTON = get_last_items(conversation, 'message', 3)
        anton_vector = gpt3_embedding(recent_ANTON)


        should_restart, image_name, user_message = handle_commands(user_input, anton_vector)
        if should_restart:
            continue  # Restart the loop if a command was handled that requires it
        if image_name is not None:
            user_input = user_message
            image_path = os.path.join(PICTURES_DIR, image_name)
        else:
            image_path = None
        
        conversation = load_json_files('chat_logs', speaker_filter='ANTON')
        recent_ANTON = get_last_items(conversation, 'message', 3)        
        anton_vector = gpt3_embedding(recent_ANTON)
        user_input_counter += 1

        #### Pull facts and profiles
        all_facts = load_json_files('facts')
        relevant_facts = fetch_items(anton_vector, all_facts, 5)
      
        all_profiles = load_json_files('profiles')
        relevant_profiles = fetch_items(anton_vector, all_profiles, 1) 
        inventory_string = format_inventory(inventory)
        episodic_memory = format_memory()

        all_physical_state = load_json_files('physical_state')
        recent_physical_state = get_last_items(all_physical_state, 'message', 1)   

        ### SUBCONSCIOUS

        subconscious_output = process_subconscious(user_input, anton_vector, inventory_string, episodic_memory, recent_physical_state)
        print('\n\nsubconscious: %s' % subconscious_output)   
        
        
        #### MAIN 

        output = generate_main_response(user_input, relevant_facts, relevant_profiles, inventory_string, episodic_memory, subconscious_output, recent_physical_state, image_path)

        print('\n\nNEKO: %s' % output)   

        create_and_save_log('ANTON', user_input)


        ### INVENTORY
        changes = detect_inventory_changes(user_input, output, inventory_string)
        inventory = update_inventory(changes, inventory)
        save_inventory(inventory)
        inventory_string = format_inventory(inventory)
        print('\n\n%s' % inventory_string)
        create_and_save_log('NEKO', output)
        
        #### FACT EXTRACTION
        fact = extract_facts_from_last_messages(anton_vector)
        #print("Extracted Important Fact: ", fact)
        decrement_memory_scores()

        ### PHYSICAL STATE TRACKING
     
        physical_state_output = process_physical_state(user_input, output, recent_physical_state, inventory_string)
        print('\n\nOUTPUT: %s' % physical_state_output)


             
