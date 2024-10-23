#тесит o1 thinking
# 1. Import necessary libraries
import openai
from openai import OpenAI
from config import Config
client = OpenAI(api_key=Config.OPENAI_API_KEY)
import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from googleapiclient.discovery import build
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import Command
import asyncio
from config import Config
from tqdm.asyncio import tqdm
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import random
import time
import re
import pickle
from difflib import SequenceMatcher
import fitz  # PyMuPDF for PDF handling
import docx  # python-docx for Word document handling

# 2. Initialize external services
# 2.1 Initialize OpenAI
# TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(base_url=Config.OPENAI_API_BASE)'
# openai.api_base = Config.OPENAI_API_BASE

# 2.2 Initialize Google Search API
google_service = build("customsearch", "v1", developerKey=Config.GOOGLE_API_KEY)

# 2.3 Initialize Telegram Bot
bot = Bot(token=Config.TELEGRAM_TOKEN)
dp = Dispatcher()

# 2.4 Initialize Rich Console for fancy statuses
console = Console()

# 3. Load data
# 3.1 Load "true-talks.json"
with open("true-talks.json", "r", encoding="utf-8") as file:
    true_talks = json.load(file)

# 3.2 Load files from the "knowledge" folder
knowledge_folder = "knowledge"
knowledge_files = []
if os.path.exists(knowledge_folder):
    for filename in os.listdir(knowledge_folder):
        file_path = os.path.join(knowledge_folder, filename)
        if filename.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as file:
                knowledge_files.append(json.load(file))
        elif filename.endswith(".pdf"):
            try:
                doc = fitz.open(file_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                knowledge_files.append({"USER": text, "BOT": "Extracted information from PDF"})
            except Exception as e:
                console.print(f"[red]PDF error: {e}[/red]")
        elif filename.endswith(".docx"):
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            knowledge_files.append({"USER": text, "BOT": "Extracted information from DOCX"})

# 4. Cache files for storing embeddings
cache_file = "true_talks_embeddings.pkl"
knowledge_cache_file = "knowledge_embeddings.pkl"

# 5. Functions
# 5.1 Function to create embeddings
def create_embeddings(text):
    # Define the maximum allowed tokens for the model
    max_tokens = 8192
    # Set a reduced chunk size to stay well within the limit
    chunk_size = 900

    # Split the text into tokens
    tokens = text.split()

    # Split the text into manageable chunks if it exceeds chunk_size
    if len(tokens) > chunk_size:
        chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    else:
        chunks = [tokens]

    embeddings = []
    for chunk in chunks:
        chunk_text = " ".join(chunk)
        # Generate embeddings for each chunk
        response = client.embeddings.create(input=[chunk_text], model=Config.OPENAI_EMBED_MODEL)
        embeddings.append(response.data[0].embedding)

    # Calculate the average embedding if multiple chunks were used
    if len(embeddings) > 1:
        return np.mean(embeddings, axis=0).tolist()
    
    return embeddings[0]

# 5.2 Function to index true-talks.json and knowledge folder using Hybrid RAG and cache for fast answers
def index_and_cache_knowledge():
    # 5.2.1 Indexing "true-talks.json"
    if not true_talks:
        console.print("[red]Error: 'true-talks.json' is empty or could not be loaded.[/red]")
        return

    embeddings_cache = {}
    total_entries = len(true_talks)

    console.print("[yellow]Indexing and caching 'true-talks.json' using Hybrid RAG...[/yellow]")

    # Animation for indexing process
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, complete_style="green"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Indexing and caching embeddings...", total=total_entries)

        for idx, entry in enumerate(true_talks):
            user_input = entry["USER"]
            embedding = create_embeddings(user_input)
            embeddings_cache[user_input] = {
                "embedding": embedding,
                "response": entry["BOT"]
            }

            progress.update(task, advance=1)
            time.sleep(0.5) 

    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings_cache, f)

    console.print("[green]Embeddings indexed and cached successfully for 'true-talks.json'![/green]")

    # 5.2.2 Indexing "knowledge" folder
    knowledge_embeddings_cache = {}
    total_files = len(knowledge_files)
    console.print("[yellow]Indexing and caching knowledge files...[/yellow]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, complete_style="green"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Indexing knowledge embeddings...", total=total_files)

        for idx, knowledge in enumerate(knowledge_files):
            user_input = knowledge.get("USER")
            if user_input:
                embedding = create_embeddings(user_input)
                knowledge_embeddings_cache[user_input] = {
                    "embedding": embedding,
                    "response": knowledge["BOT"]
                }

            progress.update(task, advance=1)
            time.sleep(0.5) 

    with open(knowledge_cache_file, 'wb') as f:
        pickle.dump(knowledge_embeddings_cache, f)

    console.print("[green]Knowledge embeddings indexed and cached successfully![/green]")

# 5.3 Check if embeddings are already cached, else create them
try:
    with open(cache_file, 'rb') as f:
        embeddings_cache = pickle.load(f)
        console.print("[green]Loaded cached embeddings from 'true-talks.json'.[/green]")
except FileNotFoundError:
    console.print("[yellow]No cached embeddings found for 'true-talks.json'. Indexing and creating new cache...[/yellow]")
    index_and_cache_knowledge()
    with open(cache_file, 'rb') as f:
        embeddings_cache = pickle.load(f)

try:
    with open(knowledge_cache_file, 'rb') as f:
        knowledge_embeddings_cache = pickle.load(f)
        console.print("[green]Loaded cached embeddings from knowledge files.[/green]")
except FileNotFoundError:
    console.print("[yellow]No cached embeddings found for knowledge files. Indexing and creating new cache...[/yellow]")
    index_and_cache_knowledge()
    with open(knowledge_cache_file, 'rb') as f:
        knowledge_embeddings_cache = pickle.load(f)

# 5.4 Function to find the closest match using cosine similarity
def find_closest_match(user_input, embeddings_cache):
    user_input_embedding = create_embeddings(user_input)
    highest_similarity = 0
    best_match = None

    for cached_input, data in embeddings_cache.items():
        cached_embedding = data["embedding"]
        similarity = cosine_similarity([user_input_embedding], [cached_embedding])[0][0]

        if similarity > highest_similarity and similarity > 0.95:
            highest_similarity = similarity
            best_match = data['response']

    return best_match

# 5.5 Matrix-inspired animated loading bar
def matrix_loading_bar():
    console.print("[bold green]Initializing Bot...[/bold green]\n")
    matrix_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, complete_style="green"),
        console=console,
    ) as progress:
        task = progress.add_task("[green]Bot Loading...", total=20)

        for i in range(20):
            random_text = "".join(random.choice(matrix_chars) for _ in range(80))
            console.print(f"[green]{random_text}[/green]", style="bold")
            progress.update(task, advance=1)
            time.sleep(0.2)

        console.print("[bold green]Bot is now active![/bold green]")

# 5.6 Chain of Thought - Break down reasoning steps
def breakdown_reasoning_steps(user_input):
    steps = [
        "Step 1: Understand the intent of the input",
        "Step 2: Extract key keywords from the input",
        "Step 3: Match input with stored knowledge or external sources",
        "Step 4: Validate response against the intended output",
        "Step 5: Justify response and provide reasoning",
        "Step 6: Use o1 thinking to validate the composed output",
        "Step 7: The final output must be relevant, robust and with total capacity of  350 tokens"
    ]
    return steps

# 5.7 CoT reasoning function for validation with added integrity check
def advanced_cot_validation(reasoning_step, data, expected=None):
    with Progress(
        SpinnerColumn(),
        TextColumn(f"[progress.description]{reasoning_step}..."),
        console=console,
    ) as progress:
        task = progress.add_task(f"[yellow]{reasoning_step}...", start=False)
        progress.update(task, advance=1)
        time.sleep(1)

        if reasoning_step == "Validating User Input":
            if not isinstance(data, str) or len(data.strip()) == 0:
                raise ValueError("Invalid input detected. Input cannot be empty.")
            if len(data) < 5:
                console.print("[red]Warning: The input appears too short and might be unclear.[/red]")

        elif reasoning_step == "Validating Answer from True Talks":
            if expected and not SequenceMatcher(None, data.lower(), expected.lower()).ratio() > 0.9:
                console.print("[red]The matched answer might not be accurate. Double-checking...[/red]")
                return False

        elif reasoning_step == "Validating GPT Response":
            if "error" in data.lower() or len(data.split()) < 3:
                console.print("[red]Invalid response generated. Regenerating response...[/red]")
                return False

        elif reasoning_step == "Final Output Integrity Check":
            if not data.endswith(('.', '!', '?')):
                console.print("[red]The final output does not seem to be complete. Checking for coherence...[/red]")
                return False
            if len(data.split()) < 5:
                console.print("[red]The response seems too brief. It may not fully address the query.[/red]")
                return False

        console.print(f"[green]{reasoning_step} passed![/green]")
    return True

# 5.8 Apply CoT Validation and breakdown input
async def apply_cot_to_input(input_text):
    reasoning_steps = breakdown_reasoning_steps(input_text)
    for step in reasoning_steps:
        console.print(f"[blue]{step}[/blue]")

    advanced_cot_validation("Validating User Input", input_text)
    keywords = await extract_keywords(input_text)

    if not keywords or len(keywords) < 1:
        console.print("[red]Keyword extraction failed. Input might be ambiguous.[/red]")

    validated_input = input_text.strip()

    if not SequenceMatcher(None, validated_input.lower(), input_text.lower()).ratio() > 0.75:
        console.print("[yellow]Validated input differs significantly from the original question.[/yellow]")

    return validated_input, keywords

# 5.9 Define the extract_keywords function
async def extract_keywords(input_text):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Extracting keywords...", start=False)

        for _ in range(3):
            progress.update(task, advance=1)
            await asyncio.sleep(0.5)

    words = re.findall(r'\b\w+\b', input_text)
    keywords = [word.lower() for word in words if len(word) > 3]

    return keywords

# 5.10 Define the search_google function
async def search_google(query):
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[blue]Searching Google...", start=False)
            for _ in range(3):
                progress.update(task, advance=1)
                await asyncio.sleep(0.5)

        search_response = google_service.cse().list(
            q=query,
            cx=Config.GOOGLE_SEARCH_ENGINE_ID,
            num=3
        ).execute()

        search_results = search_response.get('items', [])

        return search_results
    except Exception as e:
        console.print(f"[red]Error occurred during Google Search: {e}[/red]")
        return []

# 5.11 Justify the output based on reasoning steps
async def justify_output(final_output, reasoning_steps):
    return f"The answer '{final_output}' was generated based on the following steps: {', '.join(reasoning_steps)}"

# 5.12 Define the generate_output function
async def generate_output(validated_input, keywords):
    """
    Generate the output using the OpenAI API based on validated input and keywords.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[green]Generating output using OpenAI API...", start=False)

        for _ in range(3):
            progress.update(task, advance=1)
            await asyncio.sleep(0.5)

    try:
        response = client.chat.completions.create(model=Config.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant providing mobile marketing FAQ."},
            {"role": "user", "content": validated_input}
        ],
        max_tokens=350,
        n=1,
        stop=None,
        temperature=0.3)

        return response.choices[0].message.content.strip()

    except Exception as e:
        console.print(f"[red]Error generating output: {e}[/red]")
        return "An error occurred while generating the response."

# 5.13 Handle user messages with o1 model inspired thinking
async def handle_message(message: Message):
    user_message = message.text
    pre_defined_answer = find_closest_match(user_message, embeddings_cache)

    if pre_defined_answer:
        validated_output = advanced_cot_validation("Final Response Validation", pre_defined_answer)
        await message.answer(pre_defined_answer)
        return

    validated_input, keywords = await apply_cot_to_input(user_message)
    search_results = await search_google(validated_input)

    response = await generate_output(validated_input, keywords)
    reasoning_steps = breakdown_reasoning_steps(validated_input)
    justified_response = await justify_output(response, reasoning_steps)

    validated_final_output = advanced_cot_validation("Final Output Integrity Check", response)
    await message.answer(response)

# 5.14 Define /start command
async def start_command(message: Message):
    await message.answer("Hello! Welcome to the Mobile Marketing FAQ bot. You can start by asking any question related to mobile marketing.")

# 5.15 Main function
async def main():
    matrix_loading_bar()

    dp.message.register(start_command, Command(commands=['start']))
    dp.message.register(handle_message)

    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())