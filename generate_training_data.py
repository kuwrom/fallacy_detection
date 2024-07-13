import math
import os
import aiohttp
import asyncio
import csv
import json
import logging
from aiohttp import ClientSession, ClientResponseError, ClientError

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

prompt = lambda category, article, article_length: [
    {
        "role": "system",
        "content": f"""System Message: 
                                                You are designed to transform input sentences or paragraphs from specified fallacy categories into detailed articles. Your task involves contextualizing the input fallacy within a narrative, identifying the fallacy type, and providing an explanation for the fallacy presented.                                                
                                                Article Length:
                                                Short: A concise expantion on the input sentence.
                                                Medium: A detailed exploration extending on the input sentence with multipple paragraphs.
                                                Long: An in-depth argument or supportive article that narrates on a broader context with extensive examples and arguments.
                                                Output Format: Format the output as a JSON object with:

                                                [Article]: The narrative adjusted to the requested length.
                                                [Identification]: The fallacy type (e.g., "False Dilemma").
                                                [Explanation]: Detailed explanation of the fallacy.
                                                Aim to balance narrative engagement with informative content, tailored to the specified article size.

                 """,
    },
    {
        "role": "user",
        "content": f"""Category: {category}
                                                Input Data: {article}

                                                Objective:
                                                Transform the input data into a detailed article format, embedding the fallacy within a broader context. The article should subtly include the specified fallacy type, even if the original sentence does not exactly match this fallacy type. The aim is to provide a comprehensive explanation addressing and categorizing the fallacy, ensuring the content strictly contains the mentioned fallacy category only.
                                                
                                                Article Length: {article_length}

                                                Output:

                                                Article:
                                                [Generate a short article based on the input data. The article should introduce additional context, provide background information, and integrate the original sentence into a larger narrative. This narrative must include the specified fallacy type.]

                                                Identification:
                                                [The fallacy type identified should match {category}.]

                                                Explanation:
                                                [Analyze the article, elaborate on why the identified fallacy type is present, and provide an explanation. Discuss why the reasoning in the article is flawed, cite relevant examples or theories if applicable.]
                                """,
    },
]

system_message = f"""
[Fallacy Detection Activated] You're a fallacy detection engine for educational purposes, detect the fallacy type if it exists. the user will give you an article or a premise and after analyzing the input, generate an output as follows. 

Identification of Fallacy Type:
[The fallacy type identified]

A Short Paragraph Explanation of the Fallacy:
[Analyze the article, elaborate on why the identified fallacy type is present, and provide a detailed explanation. Discuss why the reasoning in the article is flawed or logically sound and cite relevant examples or theories if applicable]
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

PARALLEL_REQUESTS = 100  # Number of parallel requests
TARGET_SENTENCES = 10000 # Target number of sentences per file
# paragraph_count = 4

def determine_article_length(processed_count):
    if processed_count < 3300:
        return "short"
    elif processed_count < 6600:
        return "medium"
    else:
        return "long"

# Directory and file setup
data_dir = "data"
organic_dir = os.path.join(data_dir, "organic")
synthetic_dir = os.path.join(data_dir, "synthetic")
training_dir = os.path.join(data_dir, "training")
progress_file = os.path.join(data_dir, "generation_progress.json")

for directory in [organic_dir, synthetic_dir, training_dir]:
    os.makedirs(directory, exist_ok=True)


def load_progress():
    try:
        with open(progress_file, "r") as f:
            progress = json.load(f)
        return progress
    except FileNotFoundError:
        logging.warning(f"{progress_file} not found. Starting with empty progress.")
        return {}


def save_progress(progress):
    with open(progress_file, "w") as f:
        json.dump(progress, f)
    logging.info(f"Progress saved to {progress_file}.")


def update_progress(file_name, count_added):
    progress = load_progress()
    current_count = progress.get(file_name, 0) + count_added
    progress[file_name] = current_count
    save_progress(progress)
    logging.info(f"Updated progress for {file_name}: now {current_count} entries.")


organic_fallacy_seed_files = [
    "ad_hominem.csv",
    "appeal_to_ignorance.csv",
    "equivocation.csv",
    "the_bandwagon.csv",
    "ad_populum.csv",
    "cherry_picking.csv",
    "false_causality.csv",
    "hasty_generalization.csv",
    "red_herring.csv",
    "appeal_to_authority.csv",
    "circular_reasoning.csv",
    "false_dilemma.csv",
    "loaded_question.csv",
    "slippery_slope.csv",
]

synthetic_fallacy_seed_files = [
    "synthetic_ad_hominem.csv",
    "synthetic_appeal_to_ignorance.csv",
    "synthetic_equivocation.csv",
    "synthetic_the_bandwagon.csv",
    "synthetic_ad_populum.csv",
    "synthetic_cherry_picking.csv",
    "synthetic_false_causality.csv",
    "synthetic_hasty_generalization.csv",
    "synthetic_red_herring.csv",
    "synthetic_appeal_to_authority.csv",
    "synthetic_circular_reasoning.csv",
    "synthetic_false_dilemma.csv",
    "synthetic_loaded_question.csv",
    "synthetic_slippery_slope.csv",
]


async def generate_more_seed(
    session, sentence, system_message, semaphore, max_retries=10
):
    retries = 0
    backoff_factor = 2.0
    base_wait_time = 1.0
    logging.info(f"Attempting to process sentence: {sentence[:50]}...")

    while retries < max_retries:
        async with semaphore:  # Limit the number of concurrent requests
            try:
                response = await session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "gpt-4-1106-preview",
                        "messages": [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": sentence},
                        ],
                        "response_format": {"type": "json_object"},
                    },
                )
                response.raise_for_status()  # Raises exception for 4xx/5xx errors

                result = await response.json()

                if "choices" in result and result["choices"]:
                    logging.info(f"Successfully processed sentence: {sentence[:50]}...")

                    fallacies_content = result["choices"][0]["message"]["content"]
                    fallacies_json = json.loads(fallacies_content)

                    return {
                        "fallacies": fallacies_json["fallacies"],
                    }

                logging.error("API response is missing 'choices' or it's empty.")
                return None

            except ClientResponseError as e:
                if e.status in {429, 503}:  # Common statuses to retry on
                    wait_time = base_wait_time * (backoff_factor**retries)
                    logging.warning(
                        f"Rate limit or server error (status {e.status}). Retrying in {wait_time} seconds..."
                    )
                    await asyncio.sleep(wait_time)
                    retries += 1
                else:
                    logging.error(f"Request failed with status {e.status}: {e.message}")
                    return None
            except ClientError as e:
                logging.error(f"Network error: {e}")
                wait_time = base_wait_time * (backoff_factor**retries)
                await asyncio.sleep(wait_time)
                retries += 1
            except Exception as e:
                logging.error(
                    f"Failed to process sentences due to an unexpected error: {e}"
                )
                return None

    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")


async def fetch_and_process_article(session, prompt, semaphore, max_retries=10):
    retries = 0
    backoff_factor = 2.0  # Exponential backoff factor
    base_wait_time = 1.0  # Base wait time in seconds before retrying

    while retries < max_retries:
        async with semaphore:
            try:
                response = await session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "gpt-4-1106-preview",
                        "messages": prompt,
                        "response_format": {"type": "json_object"},
                    },
                )
                response.raise_for_status()

                result = await response.json()

                if "choices" in result and result["choices"]:
                    article_content = json.loads(
                        result["choices"][0]["message"]["content"]
                    )
                    
                    identification_str = "\n".join(article_content['Identification']) if isinstance(article_content['Identification'], list) else article_content['Identification']
                    explanation_str = "\n".join(article_content['Explanation']) if isinstance(article_content['Explanation'], list) else article_content['Explanation']
                    content_str = identification_str + "\n" + explanation_str
                    
                    print("Sucessfully Processed: " + content_str[:50])
                    formatted_data = {
                        "messages": [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": article_content["Article"]},
                            {
                                "role": "assistant",
                                "content": content_str,
                            },
                        ]
                    }
                    # logging.info(f"Successfully processed article: {article_content["Article"][:50]}...")

                    return formatted_data

                logging.error("API response is missing 'choices' or it's empty.")
                return None

            except ClientResponseError as e:
                if e.status in {429, 503}:  # Common statuses to retry on
                    wait_time = base_wait_time * (backoff_factor**retries)
                    logging.warning(
                        f"Rate limit or server error (status {e.status}). Retrying in {wait_time} seconds..."
                    )
                    await asyncio.sleep(wait_time)
                    retries += 1
                else:
                    logging.error(f"Request failed with status {e.status}: {e.message}")
                    return None
            except ClientError as e:
                logging.error(f"Network error: {e}")
                wait_time = base_wait_time * (backoff_factor**retries)
                await asyncio.sleep(wait_time)
                retries += 1
            except Exception as e:
                logging.error(
                    f"Failed to process article due to an unexpected error: {e}"
                )
                return None
    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")


async def process_sentences(sentences, system_message):
    tasks = []
    semaphore = asyncio.Semaphore(PARALLEL_REQUESTS)
    async with ClientSession() as session:
        for sentence in sentences:
            task = asyncio.ensure_future(
                generate_more_seed(
                    session=session,
                    sentence=sentence,
                    system_message=system_message,
                    semaphore=semaphore,
                )
            )
            tasks.append(task)
        responses = await asyncio.gather(*tasks)
        fallacies_responses = [
            response for response in responses if response is not None
        ]
        return fallacies_responses


async def process_articles(articles, category, current_progress):
    tasks = []
    semaphore = asyncio.Semaphore(PARALLEL_REQUESTS)
    local_counter = 0 
    async with ClientSession() as session:
        for article in articles:
            current_article_length = determine_article_length(current_progress + local_counter)
            task = asyncio.ensure_future(
                fetch_and_process_article(
                    session,
                    prompt(category=category, article=article, article_length=current_article_length),
                    semaphore,
                )
            )
            tasks.append(task)
            local_counter+=1
        responses = await asyncio.gather(*tasks)
        return responses

def read_csv(file_name, limit=None, start=0, data_type="organic"):
    file_path = os.path.join(data_dir, data_type, file_name)
    articles = []

    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)  # Convert iterator to list to support re-iteration

    total_rows = len(rows)
    if total_rows == 0:
        return articles  # Return empty if the file has no content

    # Calculate the actual start index in a cyclic manner
    start = start % total_rows

    while len(articles) < limit:
        for i in range(start, total_rows):
            if len(articles) >= limit:
                break  # Exit if we've collected enough articles
            articles.append(rows[i]["source_article"])
        start = 0  # Reset start to 0 to loop from the beginning

    return articles

def save_to_jsonl(data, file_name, data_type="training"):
    file_path = os.path.join(data_dir, data_type, file_name)
    count_added = 0

    with open(file_path, "a", encoding="utf-8") as f:
        for record in data:
            if record:
                f.write(json.dumps(record) + "\n")
                count_added += 1

    update_progress(file_name, count_added)


def save_to_csv(data, file_name, data_type="synthetic"):
    file_path = os.path.join(data_dir, data_type, file_name)
    count_added = 0
    with open(file_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if os.stat(file_path).st_size == 0:
            writer.writerow(["source_article"])
        for record in data:
            if record:
                # single_items = [{'fallacy': fallacy} for fallacy in record['fallacies']]
                for fallacy in record.get("fallacies", []):
                    fallacy_list = [fallacy]
                    writer.writerow(fallacy_list)
                    count_added += 1

    update_progress(file_name, count_added)


if __name__ == "__main__":
    progress = (
        load_progress()
    )  # Ensure you have this function defined to manage progress
    loop = asyncio.get_event_loop()

    for file_name in organic_fallacy_seed_files:
        file_path = os.path.join(organic_dir, file_name)
        synthetic_file_name = "synthetic_" + file_name
        synthetic_file_path = os.path.join(synthetic_dir, synthetic_file_name)
        current_progress = progress.get(synthetic_file_name, 0)

        if current_progress >= TARGET_SENTENCES:
                continue

        fallacy_type = file_name.split(".")[0].replace("_", " ")
        seed_system_message = f"""[Fallacy Generation Activated] You're a fallacy generation system for educational purposes, generating {fallacy_type} fallacies. the user will give you an example but generate more examples across different domains. don't generate explanations just generate 10 fallacies. and respond in json as follows eg: "fallacies": {"sentence 1", "sentence 2" "list goes on up to 10 sentences"}"""

        while current_progress < TARGET_SENTENCES:
            
            sentences_needed = min(TARGET_SENTENCES - current_progress, 100)
            sentences = read_csv(file_name, limit=sentences_needed, start= math.ceil(current_progress / 10), data_type="organic")
            
            results = loop.run_until_complete(
                process_sentences(sentences, seed_system_message)
            )
            save_to_csv(results, synthetic_file_name)

            current_progress = load_progress().get(synthetic_file_name, 0)
            if current_progress >= TARGET_SENTENCES:
                break
            
    for synthetic_file_name in synthetic_fallacy_seed_files:
        jsonl_file_name = synthetic_file_name.split(".")[0] + ".jsonl"
        file_path = os.path.join(training_dir, jsonl_file_name)

        current_progress = progress.get(jsonl_file_name, 0)

        if current_progress >= TARGET_SENTENCES:
            continue

        # articles = read_csv(synthetic_file_name, limit=1)

        while current_progress < TARGET_SENTENCES:
            
            articles_needed = min(TARGET_SENTENCES - current_progress, 100)
            articles = read_csv(synthetic_file_name, limit=articles_needed, start=current_progress, data_type="synthetic")
            
            results = loop.run_until_complete(
                process_articles(articles, system_message, current_progress)
            )
            save_to_jsonl(results, jsonl_file_name)

            current_progress = load_progress().get(jsonl_file_name, 0)
            if current_progress >= TARGET_SENTENCES:
                break
