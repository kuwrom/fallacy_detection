import openai
import os

api_url = os.getenv("ANYSCALE_BASE_URL")
api_key = os.getenv("ANYSCALE_API_KEY")

client = openai.OpenAI(
    base_url = api_url,
    api_key = api_key)
training_file_id = client.files.create(
    file=open('ft_dataset/ft.jsonl','rb'),
    purpose="fine-tune",
).id

print(training_file_id)

valid_file_id = client.files.create(
    file=open('ft_dataset/ft_validation.jsonl','rb'),
    purpose="fine-tune",
).id

print(valid_file_id)