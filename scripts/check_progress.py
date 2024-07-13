import openai
import os
import pprint

api_url = os.getenv("ANYSCALE_BASE_URL")
api_key = os.getenv("ANYSCALE_API_KEY")

client = openai.OpenAI(
    base_url = api_url,
    api_key = api_key)
progress = client.fine_tuning.jobs.retrieve("eftjob_qpn81gs5v1vukrgi5yk1snmhvn")

pprint.pprint(progress)

result_file_id = progress.result_files[0]

print(result_file_id)
client.files.retrieve_content(result_file_id)