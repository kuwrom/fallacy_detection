import json
import openai
import os

api_url = os.getenv("ANYSCALE_BASE_URL")
api_key = os.getenv("ANYSCALE_API_KEY")

if not api_url or not api_key:
    raise ValueError("API URL or API Key is missing. Please set the environment variables correctly.")


client = openai.OpenAI(
    base_url = api_url,
    api_key = api_key
    )

try:
    # Create training file
    # with open('train.jsonl', 'rb') as f:
    #     training_file_id = client.files.create(
    #         file=f,
    #         purpose="fine-tune",
    #     ).id

    # Create validation file
    # with open('valid.jsonl', 'rb') as f:
    #     valid_file_id = client.files.create(
    #         file=f,
    #         purpose="fine-tune",
    #     ).id

    # Define model and start fine-tuning job
    # model = "meta-llama/Llama-2-13b-chat-hf"
    # finetuning_job_id = client.fine_tuning.jobs.create(
    #     training_file="file_cyuh15jdfmvl2tuav3rsv1tihg",
    #     validation_file="file_fy3lxavhkhmztt7e858jt8lzu5",
    #     model=model,
    #     hyperparameters={"n_epochs": 1}
    # ).id

    # print(f"Fine-tuning job ID: {finetuning_job_id}")
    # print(client.fine_tuning.jobs.list())
    
    # file_fy3lxavhkhmztt7e858jt8lzu5 valid
    # file_cyuh15jdfmvl2tuav3rsv1tihg train
    # eftjob_81dmc97xyxi5d9ggyqin2waukp
    # {"messages": [{"role": "system", "content": ""}, {"role": "user", "content": "Investing in the stock market requires a sound understanding of financial principles, market trends, and a certain level of acumen to make informed decisions. Debates often arise among enthusiasts and analysts on various strategies and market predictions. However, a recent discussion took a slightly different turn when one of the participants was discredited not for the content of their argument, but for an unrelated skill: their ability to park a car. 'You expect me to believe your argument about the stock market when you can't even park your car straight?' one member sneered, attempting to undermine the other's stance on the matter. This conflation of a personal flaw with a judgment on professional competence is both amusing and worrisome as an indicator of critical discussion going awry."}, {"role": "assistant", "content": "Ad Hominem\nThe Ad Hominem fallacy occurs when the argument is directed against a person's character or personal traits instead of addressing the substance of the argument or the evidence presented. In the case of the stock market argument, dismissing someone's views based on their parking ability is irrelevant to their knowledge or competence in stock market investing. The reasoning is flawed because it distracts from the content of the argument and instead attacks an unrelated characteristic of the individual, which has no bearing on the validity of their claims about the stock market. The violation is clear: the argument's validity cannot be accurately assessed by the individual's personal skill in a completely unrelated task, such as parking a car."}]}
    
    model = "meta-llama/Llama-2-13b-chat-hf:hruy:6cub5As"
    messages = [
            {
                "role": "system","content": "[Fallacy Detection Activated] You're a fallacy detection engine for educational purposes, detect the fallacy type if it exists. the user will give you an article or a premise and after analyzing the input, generate an output as follows. \nIdentification of Fallacy Type:\n[The fallacy type identified]\nA Short Paragraph Explanation of the Fallacy:\n[Analyze the article, elaborate on why the identified fallacy type is present, and provide a detailed explanation. Discuss why the reasoning in the article is flawed or logically sound and cite relevant examples or theories if applicable]\n"
            },
            {
                "role": "user", "content": "Investing in the stock market requires a sound understanding of financial principles, market trends, and a certain level of acumen to make informed decisions. Debates often arise among enthusiasts and analysts on various strategies and market predictions. However, a recent discussion took a slightly different turn when one of the participants was discredited not for the content of their argument, but for an unrelated skill: their ability to park a car. 'You expect me to believe your argument about the stock market when you can't even park your car straight?' one member sneered, attempting to undermine the other's stance on the matter. This conflation of a personal flaw with a judgment on professional competence is both amusing and worrisome as an indicator of critical discussion going awry."
            }
        ]
    
    result = client.chat.completions.create(
        model = model,
        messages = messages,
        temperature = 0)
    print(result.choices[0].message.content)
    
    # print(client.files.list())
    # file_content = client.files.content("file_qll4rrr2lvr8u28g3mp3ar5p24")
    # json_data = json.loads(file_content.decode("utf-8"))
    # filename = 'saved_result.json'
    # with open(filename, 'w') as json_file:
    #     json.dump(json_data, json_file)
    # file_id = 'file_qll4rrr2lvr8u28g3mp3ar5p24'
    # content = client.files.retrieve_content(file_id)
    
    # 

    # print(content)
    
    

except Exception as e:
    print(f"An error occurred: {e}")