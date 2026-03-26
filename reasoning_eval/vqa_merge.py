from openai import OpenAI
import simplejson as json
import base64
import os

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def call_gpt(data, GPT_URL, GPT_KEY, GPT_MODEL):

    client = OpenAI(
                base_url=GPT_URL,
                api_key=GPT_KEY
                )
   
    if "frame_path" in data:  ################
        prompt_system = """
            You are a video understanding assistant.  
            Answer the user's questions and explain the reasons based ONLY on the provided video frames. 
            Do NOT guess or hallucinate.
            For each queation, answer strictly in JSON Format: [{"question": "repeat the question", "answer":"Yes or No", "reason":"the reason"}]
            For each input video, if there are multiple questions, you MUST return the answers as a JSON list of dictionaries.
            Example output:
            [
            {
                "question": "repeat the question",
                "answer": "Yes or No",
                "reason": "the reason"
            },
            {
                "question": "repeat the question",
                "answer": "Yes or No",
                "reason": "the reason"
            }
            ]
            Do NOT wrap the JSON output in markdown code blocks (no ```json, no ```).
            Return only a valid JSON array.
             """
        
        raw_frames = data["frame_path"]
        questions = data["questions"]


        frames = []
        for raw_frame in raw_frames:
            frames.append(encode_image(raw_frame))

        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": [
                *[{"type": "image_url",  "image_url": {"url": f"data:image/jpeg;base64,{path}"}} for path in frames],
                *[{"type": "text", "text": f"{q}"} for q in questions]
            ]}
        ]
        flag = 1
        resps = []
        single_score = {}
        while(flag <= 5):
            try:
                response = client.chat.completions.create(
                    model=GPT_MODEL,
                    #model="gpt-4o",
                    messages=messages,
                    temperature=0.0
                )
                raw_response = response.choices[0].message.content
                #print(raw_response)
                resp  = json.loads(response.choices[0].message.content)
                count = len(resp)
                score = 0
                for item in resp:
                    item["task_id"] = data["task_id"]
                    resps.append(item)
                    if 'yes' in item["answer"].lower():
                        score  = score + 1
                final_score = score / count
                
                single_score["task_id"] = data['task_id']
                single_score['score'] = final_score
                flag = 10
            except Exception as e:
                print(f"{data['task_id']} reasoning eval {flag} time fail: {e}")
                flag += 1
                continue
        if flag == 6:
            print(f"{data['task_id']} fail over 5 times!")
            single_score["task_id"] = data["task_id"]
            single_score['score'] = "bad response"
            result = {}
            result['task_id'] = data["task_id"]
            result['status'] = "bad response"
            resps.append(result)
            
        return resps, single_score  #list dict
    
    else:
        print(f"no frames for {data['task_id']}")
     
        
    
