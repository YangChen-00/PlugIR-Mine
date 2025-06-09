# ======================Configuration for Image Retrieval Modules====================
BLIP_ITR_MODEL_PATH = 'Salesforce/blip-itm-large-coco'
BLIP_CAPTION_MODEL_PATH = 'Salesforce/blip-image-captioning-large'

TOP_N_NUM = 50
CLUSTER_NUM = 10

# ======================Configuration for LLM modules====================
# LLM Connector configuration
LLM_API_KEY = "sk-DsWTD0DPOkqL2RYKXB8PDSEehWwdsncNimaoALFKDlSLkGCY"
LLM_BASE_URL = "https://api.aiearth.dev/v1"
LLM_MODEL_NAME = 'gpt-3.5-turbo'

# ======================Configuration for MLLM modules====================
MLLM_API_KEY = "sk-hccvgxzvtizkzkpemzhawylgoenzphkdcgceqqxjwsgdkzah"
MLLM_BASE_URL = "https://api.siliconflow.cn/v1"
MLLM_MODEL_NAME = 'Qwen/Qwen2.5-VL-72B-Instruct'

# ======================Corpus and Embedding Configuration====================
# Corpus path
IMG_ROOT_PATH = '/home/chenyang/datasets/VisDial/VisualDialog_val2018/'
# Corpus embedding root path
IMG_EMBEDDING_ROOT_PATH = '/home/chenyang/projects/PlugIR_mine/embeddings/'

# ======================Hyperparameters for LLM modules====================
# Hyperparameters for each module
QUESTIONER_HYPERPARAMS = {
    "max_tokens": 32,  # Maximum number of tokens in the generated question
    # "temperature": 0.7,  # Controls randomness in the output
    "temperature": 0.7,  # Controls randomness in the output
    "num_questions": 5  # Number of questions to generate
}

REFORMULATOR_HYPERPARAMS = {
    "max_tokens": 512,  # Maximum number of tokens in the reformulated caption
    "temperature": 0.0,  # Lower temperature for more deterministic output
    "n": 1
}

FILTERING_HYPERPARAMS = {
    "max_tokens": 512,  # Maximum number of tokens in the reformulated caption
    "temperature": 0.0,  # Lower temperature for more deterministic output
    "n": 1
}

USER_SIMULATOR_HYPERPARAMS = {
    "max_tokens": 512,  # Maximum number of tokens in the answer
    "temperature": 0.7,  # Lower temperature for more deterministic output
    "n": 1
}

# ====================Prompts for LLM modules====================
# Table 17: 1-shot prompting example for LLM questioner of baseline.
QUESTIONER_BASELINE_PROMPT = {
    "System": """
        You are a proficient question generator tasked with aiding in the retrieval of a target image. Your role is to generate questions about the target image of the description via leveraging two key information sources:
        [Description]: This is a concise explanation of the target image. 
        [Dialogue]: Comprising question and answer pairs that seek additional details about the target image. 
        NOTE: Your generated question about the description must be clear, succinct, and concise, while differing from prior questions in the [Dialogue].
        """,
    
    "User_example": """
        [Description] a man is doing a trick on a skateboard 
        [Dialogue] Question: What type of trick is the man performing on the skateboard? Answer: a jump Question: What is the location of the jump trick being performed? Answer: a skate park 
        Question:
        """,
    
    "Assistant": """
        Question: what is the outfit of the man performing the jump trick at a skate park?
        """,
    
    "User_query": """
        [Description] {initial_description}
        [Dialogue] {dialogues} 
        Question:
        """,
}

# Table 18: 1-shot prompting example for LLM questioner utilizing CoT and the additional context from the set of retrieval candidates.
QUESTIONER_COT_PROMPT = {
    "System": """
        You are a proficient question generator tasked with aiding in the retrieval of a target image. Your role is to generate questions about the target image of the description via leveraging three key information sources:
        [Retrieval Candidates]: These are captions of images which are the candidates of the retrieval task for the target image described in [Description].
        [Description]: This is a concise explanation of the target image. 
        [Dialogue]: Comprising question and answer pairs that seek additional details about the target image. 
        You should craft a question that narrows down the options for the attributes of the target image through drawing the information from the retrieval candidates. 
        NOTE: The generated question about the target image must be clear, succinct, and concise. Also, the question should only be asked about common objects in the description and candidates, which cannot be answered only from the description and the dialogue.
        """,
    
    "User_example": """
        [Retrieval Candidates] 0. man in yellow shirt 1. a boy in a skateboard park 2. the biker is performing a trick 3. a man in a green hat doing half-pipe with a skateboard 4. a skateboarding man catches the air in the midst of a trick 
        [Description] a man is doing a trick on a skateboard 
        [Dialogue] Question: what type of trick is the man performing on the skateboard? Answer: a jump Question: what is the location of the jump trick being performed? Answer: a skate park
        Question:
        """,
    
    "Assistant": """
        Question: what is the outfit of the man performing the jump trick at a skate park?
        """,
    
    "User_query": """
        [Retrieval Candidates] {retrieval_candidates}
        [Description] {initial_description}
        [Dialogue] {dialogues} 
        Question:
        """,
}

# Table 20: 1-shot prompting example for LLM to reformulate the dialogue context.
REFORMULATOR_PROMPT = {
    "System": """
        Your role is to reconstruct the [Caption] with the additional information given by following [Dialogue] to get reconstructed [New Caption].
        NOTE: The reconstructed [New Caption] should be concise and in appropriate form to retrieve a target image from a pool of candidate images.
        """,
    
    "User_example": """
        [Caption]: a woman sits on a bench holding a guitar in her lap 
        [Dialogue]: Question: is this in a park? Answer: yes, i believe it is Question: are there others around? Answer: no, she is alone Question: does she have a collection bucket? Answer: no Question: is her hair long? Answer: yes, pretty long Question: is she wearing a dress? Answer: i don’t think so, hard to tell Question: does she have shoes on? Answer: yes, flip flops Question: is there grass nearby? Answer: yes, everywhere Question: is it a sunny day? Answer: yes Question: are there trees? Answer: in the background there are trees Question: is the guitar new? Answer: i don’t think so 
        New Caption:
        """,
    
    "Assistant": """
        New Caption: a woman with pretty long hair sits alone on a grassy bench in a park on a sunny day, holding a guitar in her lap without a collection bucket, wearing flip flops, with trees in the background, with a slightly worn guitar
        """,
    
    "User_query": """
        [Caption]: {caption} 
        [Dialogue]: {dialogue} 
        New Caption:
        """,
}

FILTERING_PROMPT = {
    "System": """
        Your role is to filter the questions in the newly generated series of [New questions] that can be answered from [Dialogue] and [Description], or identify questions that are repeated within [Dialogue] and [Description]. 
        For those questions that meet these conditions, generate a list of their indices as [Invalid Question Indices]. The remaining questions that do not meet the conditions should have their indices listed as [Valid Question Indices].
        """,
    
    "User_example": """
        [New questions]: 0. What is the tool showcased in these images with the man in a yellow shirt and a biker performing a trick? 1. What color is the shirt worn by the man doing a trick at the skate park? 2. What apparel is the man wearing while performing the skateboard trick at a skate park? 3. What activity is the man excelling at in a specific type of park setting depicted in the candidates? 4. Based on the candidates and the information provided, what attire is the man wearing while performing the trick at a skate park?
        [Dialogue]: Question: what is the location of the jump trick being performed? Answer: a skate park
        [Description] a man is doing a trick on a skateboard
        Invalid Question Indices:
        Valid Question Indices:
        """,

    "Assistant": """
        Invalid Question Indices: 3
        Valid Question Indices: 0, 1, 2, 4
        """,
    
    "User_query": """
        [New questions]: {new_questions} 
        [Dialogue]: {dialogue} 
        [Description] {initial_description}
        Invalid Question Indices:
        Valid Question Indices:
        """,
}

# Table 19: 0-shot prompting example for LLM agent guided to answer the question according to the given context.
USER_SIMULATOR_PROMPT = {
    "System": """
        Answer the [Question] only according to the given [Target Image]. If you cannot determine the answer or there are no objects that are asked by the question in the context , answer "Uncertain". Reply begins with \'Answer:\'.
        """,
    
    "User_query": [
            {
                "type": "text",
                "text": """[Question]: {question}"""
            },
            {
                "type": "image_url",
                "image_url":{
                    "url": """{target_image_query}"""
                }
            },
        ]
}
