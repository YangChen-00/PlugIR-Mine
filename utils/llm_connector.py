from openai import OpenAI
import utils.config as config

class LLM_Connector:
    # Table 17: 1-shot prompting example for LLM questioner of baseline.
    QUESTIONER_BASELINE_PROMPT = config.QUESTIONER_BASELINE_PROMPT
    
    # Table 18: 1-shot prompting example for LLM questioner utilizing CoT and the additional context from the set of retrieval candidates.
    QUESTIONER_COT_PROMPT = config.QUESTIONER_COT_PROMPT

    # Table 20: 1-shot prompting example for LLM to reformulate the dialogue context.
    REFORMULATOR_PROMPT = config.REFORMULATOR_PROMPT
    
    # Table 19: 0-shot prompting example for LLM agent guided to answer the question according to the given context.
    USER_SIMULATOR_PROMPT = config.USER_SIMULATOR_PROMPT
    
    FILTERING_PROMPT = config.FILTERING_PROMPT
    
    # Hyperparameters for each module
    QUESTIONER_HYPERPARAMS = config.QUESTIONER_HYPERPARAMS
    
    REFORMULATOR_HYPERPARAMS = config.REFORMULATOR_HYPERPARAMS
    
    FILTERING_HYPERPARAMS = config.FILTERING_HYPERPARAMS
    
    
    def __init__(self, api_key: str, base_url: str, model_name: str = "grok-3"):
        """
        Initializes the QuestionGenerator with OpenAI API credentials and model.

        Args:
            api_key (str): Your OpenAI API key.
            base_url (str): The base URL for the OpenAI API.
            model_name (str): The name of the model to use (e.g., "deepseek-chat", "gpt-3.5-turbo").
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        
    def questioner(self,
                   retrieval_candidates: list,
                   initial_description: str,
                   dialogues: list,
                   mode = 'baseline') -> str:
        """
        Generates a question based on the provided retrieval candidates, initial description, dialogues, and type of question prompt.
        Args:
            retrieval_candidates (list): [{'id', 'path', 'caption'}]. A list of dictionaries containing retrieval candidates with their captions.
            initial_description (str): The initial description of the target image.
            dialogues (list): [{'question', 'answer'}]. A list of dictionaries containing question-answer pairs that provide additional context.
            type (str): The type of question generation ('baseline' or 'cot').
        Returns:
            list: The generated question list.
        """
        
        assert mode in ['baseline', 'cot'], "Type must be either 'baseline' or 'cot'."
        
        prompt = ""
        
        dialogues_str = ' '.join([f"Question: {dialogue['question']} Answer: {dialogue['answer']}" for dialogue in dialogues])
        
        if mode == 'baseline':
            prompt = self.QUESTIONER_BASELINE_PROMPT
            prompt['User_query'] = prompt['User_query'].format(
                initial_description=initial_description,
                dialogues=dialogues_str
            )
        elif mode == 'cot':
            retrieval_candidates_captions = ' '.join([f"{i}. {candidate['caption']}" for i, candidate in enumerate(retrieval_candidates)])
            
            prompt = self.QUESTIONER_COT_PROMPT
            prompt['User_query'] = prompt['User_query'].format(
                retrieval_candidates=retrieval_candidates_captions,
                initial_description=initial_description,
                dialogues=dialogues_str
            )
            
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt["System"]},
                    {"role": "user", "content": prompt["User_example"]},
                    {"role": "assistant", "content": prompt["Assistant"]},
                    {"role": "user", "content": prompt["User_query"]}
                ],
                max_tokens=self.QUESTIONER_HYPERPARAMS['max_tokens'], # Consider making this configurable if needed
                temperature=self.QUESTIONER_HYPERPARAMS['temperature'],
                n=self.QUESTIONER_HYPERPARAMS['num_questions']
            )
            
            new_questions = []
            for choice in response.choices:
                full_response = choice.message.content.strip()
            
                question = None
                
                if "Question:" in full_response:
                    question_part = full_response.split("Question:", 1)[1] # Split only on the first occurrence
                    question = question_part.split("\n")[0].strip()
                    
                new_questions.append(question)
                    
            return new_questions
        except Exception as e:
            print(f"Error generating question: {e}")
            return None
        
    def reformulator(self, 
                     caption: str, 
                     dialogues: list) -> str:
        """
        Reformulates the caption based on the dialogues provided.
        Args:
            caption (str): The original caption of the image.
            dialogues (list): [{'question', 'answer'}]. A list of dictionaries containing question-answer pairs that provide additional context.
        Returns:
            str: The reformulated caption.
        """
        
        dialogues_str = ' '.join([f"Question: {dialogue['question']} Answer: {dialogue['answer']}" for dialogue in dialogues])
        prompt = self.REFORMULATOR_PROMPT
        prompt['User_query'] = prompt['User_query'].format(
            caption=caption,
            dialogue=dialogues_str
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt["System"]},
                    {"role": "user", "content": prompt["User_example"]},
                    {"role": "assistant", "content": prompt["Assistant"]},
                    {"role": "user", "content": prompt["User_query"]}
                ],
                max_tokens=self.REFORMULATOR_HYPERPARAMS['max_tokens'], # Consider making this configurable if needed
                temperature=self.REFORMULATOR_HYPERPARAMS['temperature'],
                n=self.REFORMULATOR_HYPERPARAMS['n']
            )
            full_response = response.choices[0].message.content.strip()
            
            new_caption = None
            
            if "New Caption:" in full_response:
                new_caption_part = full_response.split("New Caption:", 1)[1] # Split only on the first occurrence
                new_caption = new_caption_part.split("\n")[0].strip()
                
            return new_caption
        except Exception as e:
            print(f"Error generating question: {e}")
            return None
    
    def filtering(self,
                  new_questions: list,
                  initial_description: str,
                  dialogues: list) -> str:
        if len(new_questions) == 0:
            return None
        
        if len(new_questions) == 1:
            return new_questions
        
        dialogues_str = ' '.join([f"Question: {dialogue['question']} Answer: {dialogue['answer']}" for dialogue in dialogues])
        new_questions_str = ' '.join([f"{i}. {new_questions[i]}" for i in range(len(new_questions))])
        
        prompt = self.FILTERING_PROMPT
        prompt['User_query'] = prompt['User_query'].format(
            new_questions=new_questions_str,
            dialogue=dialogues_str,
            initial_description=initial_description
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt["System"]},
                    {"role": "user", "content": prompt["User_example"]},
                    {"role": "assistant", "content": prompt["Assistant"]},
                    {"role": "user", "content": prompt["User_query"]}
                ],
                max_tokens=self.FILTERING_HYPERPARAMS['max_tokens'], # Consider making this configurable if needed
                temperature=self.FILTERING_HYPERPARAMS['temperature'],
                n=self.FILTERING_HYPERPARAMS['n']
            )
            full_response = response.choices[0].message.content.strip()
            
            """
            Invalid Question Indices: 0
            Valid Question Indices: 1, 2
            """
            
            invalid_question_indices = []
            valid_question_indices = []
            
            if "Invalid Question Indices:" in full_response:
                invalid_part = full_response.split("Invalid Question Indices:", 1)[1]
                invalid_part = invalid_part.split("Valid Question Indices:")[0].strip()
                invalid_question_indices = [int(i.strip()) for i in invalid_part.split(',') if i.strip().isdigit()]
            if "Valid Question Indices:" in full_response:
                valid_part = full_response.split("Valid Question Indices:", 1)[1].strip()
                valid_question_indices = [int(i.strip()) for i in valid_part.split(',') if i.strip().isdigit()]
            
            invalid_question = [new_questions[i] for i in invalid_question_indices]
            valid_questions = [new_questions[i] for i in valid_question_indices]
                
            return invalid_question, valid_questions
        except Exception as e:
            print(f"Error generating question: {e}")
            return None

def test_questioner(llm_connector_instance, mode='baseline'):
    retrieval_candidates = [
        {"id": 0, "path": "path/to/image0.jpg", "caption": "man in yellow shirt"},
        {"id": 1, "path": "path/to/image1.jpg", "caption": "a boy in a skateboard park"},
        {"id": 2, "path": "path/to/image2.jpg", "caption": "the biker is performing a trick"},
        # {"id": 3, "path": "path/to/image3.jpg", "caption": "a man in a green hat doing half-pipe with a skateboard"},
        # {"id": 4, "path": "path/to/image4.jpg", "caption": "a skateboarding man catches the air in the midst of a trick"},
    ]
    
    initial_description = 'a man is doing a trick on a skateboard'
    
    dialogues = [
        # {"question": "what type of trick is the man performing on the skateboard?", "answer": "a jump"},
        {"question": "what is the location of the jump trick being performed?", "answer": "a skate park"}   
    ]
    
    question = llm_connector_instance.questioner(
        retrieval_candidates=retrieval_candidates,
        initial_description=initial_description,
        dialogues=dialogues,
        mode=mode
    )
    
    print(f"Generated Question: {question}")
    
    return question
    
def test_reformulator(llm_connector_instance):
    caption = 'a woman sits on a bench holding a guitar in her lap '
    
    dialogues = [
        {"question": "is this in a park?", "answer": "yes, i believe it is"},
        {"question": "are there others around?", "answer": "no, she is alone"},
        {"question": "does she have a collection bucket?", "answer": "no"},
        {"question": "is her hair long?", "answer": "yes, pretty long"},
        {"question": "is she wearing a dress?", "answer": "i don’t think so, hard to tell"},
        {"question": "does she have shoes on?", "answer": "yes, flip flops"},
        {"question": "is there grass nearby?", "answer": "yes, everywhere"},
        {"question": "is it a sunny day?", "answer": "yes"},
        {"question": "are there trees?", "answer": "in the background there are trees"},
        {"question": "is the guitar new?", "answer": "i don’t think so"}
    ]
    
    new_caption = llm_connector_instance.reformulator(
        caption=caption,
        dialogues=dialogues
    )
    
    print(f"Reformulated Caption: {new_caption}")
    
    return new_caption
         

def test_filtering(llm_connector_instance):
    dialogues = [
        {"question": "What object is the woman holding in her hand?", "answer": "tennis racket"},
    ]
    
    # [New questions]: 0. What object is the woman holding? 1. What clothes do women wear? 2. What are women doing?
    new_questions = [
        "What object is the woman holding?",
        "What clothes do women wear?",
        "What are women doing?",
    ]
    
    initial_description = "A woman outside"
    
    # Filtering the new questions based on the initial description and dialogues
    invalid_questions, valid_questions = llm_connector_instance.filtering(
        new_questions=new_questions,
        initial_description=initial_description,
        dialogues=dialogues)
    
    print(f"Invalid Questions: {invalid_questions}")
    print(f"Valid Questions: {valid_questions}")
    
    return invalid_questions, valid_questions
    
            
if __name__ == "__main__":
    # Example
    LLM_API_KEY = "sk-DsWTD0DPOkqL2RYKXB8PDSEehWwdsncNimaoALFKDlSLkGCY"
    LLM_BASE_URL = "https://api.aiearth.dev/v1"
    LLM_MODEL_NAME = 'gpt-3.5-turbo'

    llm_connector_instance = LLM_Connector(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, model_name=LLM_MODEL_NAME)
    
    retrieval_candidates = [
        {"id": 0, "path": "path/to/image0.jpg", "caption": "man in yellow shirt"},
        {"id": 1, "path": "path/to/image1.jpg", "caption": "a boy in a skateboard park"},
        {"id": 2, "path": "path/to/image2.jpg", "caption": "the biker is performing a trick"},
        # {"id": 3, "path": "path/to/image3.jpg", "caption": "a man in a green hat doing half-pipe with a skateboard"},
        # {"id": 4, "path": "path/to/image4.jpg", "caption": "a skateboarding man catches the air in the midst of a trick"},
    ]
    
    initial_description = 'a man is doing a trick on a skateboard'
    
    dialogues = [
        # {"question": "what type of trick is the man performing on the skateboard?", "answer": "a jump"},
        {"question": "what is the location of the jump trick being performed?", "answer": "a skate park"}   
    ]
    
    new_questions = llm_connector_instance.questioner(
        retrieval_candidates=retrieval_candidates,
        initial_description=initial_description,
        dialogues=dialogues,
        mode='cot'
    )
    
    print(f"Generated Question: {new_questions}")
    
    # Filtering the new questions based on the initial description and dialogues
    invalid_questions, valid_questions = llm_connector_instance.filtering(
        new_questions=new_questions,
        initial_description=initial_description,
        dialogues=dialogues)
    
    print(f"Invalid Questions: {invalid_questions}")
    print(f"Valid Questions: {valid_questions}")

    
    