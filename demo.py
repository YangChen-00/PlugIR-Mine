from utils.llm_connector import LLM_Connector
from utils.interactive_retrieval_model import InteractiveRetrievalModel
import utils.config as config

class InteractiveRetrievalDemo:
    
    # LLM Connector configuration
    LLM_API_KEY = config.LLM_API_KEY
    LLM_BASE_URL = config.LLM_BASE_URL
    LLM_MODEL_NAME = config.LLM_MODEL_NAME

    # Corpus path
    IMG_ROOT_PATH = config.IMG_ROOT_PATH
    # Corpus embedding root path
    IMG_EMBEDDING_ROOT_PATH = config.IMG_EMBEDDING_ROOT_PATH

    MEMORY_STORAGE = {}
    
    def __init__(self):
        self.initialize_models_and_data()

    def initialize_models_and_data(self):
        print('\n' + "-" * 30 + "Initializing models and data..." + "-" * 30)
        self.llm_connector_instance = LLM_Connector(api_key=self.LLM_API_KEY, 
                                               base_url=self.LLM_BASE_URL, 
                                               model_name=self.LLM_MODEL_NAME)
        self.interactive_retrieval_model = InteractiveRetrievalModel(self.IMG_ROOT_PATH, self.IMG_EMBEDDING_ROOT_PATH, model_name='BLIP')
        
    def retrieve(self, query):
        retrieved_images_and_captions = self.interactive_retrieval_model.retrieval_context_extraction(query)
        
        self.MEMORY_STORAGE['retrieved_images_and_captions'] = retrieved_images_and_captions
        
        print("Retrieval Context (Captions):")
        for img_id, caption in retrieved_images_and_captions.items():
            print(f'Image ID: {img_id}   Caption: {caption}')
            
        return retrieved_images_and_captions
    
    def questioner_and_filtering(self, round):
        print('\n' + "-" * 30 + f"Round {round} (Generate LLM question)" + "-" * 30)
        new_questions = self.llm_connector_instance.questioner(self.MEMORY_STORAGE['retrieved_images_and_captions'],
                                                                    self.MEMORY_STORAGE['initial_query'],
                                                                    self.MEMORY_STORAGE['dialogues'])
        
        print("+ Generated LLM question:")
        for i, question in enumerate(new_questions):
            print(f"{i}. {question}")
        
        invalid_questions, valid_questions = self.llm_connector_instance.filtering(new_questions=new_questions,
                                                                              initial_description=self.MEMORY_STORAGE['initial_query'],
                                                                              dialogues=self.MEMORY_STORAGE['dialogues'])
        
        print("+ Valid questions:")
        for i, question in enumerate(valid_questions):
            print(f"{i}. {question}")
            
        print("+ Invalid questions:")
        for i, question in enumerate(invalid_questions):
            print(f"{i}. {question}")
        
        if not valid_questions:
            print("+ No valid questions generated. Exiting the dialogue.")
            exit()
        
        chosen_question = valid_questions[0]
        
        print(f"+ Chosen question: {chosen_question}")
        
        return chosen_question
    
    def first_round_retrieval(self):
        print('\n' + "-" * 30 + "Round 0 (User query)" + "-" * 30)
        initial_query = input("+ Please enter the initial query: ")
        self.MEMORY_STORAGE['initial_query'] = initial_query
        self.MEMORY_STORAGE['last_query'] = initial_query
        self.MEMORY_STORAGE['dialogues'] = []

        print('\n' + "-" * 30 + "Round 0 (Retrieve images)" + "-" * 30)
        self.retrieve(initial_query)
        

    def interactive_retrieval_loop(self):
        chosen_question = self.questioner_and_filtering(round=1)
        
        round = 2
        while True:
            answer = input("+ Please enter the answer (\'exit\' to quit):  ")
            
            if answer.lower() == 'exit':
                print("Exiting the dialogue.")
                break
            
            self.MEMORY_STORAGE['dialogues'].append({
                'question': chosen_question,
                'answer': answer
            })
            
            reformulated_query = self.llm_connector_instance.reformulator(self.MEMORY_STORAGE['last_query'],
                                                                            self.MEMORY_STORAGE['dialogues'])
            print("+ Reformulated response:", reformulated_query)
            self.MEMORY_STORAGE['last_query'] = reformulated_query
            
            self.retrieve(reformulated_query)
            
            print('\n' + "-" * 30 + f"Round {round} (Generate LLM question)" + "-" * 30)
            
            chosen_question = self.questioner_and_filtering(round=round)
            
            round += 1
    
    def main(self):
        self.first_round_retrieval()
        
        self.interactive_retrieval_loop()
    
    
if __name__ == "__main__":
    demo = InteractiveRetrievalDemo()
    demo.main()
    
    # You can add more methods to continue the interactive retrieval loop as needed.
    # For example, you can implement a method to handle subsequent rounds of retrieval and interaction.
