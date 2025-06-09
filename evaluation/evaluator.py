from utils.llm_connector import LLM_Connector
from utils.mllm_connector import MLLM_Connector
from utils.interactive_retrieval_model import InteractiveRetrievalModel
import utils.config as config
import os

class InteractiveRetrievalEvaluator:
    
    # LLM Connector configuration
    LLM_API_KEY = config.LLM_API_KEY
    LLM_BASE_URL = config.LLM_BASE_URL
    LLM_MODEL_NAME = config.LLM_MODEL_NAME

    # Corpus embedding root path
    IMG_EMBEDDING_ROOT_PATH = config.IMG_EMBEDDING_ROOT_PATH

    MEMORY_STORAGE = {}
    
    def __init__(self, img_root_path):
        self.IMG_ROOT_PATH = img_root_path
        
        self.initialize_models_and_data()
        
        self.img_id_2_name = {} # id to name
        for name in os.listdir(self.IMG_ROOT_PATH):
            img_id = name.split('.')[0]
            self.img_id_2_name[img_id] = name

    def initialize_models_and_data(self):
        # print('\n' + "-" * 30 + "Initializing models and data..." + "-" * 30)
        self.llm_connector_instance = LLM_Connector(api_key=self.LLM_API_KEY, 
                                               base_url=self.LLM_BASE_URL, 
                                               model_name=self.LLM_MODEL_NAME)
        self.mllm_connector_instance = MLLM_Connector(api_key=config.MLLM_API_KEY,
                                                  base_url=config.MLLM_BASE_URL, 
                                                  model_name=config.MLLM_MODEL_NAME)
        self.interactive_retrieval_model = InteractiveRetrievalModel(self.IMG_ROOT_PATH, self.IMG_EMBEDDING_ROOT_PATH, model_name='BLIP')
        
    def retrieve(self, query):
        retrieved_images_and_captions = self.interactive_retrieval_model.retrieval_context_extraction(query)
        
        self.MEMORY_STORAGE['retrieved_images_and_captions'] = retrieved_images_and_captions
        
        # print("Retrieval Context (Captions):")
        # for img_id, caption in retrieved_images_and_captions.items():
            # print(f'Image ID: {img_id}   Caption: {caption}')
            
        return retrieved_images_and_captions
    
    def questioner_and_filtering(self, round):
        # print('\n' + "-" * 30 + f"Round {round} (Generate LLM question)" + "-" * 30)
        new_questions = self.llm_connector_instance.questioner(self.MEMORY_STORAGE['retrieved_images_and_captions'],
                                                                    self.MEMORY_STORAGE['initial_query'],
                                                                    self.MEMORY_STORAGE['dialogues'])
        
        # print("+ Generated LLM question:")
        # for i, question in enumerate(new_questions):
        #     print(f"{i}. {question}")
        
        # invalid_questions, valid_questions = self.llm_connector_instance.filtering(new_questions=new_questions,
        #                                                                       initial_description=self.MEMORY_STORAGE['initial_query'],
        #                                                                       dialogues=self.MEMORY_STORAGE['dialogues'])
        
        # print("+ Valid questions:")
        # for i, question in enumerate(valid_questions):
        #     print(f"{i}. {question}")
            
        # print("+ Invalid questions:")
        # for i, question in enumerate(invalid_questions):
        #     print(f"{i}. {question}")
        
        # if not valid_questions:
        #     print("+ No valid questions generated. Exiting the dialogue.")
        #     exit()
        
        chosen_question = new_questions[0]
        
        # print(f"+ Chosen question: {chosen_question}")
        
        return chosen_question
    
    def check_retrieval_rank(self, target_img_relative_path, retrieved_images_and_captions):
        target_img_id = target_img_relative_path.split('.')[0]
        
        if target_img_id in retrieved_images_and_captions:
            retrieval_rank = list(retrieved_images_and_captions.keys()).index(target_img_id) + 1
            # print(f"Retrieval rank of the target image ({target_img_relative_path}): {retrieval_rank}")
        else:
            # print(f"Target image ({target_img_relative_path}) not found in the retrieved images.")
            retrieval_rank = -1
            
        return retrieval_rank
    
    def interactive_retrieval(self, target_img_relative_path, initial_query):
        
        retrieval_ranks = []
        
        for round in range(10):
            if round == 0:
                self.MEMORY_STORAGE['initial_query'] = initial_query
                self.MEMORY_STORAGE['last_query'] = initial_query
                self.MEMORY_STORAGE['dialogues'] = []

                # print('\n' + "-" * 30 + "Round 0 (Retrieve images)" + "-" * 30)
                retrieved_images_and_captions = self.retrieve(initial_query)
                self.MEMORY_STORAGE['retrieved_images_and_captions'] = retrieved_images_and_captions
                
                retrieval_ranks.append(self.check_retrieval_rank(target_img_relative_path, retrieved_images_and_captions))
        
                chosen_question = self.questioner_and_filtering(round=round)
            else:
                answer = self.mllm_connector_instance.user_simulator(os.path.join(self.IMG_ROOT_PATH, target_img_relative_path),
                                                                    chosen_question)

                self.MEMORY_STORAGE['dialogues'].append({
                    'question': chosen_question,
                    'answer': answer
                })
                
                reformulated_query = self.llm_connector_instance.reformulator(self.MEMORY_STORAGE['last_query'],
                                                                                self.MEMORY_STORAGE['dialogues'])
                # print("+ Reformulated response:", reformulated_query)
                self.MEMORY_STORAGE['last_query'] = reformulated_query
                
                retrieved_images_and_captions = self.retrieve(reformulated_query)
                self.MEMORY_STORAGE['retrieved_images_and_captions'] = retrieved_images_and_captions
                
                retrieval_ranks.append(self.check_retrieval_rank(target_img_relative_path, retrieved_images_and_captions))
                
                # print('\n' + "-" * 30 + f"Round {round} (Generate LLM question)" + "-" * 30)
                
                chosen_question = self.questioner_and_filtering(round=round)
                
        return retrieval_ranks
    
    def evaluate(self, target_img_relative_path, initial_query):
        return self.interactive_retrieval(target_img_relative_path, initial_query)
        
if __name__ == "__main__":
    img_root_path = "/home/chenyang/datasets/unlabeled2017"
    evaluator = InteractiveRetrievalEvaluator(img_root_path)
    
    # Example usage
    target_img_relative_path = "000000284024.jpg"  # Replace with actual image path
    initial_query = 'a woman sits on a bench holding a guitar in her lap'  # Replace with actual initial query
    
    retrieval_ranks = evaluator.evaluate(target_img_relative_path, initial_query)
    
    print(f"Retrieval ranks for the target image ({target_img_relative_path}): {retrieval_ranks}")