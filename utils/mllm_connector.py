from openai import OpenAI
import utils.config as config
import base64

class MLLM_Connector:
    USER_SIMULATOR_PROMPT = config.USER_SIMULATOR_PROMPT
    
    USER_SIMULATOR_HYPERPARAMS = config.USER_SIMULATOR_HYPERPARAMS

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

    def convert_image_2_base64(self, image_path: str) -> str:
        """
        Converts an image file to a base64 encoded string.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: Base64 encoded string of the image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def user_simulator(self, target_image_path, question):
        # 原图片转base64
        target_image_base64 = self.convert_image_2_base64(target_image_path)
        
        target_image_query = f"data:image/jpeg;base64,{target_image_base64}"
        
        prompt = self.USER_SIMULATOR_PROMPT
        prompt["User_query"][0]['text'] = \
            prompt["User_query"][0]['text'].format(question=question)
        prompt["User_query"][1]['image_url']['url'] = \
            prompt["User_query"][1]['image_url']['url'].format(target_image_query=target_image_query)
        
        try:
            #提交信息
            response = self.client.chat.completions.create(
                model=self.model_name,#选择模型
                messages=[
                    {"role": "system", "content": prompt["System"]},
                    {"role": "user", "content": prompt["User_query"]}
                ],
                max_tokens=self.USER_SIMULATOR_HYPERPARAMS['max_tokens'], # Consider making this configurable if needed
                temperature=self.USER_SIMULATOR_HYPERPARAMS['temperature'],
                n=self.USER_SIMULATOR_HYPERPARAMS['n']
            )
            full_response = response.choices[0].message.content.strip()
            
            answer = None
            
            if "Answer:" in full_response:
                answer_part = full_response.split("Answer:", 1)[1] # Split only on the first occurrence
                answer = answer_part.split("\n")[0].strip()
                
            return answer
        except Exception as e:
            print(f"Error generating question: {e}")
            return None

if __name__ == "__main__":
    mllm_connector_instance = MLLM_Connector(api_key=config.MLLM_API_KEY,
                                                base_url=config.MLLM_BASE_URL,
                                                model_name=config.MLLM_MODEL_NAME)
    target_image_path = '123.jpg'  # Replace with your image path
    question = "What is the main color of this image?"  # Replace with your question
    answer = mllm_connector_instance.user_simulator(target_image_path, question)
    
    print(f"Generated answer: {answer}")