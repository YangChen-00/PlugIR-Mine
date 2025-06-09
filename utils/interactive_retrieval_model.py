import os
import json
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import BlipProcessor, BlipForImageTextRetrieval, BlipForConditionalGeneration
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
import utils.config as config

class InteractiveRetrievalModel:
    BLIP_ITR_MODEL_PATH = config.BLIP_ITR_MODEL_PATH
    BLIP_CAPTION_MODEL_PATH = config.BLIP_CAPTION_MODEL_PATH
    
    TOP_N_NUM = config.TOP_N_NUM
    CLUSTER_NUM = config.CLUSTER_NUM
    
    def __init__(self, 
                 img_root_path, 
                 img_embeddings_root_path,
                 model_name='BLIP'):
        assert model_name in ['BLIP', 'CLIP'], "Type must be either 'baseline' or 'cot'."
        
        self.img_root_path = img_root_path
        self.img_embeddings_root_path = img_embeddings_root_path
        self.model_name = model_name
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
                
        # Load Models
        if self.model_name == 'BLIP':
            self.retrieval_model = BlipForImageTextRetrieval.from_pretrained(self.BLIP_ITR_MODEL_PATH).to(self.device)
            self.caption_model = BlipForConditionalGeneration.from_pretrained(self.BLIP_CAPTION_MODEL_PATH).to(self.device)
            self.processor = BlipProcessor.from_pretrained(self.BLIP_ITR_MODEL_PATH)
        else:
            pass
            
        self.compute_image_embeddings()
    
    # Compute Image Embeddings
    def compute_image_embeddings(self):
        if not os.path.exists(self.img_root_path):
            print(f"Image root path {self.img_root_path} does not exist. Please check the directory.")
            return {}
        
        # Determine the dataset name from the image root path
        dataset_name = self.img_root_path.split(os.sep)[-1] if self.img_root_path.split(os.sep)[-1] != '' else self.img_root_path.split(os.sep)[-2]
        # Construct the path for image embeddings
        img_embeddings_path = os.path.join(self.img_embeddings_root_path, 
                                             f'{self.model_name}_{dataset_name}_img_embeddings.json') if self.img_embeddings_root_path else None
        
        # Check if the image embeddings file already exists, if so, load it
        if img_embeddings_path and os.path.exists(img_embeddings_path):
            print(f"Loading image embeddings from {img_embeddings_path}...")
            with open(img_embeddings_path, 'r') as f:
                img_embeddings = json.load(f)
            # Convert lists back to numpy arrays
            self.img_embeddings = {k: np.array(v) for k, v in img_embeddings.items()}
            return
        
        # If not, compute the embeddings
        img_info = {}
        for img_name in os.listdir(self.img_root_path):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_id = img_name.split('.')[0]
                img_path = os.path.join(self.img_root_path, img_name)
                img_info[img_id] = img_path
        
        # Check if any images were found
        if not img_info:
            print(f"No images found in {self.img_root_path}. Please check the directory.")
            return {}
        
        self.img_embeddings = {}
        
        # Compute embeddings for each image
        print(f"Computing image embeddings for {len(img_info)} images...")
        for i in tqdm(range(len(img_info))):
            img_id = list(img_info.keys())[i]
            img_path = img_info[img_id]

            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
                inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
                with torch.no_grad():
                    vision_outputs = self.retrieval_model.vision_model(pixel_values=inputs.pixel_values)
                    vision_outputs = self.retrieval_model.vision_proj(vision_outputs[0][:, 0, :])
                    
                    image_embed = F.normalize(vision_outputs, dim=-1)
                    image_embed = image_embed.cpu().numpy()  # [CLS] token
                self.img_embeddings[img_id] = image_embed
                
        # if image_embeddings_path is not None and not os.path.exists(image_embeddings_path):
        if img_embeddings_path:
            print(f"Saving image embeddings to {img_embeddings_path}...")
            with open(img_embeddings_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.img_embeddings.items()}, f)
    
    # @torch.no_grad()
    # def get_text_embedding(self, text):
    #     inputs = self.processor(text=text, return_tensors="pt", max_length=1024).to(self.device)
        
    #     text_encoder_module = None
    #     if hasattr(self.retrieval_model, 'text_encoder'):
    #         text_encoder_module = self.retrieval_model.text_encoder
    #     elif hasattr(self.retrieval_model, 'text_model'): 
    #             text_encoder_module = self.retrieval_model.text_model
    #     else:
    #         print(f"Error: No text_encoder/text_model on self.retrieval_model")

    #     text_outputs = text_encoder_module(
    #         input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, return_dict=True
    #     )
    #     text_cls_embedding = text_outputs.last_hidden_state[:, 0, :]
    #     text_embeds_projected = self.retrieval_model.text_proj(text_cls_embedding)

    #     if text_embeds_projected is None: 
    #         print(f"Warning: Projected text embedding is None")
        
    #     text_features_projected_normalized = F.normalize(text_embeds_projected, p=2, dim=-1)

    #     return text_features_projected_normalized.cpu().numpy()

    def get_text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            text_outputs = self.retrieval_model.text_encoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )
            text_outputs = self.retrieval_model.text_proj(text_outputs[0][:, 0, :])  # [CLS] token
            
            text_embed = F.normalize(text_outputs, dim=-1)
            text_embed = text_embed.cpu().numpy()  # [CLS] token

        return text_embed.tolist()

    # Get Top-n Similar Images
    def get_top_n_similar(self, query_caption):
        text_embed = self.get_text_embedding(query_caption)
        
        similarities = []
        for img_id, img_emb in self.img_embeddings.items():
            sim = np.dot(text_embed, img_emb.T)
            similarities.append((img_id, sim[0][0]))
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_n = similarities[:self.TOP_N_NUM]
        top_n_ids = [x[0] for x in top_n]
        top_n_embs = np.array([self.img_embeddings[x[0]] for x in top_n])
        top_n_sims = {x[0]: x[1] for x in top_n}
        
        return top_n_ids, top_n_embs, top_n_sims
    
    # Select Representatives
    def select_representatives(self, top_n_ids, top_n_embs, top_n_sims):
        kmeans = KMeans(n_clusters=self.CLUSTER_NUM)
        clusters = kmeans.fit_predict(top_n_embs.squeeze(axis=1))
        representatives = []
        for cluster in range(self.CLUSTER_NUM):
            cluster_idx = np.where(clusters == cluster)[0]
            if len(cluster_idx) > 0:
                sims_in_cluster = [top_n_sims[top_n_ids[i]] for i in cluster_idx]
                max_sim_idx_in_cluster = cluster_idx[np.argmax(sims_in_cluster)]
                rep_id = top_n_ids[max_sim_idx_in_cluster]
                representatives.append(rep_id)
        return representatives

    @torch.no_grad()
    def generate_caption(self, image_path=str):
        """ Generate a caption for an image.
        Args:
            image_path (str): Path to the image file.
            text_prompt (str): Optional text prompt for conditional captioning.
        Returns:
            str: Generated caption.
        """
        raw_image = Image.open(image_path).convert('RGB')

        # unconditional image captioning
        inputs = self.processor(raw_image, return_tensors="pt").to(self.device)

        out = self.caption_model.generate(**inputs)
        
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        return caption
        
    # Generate Captions
    def generate_representatives_captions(self, representative_ids):
        img_id_2_name = {} # id to name
        for name in os.listdir(self.img_root_path):
            img_id = name.split('.')[0]
            img_id_2_name[img_id] = name
        
        retrieved_images_and_captions = {}
        for img_id in representative_ids:
            # 找到img_id对应的图片文件名的文件后缀
            img_name = img_id_2_name.get(img_id)
            img_path = os.path.join(self.img_root_path, img_name)
            if os.path.exists(img_path):
                caption = self.generate_caption(img_path)
                # captions.append(caption)
                retrieved_images_and_captions[img_id] = caption
        return retrieved_images_and_captions

    # Main Function
    def retrieval_context_extraction(self, 
                                     query_caption):
        
        top_n_ids, top_n_embs, top_n_sims = self.get_top_n_similar(query_caption)
        rep_ids = self.select_representatives(top_n_ids, top_n_embs, top_n_sims)
        retrieved_images_and_captions = self.generate_representatives_captions(rep_ids)
        
        return retrieved_images_and_captions

# Example Usage
if __name__ == "__main__":
    
    img_root_path = '/home/chenyang/datasets/VisDial/VisualDialog_val2018/'
    img_embeddings_root_path = '/home/chenyang/projects/PlugIR_mine/embeddings/'
    query_caption = 'woman is playing tennis'
    
    retrieval_model = InteractiveRetrievalModel(img_root_path, img_embeddings_root_path, model_name='BLIP')
    
    captions = retrieval_model.retrieval_context_extraction(query_caption)
    print("Retrieval Context (Captions):")
    for img_id, caption in captions.items():
        print(f'Image ID: {img_id}   Caption: {caption}')