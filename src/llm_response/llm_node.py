import os
import re
import torch
from rclpy.node import Node
from std_msgs.msg import String
from custom_msgs.msg import FatigueFeatures  # Replace with your actual package/msg
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from model_wrapper_with_mlp_adapter import FeaturePrefixAdapter
from faiss_vd import runtime_add, retrieve_similar_vectors


class LLMNode(Node):
    def __init__(self):
        super().__init__('fatigue_intervention_node')

        # Parameters/config
        self.MODEL_ID = "meta-llama/Llama-2-7b-hf"
        self.MODEL_DIR = "/content/drive/MyDrive/llm/LLM-based-Agent-for-Driver-Sleepiness-Detection-and-Mitigation-in-Automotive-Systems/llm_and_fatigue_handling/llama_prefix_final_model"
        self.HF_TOKEN = "your hugging face access token here"  # Declare token explicitly
        self.FEATURE_DIM = 9
        self.EMBEDDING_DIM = 4096
        self.PREFIX_TOKEN_COUNT = 5
        self.MAX_LENGTH = 256
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Load models/tokenizer once at node init
        try:
            # === Tokenizer ===
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID, token=self.HF_TOKEN)
            self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"

            # === Quantization Config ===
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )

            # === Base Model ===
            base_model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID,
                device_map="auto",
                trust_remote_code=True,
                token=self.HF_TOKEN,
                quantization_config=self.bnb_config,
                low_cpu_mem_usage=True
            )

            # === LoRA Adapter ===
            self.llama_model = PeftModel.from_pretrained(
                base_model,
                self.MODEL_DIR,
                device_map="auto",
                use_auth_token=False
            )
            self.llama_model.eval()

            # === Prefix Adapter ===
            self.adapter = FeaturePrefixAdapter(
                input_dim=self.FEATURE_DIM,
                hidden_dim=256,
                output_dim=self.EMBEDDING_DIM,
                num_tokens=self.PREFIX_TOKEN_COUNT
            )
            self.adapter.load_state_dict(torch.load(os.path.join(self.MODEL_DIR, "prefix_adapter.pth")))
            target_dtype = next(self.llama_model.parameters()).dtype
            self.adapter = self.adapter.to(dtype=target_dtype, device=self.DEVICE)
            self.adapter.eval()

        except Exception as e:
            self.get_logger().error(f"Error initializing model: {e}")
            raise

        # ROS2 subscriber and publisher
        self.subscription = self.create_subscription(
            FatigueFeatures,
            '/driver_fatigue_features',
            self.listener_callback,
            10
        )
        self.publisher_ = self.create_publisher(String, '/driver_intervention', 10)

        self.get_logger().info("Fatigue Intervention Node Initialized")

    def listener_callback(self, msg):
        features = list(msg.features)
        fatigue_levels = list(msg.fatigue_levels)

        intervention_text = self.generate_intervention(features, fatigue_levels)

        intervention_msg = String()
        intervention_msg.data = intervention_text
        self.publisher_.publish(intervention_msg)

        self.get_logger().info(f'Published intervention: {intervention_text}')

    def is_valid_intervention(self, output: str) -> bool:
        pattern = r"^Fan: Level [123]\nMusic: (On|Off)\nVibration: (On|Off)\nReason: .+"
        return bool(re.match(pattern, output.strip()))

    def generate_intervention(self, features, fatigue_levels):
        feature_tensor = torch.tensor([features], dtype=next(self.llama_model.parameters()).dtype).to(self.DEVICE)
        prefix_embeddings = self.adapter(feature_tensor)
        token_matrix = prefix_embeddings.squeeze(0).detach().cpu().numpy()

        # FAISS Retrieval
        results = retrieve_similar_vectors(token_matrix, k=3)
        retrieved_interventions = [
            meta.get("intervention") for _, meta, _ in results
            if meta.get("intervention") and meta.get("intervention").strip().lower() not in {"", "none", "driver alert"}
        ]
        context = (
            "Previously suggested interventions for similar scenarios: "
            + "; ".join(retrieved_interventions) + ". "
            if retrieved_interventions else ""
        )

        prompt = f"""{context}
          You are an intelligent in-cabin assistant.
          
          Fatigue levels:
          - Camera: {fatigue_levels[0]}
          - Steering: {fatigue_levels[1]}
          - Lane: {fatigue_levels[2]}
          
          Based on the above driver state and past examples, suggest an intervention to keep the driver alert.
          
          ⚠️ IMPORTANT: You must output in this fixed format — no extra text.
          
          Fan: Level X      ← X is a number like 1, 2, or 3  
          Music: On/Off  
          Vibration: On/Off  
          Reason: 
          
          Example output:
          Fan: Level 2  
          Music: On  
          Vibration: Off  
          Reason: High blink rate and PERCLOS indicate moderate drowsiness.
          
          Now, provide your intervention:""".strip()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.MAX_LENGTH - self.PREFIX_TOKEN_COUNT,
            truncation=True,
            padding="max_length"
        )
        input_ids = inputs["input_ids"].to(self.DEVICE)
        attention_mask = inputs["attention_mask"].to(self.DEVICE)

        input_embeddings = self.llama_model.base_model.get_input_embeddings()(input_ids)
        prefix_embeddings = prefix_embeddings.to(dtype=input_embeddings.dtype)
        combined_embeddings = torch.cat([prefix_embeddings, input_embeddings], dim=1)

        prefix_attention_mask = torch.ones(1, self.PREFIX_TOKEN_COUNT, dtype=torch.long).to(self.DEVICE)
        extended_attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

        with torch.no_grad():
            output = self.llama_model.generate(
                inputs_embeds=combined_embeddings,
                attention_mask=extended_attention_mask,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(output[0, self.PREFIX_TOKEN_COUNT:], skip_special_tokens=True).strip()

        if self.is_valid_intervention(response):
            runtime_add(token_matrix, intervention=response)
        else:
            self.get_logger().warn("Generated response did not match required format. Skipping FAISS update.")

        return response


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = LLMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
