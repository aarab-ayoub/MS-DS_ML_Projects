import os
import time
import psutil
from pathlib import Path
from typing import Optional
from llama_cpp import Llama # bach ndiro installation :=> pip install llama-cpp-python==0.3.9

"""
    Aussi 5ass cpp tkun déja installé

"""
def get_ram_usage_gb() -> float:
    """RAM du processus courant en Go."""
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def load_model(
    model_path: str,
    n_threads : int = 2,
    n_ctx     : int = 32768,
) -> Llama:
    """
    Charge le modele GGUF avec configuration optimisee CPU.

    Args:
        model_path : chemin vers le fichier .gguf
        n_threads  : nombre de threads CPU 
        n_ctx      : taille du contexte 

    Returns:
        Llama : instance prete a l inference
    """
    return Llama(
        model_path   = model_path,
        n_threads    = n_threads,
        n_ctx        = n_ctx,
        n_batch      = 512,
        n_gpu_layers = 0,       # CPU pur
        verbose      = False,
    )


class VLEncoder:
    """
    Encodeur Vision-Langage pour MMedAgent.

    """

    def __init__(self, llm: Llama, image_size: int = 448):
        self.llm        = llm
        self.image_size = image_size
        self._last_bm   = {}

    def encode(
        self,
        image_path : str,
        question   : str,
        max_tokens : int = 256,
        temperature: float = 0.0,
        system_msg : Optional[str] = None,
        rag_context: Optional[str] = None,
    ) -> str:
        """
        Genere une reponse medicale a partir d une image et d une question.

        Args:
            image_path  : chemin vers l image (JPEG/PNG)
            question    : question clinique
            max_tokens  : tokens maximum en sortie
            temperature : 0.0 = deterministe
            system_msg  : message systeme optionnel
            rag_context : contexte RAG optionnel 

        Returns:
            str : reponse du modele
        """
        prompt     = self._build_prompt(image_path, question, system_msg, rag_context)
        ram_before = get_ram_usage_gb()
        t_start    = time.time()

        output = self.llm(
            prompt,
            max_tokens  = max_tokens,
            temperature = temperature,
            stop        = ["<|im_end|>", "<|endoftext|>"],
            echo        = False,
        )

        latency_s = time.time() - t_start
        ram_after = get_ram_usage_gb()
        n_tokens  = output["usage"]["completion_tokens"]

        self._last_bm = {
            "latency_s"       : round(latency_s, 3),
            "tokens_per_sec"  : round(n_tokens / latency_s, 2) if latency_s > 0 else 0.0,
            "tokens_generated": n_tokens,
            "prompt_tokens"   : output["usage"]["prompt_tokens"],
            "ram_before_gb"   : round(ram_before, 3),
            "ram_after_gb"    : round(ram_after,  3),
            "ram_delta_gb"    : round(ram_after - ram_before, 3),
        }
        return output["choices"][0]["text"].strip()

    def get_last_benchmark(self) -> dict:
        """Retourne les metriques de la derniere inference."""
        return self._last_bm

    def warmup(self) -> None:
        """Inference de chauffe pour initialiser les caches."""
        self.llm(
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
            max_tokens=5, temperature=0.0, echo=False
        )

    def _build_prompt(
        self,
        image_path : str,
        question   : str,
        system_msg : Optional[str] = None,
        rag_context: Optional[str] = None,
    ) -> str:
        """
        Construit le prompt au format ChatML Qwen2.5-VL.
        """
        if system_msg is None:
            system_msg = (
                "You are a medical AI assistant specialized in visual "
                "diagnosis. Answer concisely and accurately."
            )

        parts = [f"<|im_start|>system\n{system_msg}<|im_end|>\n"]

        user_content = ""

        img_name      = Path(image_path).stem
        user_content += f"[Medical image: {img_name}]\n"

        if rag_context:
            user_content += f"Reference cases:\n{rag_context}\n\n"

        user_content += question

        parts.append(f"<|im_start|>user\n{user_content}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)