import os

import folder_paths
from folder_paths import add_model_folder_path
from comfy_api.latest import ComfyExtension


def _register_llm_folder():
    llm = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "models", "LLM"))
    if os.path.isdir(llm):
        add_model_folder_path("llm", llm)
        folder_paths.folder_names_and_paths["llm"][1].add(".safetensors")


_register_llm_folder()

from .nodes import JEVModelLoader, JEVDecide, JEVChat  # noqa: E402


class JEVExtension(ComfyExtension):
    async def get_node_list(self):
        return [JEVModelLoader, JEVDecide, JEVChat]


async def comfy_entrypoint():
    return JEVExtension()
