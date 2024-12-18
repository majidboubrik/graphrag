#import dku_graphrag.index.monkey_patch_dataiku  # This sets up the custom LLMs

import logging

import time

from graphrag.api import build_index

from graphrag.logger.factory import LoggerFactory, LoggerType

class DataikuGraphragIndexBuilder:
    """
    A class to handle building and updating a GraphRAG index, similar to the CLI commands,
    but without sys.exit() and signal handling. Ideal for use inside a Dataiku Python recipe.
    """
    
    def __init__(self, chat_completion_llm_id, embedding_llm_id):
        self.logger = logging.getLogger(__name__)
        self.logger_type = LoggerType.RICH
        self.chat_completion_llm_id = chat_completion_llm_id
        self.embedding_llm_id = embedding_llm_id
        # monkey_patch chat completion and embeddings models
        self.logger.info(f"Start oading Dataiku in graphrag. chat_completion_llm_id={chat_completion_llm_id}, embedding_llm_id={embedding_llm_id}")

        from dku_graphrag.index.dataiku_chat_llm import DataikuChatLLM
        from dku_graphrag.index.dataiku_embeddings_llm import DataikuEmbeddingsLLM
        from graphrag.config.enums import LLMType
        from graphrag.index.llm.load_llm import loaders

        def _load_dataiku_chat_llm(on_error, cache, config):
            return DataikuChatLLM(self.chat_completion_llm_id)

        def _load_dataiku_embeddings_llm(on_error, cache, config):
            return DataikuEmbeddingsLLM(self.embedding_llm_id)

        # Override OpenAIChat loader
        loaders[LLMType.OpenAIChat] = {
            "load": _load_dataiku_chat_llm,
            "chat": True,
        }

        # Override the openai_embedding entry to load our embeddings LLM
        loaders[LLMType.OpenAIEmbedding] = {
            "load": _load_dataiku_embeddings_llm,
            "chat": False,
        }

        self.logger.info("Loading Dataiku in graphrag: DONE")

    

    async def run_build_index_pipeline(
        self,
        config,
        verbose: bool,
        resume: str | None,
        memprofile: bool
    ) -> None:
       
     
        progress_logger = LoggerFactory().create_logger(self.logger_type)
        if verbose:
            self.logger.setLevel(level=logging.DEBUG)

        run_id = resume or time.strftime("%Y%m%d-%H%M%S")

        self.logger.info(f"Starting indexing pipeline run with run_id={run_id}, verbose={verbose}, resume={resume}, memprofile={memprofile}")

        self.logger.debug(f"graphrag config is config=\n{config}")

        
        outputs = await build_index(
            config=config,
            run_id=run_id,
            is_resume_run=bool(resume),
            memory_profile=memprofile,
            progress_logger=progress_logger,
            callbacks=None,
        )
        
        self.logger.info("Index building completed.")

        # Process results
        for output in outputs:
            if output.errors:
                for err in output.errors:
                    self.logger.error(f"Error in workflow {output.workflow}: {err}")
            else:
                self.logger.info(f"Workflow completed successfully: {output.workflow}")

    async def run_update_index_pipeline(
       self,
        config,
        verbose: bool,
        resume: str | None,
        memprofile: bool
    ) -> None:
        """
        Run the update indexing.
        """
        progress_logger = LoggerFactory().create_logger(self.logger_type)
        if verbose:
            self.logger.setLevel(level=logging.DEBUG)

        run_id = resume or time.strftime("%Y%m%d-%H%M%S")

        # Check if update storage exists, if not configure it with default values
        if not config.update_index_storage:
            raise ValueError("please configure the update_index_storage")
            
        run_id = "update_run"
        self.logger.info(f"Starting update indexing pipeline run with run_id={run_id}")

       
        try:
            outputs = await build_index(
                config=config,
                run_id=run_id,
                is_resume_run=False,
                memory_profile=memprofile,
                progress_logger=progress_logger,
                callbacks=None,
            )
        except Exception:
            self.logger.exception("An unexpected error occurred during the update indexing.")
            return

        self.logger.info("Update indexing completed.")

        # Process results
        for output in outputs:
            if output.errors:
                for err in output.errors:
                    self.logger.error(f"Error in workflow {output.workflow}: {err}")
            else:
                self.logger.info(f"Workflow completed successfully: {output.workflow}")