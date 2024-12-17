import dku_graphrag.index.monkey_patch_dataiku  # This sets up the custom LLMs

import logging
from pathlib import Path
from typing import Optional
import time

from graphrag.api import build_index
from graphrag.config.enums import CacheType
from graphrag.config.load_config import load_config
from graphrag.config.logging import enable_logging_with_config
from graphrag.index.validate_config import validate_config_names
from graphrag.logger.factory import LoggerFactory, LoggerType
from graphrag.logger.base import ProgressLogger

class DataikuGraphragIndexBuilder:
    """
    A class to handle building and updating a GraphRAG index, similar to the CLI commands,
    but without sys.exit() and signal handling. Ideal for use inside a Dataiku Python recipe.
    """
    
    def __init__(self, logger_type: LoggerType = LoggerType.RICH):
        self.logger = logging.getLogger(__name__)
        self.logger_type = logger_type

    

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

        self.logger.debug(f"config={config}")
        # Validate configuration if skip_validation is False
        #if not skip_validation:
            #validate_config_names(progress_logger, config)

        
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
        root_dir: Path,
        verbose: bool,
        memprofile: bool,
        cache: bool,
        config_filepath: Optional[Path],
        skip_validation: bool,
        output_dir: Optional[Path],
    ) -> None:
        """
        Run the update indexing pipeline similarly to the cli.index.update_cli function,
        but without sys.exit(), signal handling, or dry_run logic.
        """
        config, progress_logger = self._configure_pipeline(
            root_dir=root_dir,
            config_filepath=config_filepath,
            output_dir=output_dir,
            cache=cache,
            verbose=verbose,
        )

        # Check if update storage exists, if not configure it with default values
        if not config.update_index_storage:
            from graphrag.config.defaults import STORAGE_TYPE, UPDATE_STORAGE_BASE_DIR
            from graphrag.config.models.storage_config import StorageConfig

            config.update_index_storage = StorageConfig(
                type=STORAGE_TYPE,
                base_dir=UPDATE_STORAGE_BASE_DIR,
            )

        run_id = "update_run"
        self.logger.info(f"Starting update indexing pipeline run with run_id={run_id}")

        # Validate configuration if skip_validation is False
        #if not skip_validation:
            #validate_config_names(progress_logger, config)

        try:
            outputs = await build_index(
                config=config,
                run_id="",
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