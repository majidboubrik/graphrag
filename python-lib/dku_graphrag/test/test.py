import monkey_patch_dataiku  # This sets up the custom LLMs



import asyncio
from pathlib import Path
from graphrag.logger.factory import LoggerType
from index import DataikuGraphragIndexBuilder  
import logging
logging.basicConfig(level=logging.DEBUG)  # Set minimum level to DEBUG


def ensure_event_loop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

def main():
    builder = DataikuGraphragIndexBuilder(logger_type=LoggerType.RICH)
    loop = ensure_event_loop()
    
    loop.run_until_complete(builder.run_build_index_pipeline( 
        root_dir=Path("./"),
        verbose=True,
        resume=None,
        memprofile=False,
        cache=True,
        config_filepath=None,
        skip_validation=False,
        output_dir=None,
    ))

if __name__ == "__main__":
    main()
