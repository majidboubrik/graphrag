import monkey_patch_dataiku  # This sets up the custom LLMs



import asyncio
from pathlib import Path
from graphrag.logger.factory import LoggerType
from index import DataikuGraphragIndexBuilder  

async def main():
    builder = DataikuGraphragIndexBuilder(logger_type=LoggerType.RICH)
    await builder.run_build_index_pipeline( 
        root_dir=Path("./"),
        verbose=True,
        resume=None,
        memprofile=False,
        cache=True,
        config_filepath=None,
        skip_validation=False,
        output_dir=None,
    )

if __name__ == "__main__":
    asyncio.run(main())
