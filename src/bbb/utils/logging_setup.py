import logging

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',

    # Uncomment the preferred logging level
    # level=logging.DEBUG,
    level=logging.INFO,
)
logging.getLogger('matplotlib').setLevel(logging.INFO)
