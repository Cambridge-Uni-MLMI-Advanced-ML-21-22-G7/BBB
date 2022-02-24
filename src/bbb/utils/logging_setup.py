import logging

# Ensure logging standardisation
# Log level can be changed to debug by passind `-v` flag to bbb on command line
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

# Make sure that matplotlib logging level is always INFO
# It is very chatting in DEBUG
logging.getLogger('matplotlib').setLevel(logging.INFO)
