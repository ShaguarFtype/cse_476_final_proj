from flask import Flask, render_template, request, jsonify
import logging
import time
import os
# model interface
from model.llm_interface import LlamaInterface

# initalize
app = Flask(__name__)

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# initalize model interface
model_interface = LlamaInterface()

# send query to model
def query_model(prompt, **kwargs):
    """
    Query the LLM with the given prompt.
    
    Args:
        prompt (str): User input prompt
        **kwargs: Additional parameters for the model
        
    Returns:
        str: Model response
    """
    # call model interface
    return model_interface.generate_response(
        prompt=prompt,
        max_length=kwargs.get('max_length', 100),
        temperature=kwargs.get('temperature', 0.7)
    )

# load chat page
@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

# send prompt to model
@app.route('/query', methods=['POST'])
def process_query():
    """Handle API queries to the model."""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({'error': 'Empty prompt'}), 400
        
        # log prompt for debug
        logger.info(f"Prompt length = {len(prompt)} characters")
        start_time = time.time()
        
        # send prompt
        response = query_model(prompt)
        
        # verify time per instructions
        processing_time = time.time() - start_time
        logger.info(f"Query took {processing_time:.2f} seconds")
        
        return jsonify({
            'response': response,
            'processing_time': f"{processing_time:.2f}s"
        })
    
    except Exception as e:
        logger.exception("Error processing query")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # load model first
    logger.info("Loading model")
    model_interface.load_model()
    
    # launch app
    logger.info("Launching App")
    app.run(debug=True, host='0.0.0.0', port=5000)