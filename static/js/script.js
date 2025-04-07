document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const statusElement = document.getElementById('status');
    const processingTimeElement = document.getElementById('processing-time');
    
    // this function adds the message to the chat as they are sent and received
    function addMessage(text, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.classList.add(isUser ? 'user' : 'system');
        
        if (!isUser) {
            // Apply markdown rendering for system messages
            messageDiv.innerHTML = renderMarkdown(text);
        } else {
            messageDiv.textContent = text;
        }
        
        chatMessages.appendChild(messageDiv);
        
        // Scroll to the bottom of the chat
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // sends the query to the API
    async function sendQuery(prompt) {
        try {
            statusElement.textContent = 'Processing...';
            processingTimeElement.textContent = '';
            
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ prompt })
            });
            
            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            addMessage(data.response);
            statusElement.textContent = 'Ready';
            processingTimeElement.textContent = `(${data.processing_time})`;
            
        } catch (error) {
            console.error('Error:', error);
            addMessage(`Error: ${error.message}`, false);
            statusElement.textContent = 'Error';
        }
    }
    
    function renderMarkdown(text) {
        // basic markdown support
        text = text.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
        
        // Inline code
        text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Bold
        text = text.replace(/\*\*([^\*]+)\*\*/g, '<strong>$1</strong>');
        
        // Italic
        text = text.replace(/\*([^\*]+)\*/g, '<em>$1</em>');
        
        return text;
    }

    // event listener for send button
    sendButton.addEventListener('click', function() {
        const prompt = userInput.value.trim();
        
        if (prompt) {
            addMessage(prompt, true);
            sendQuery(prompt);
            userInput.value = '';
        }
    });
    
    // event listener so user can hit enter to send message
    userInput.addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendButton.click();
        }
    });
});