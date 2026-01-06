document.addEventListener('DOMContentLoaded', function() {
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');
    const chatContainer = document.getElementById('chat-container');
    const resetButton = document.getElementById('reset-chat');
    const quickQuestionButtons = document.querySelectorAll('.quick-question-btn');
    const welcomeHeading = document.getElementById('welcome-heading');
    const suggestionsDropdown = document.getElementById('suggestions-dropdown');
    
    let allPatterns = [];
    let isProcessing = false;
    
    function loadPatterns() {
        fetch('/api/patterns')
            .then(response => response.json())
            .then(data => {
                allPatterns = data.patterns || [];
                console.log(`Loaded ${allPatterns.length} patterns`);
            })
            .catch(error => {
                console.error('Error loading patterns:', error);
            });
    }
    
    function filterPatterns(query) {
        if (!query || query.trim() === '') {
            return [];
        }
        
        const queryWords = query.toLowerCase().trim().split(/\s+/);
        
        const scoredPatterns = allPatterns.map(pattern => {
            const lowerPattern = pattern.toLowerCase();
            let score = 0;
            let matchedWords = 0;
            
            queryWords.forEach(word => {
                if (word.length < 2) return;
                
                if (lowerPattern.includes(word)) {
                    matchedWords++;
                    
                    if (lowerPattern.startsWith(word)) {
                        score += 3;
                    } else if (lowerPattern.split(/\s+/).some(w => w.startsWith(word))) {
                        score += 2;
                    } else {
                        score += 1;
                    }
                }
            });
            
            if (matchedWords > 0) {
                score += matchedWords * 10;
                score += (matchedWords / queryWords.length) * 20;
            }
            
            return { pattern, score };
        });
        
        return scoredPatterns
            .filter(item => item.score > 0)
            .sort((a, b) => b.score - a.score)
            .slice(0, 10)
            .map(item => item.pattern);
    }
    
    function showSuggestions(suggestions) {
        if (suggestions.length === 0) {
            suggestionsDropdown.classList.add('hidden');
            return;
        }
        
        suggestionsDropdown.innerHTML = '';
        
        const header = document.createElement('div');
        header.className = 'px-4 py-2 bg-gray-50 border-b border-gray-200 text-xs font-semibold text-gray-600 uppercase tracking-wider cursor-default';
        header.innerHTML = 'Pilih Pertanyaan';
        suggestionsDropdown.appendChild(header);
        
        suggestions.forEach((pattern) => {
            const item = document.createElement('div');
            item.className = 'px-4 py-3 hover:bg-yellow-100 cursor-pointer border-b border-gray-100 flex items-center';
            item.innerHTML = `
                <div class="flex-shrink-0 mr-3">
                    <i class="fas fa-comment-dots text-yellow-500 text-sm"></i>
                </div>
                <div class="flex-grow">
                    <p class="text-sm text-gray-800 font-medium">${escapeHtml(pattern)}</p>
                </div>
            `;
            
            item.addEventListener('mousedown', function(e) {
                e.preventDefault();
                selectAndSendPattern(pattern);
            });
            
            item.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
            });
            
            suggestionsDropdown.appendChild(item);
        });
        
        suggestionsDropdown.classList.remove('hidden');
    }
    
    function hideSuggestions() {
        suggestionsDropdown.classList.add('hidden');
    }
    
    function selectAndSendPattern(pattern) {
        if (isProcessing) return;
        isProcessing = true;
        
        suggestionsDropdown.classList.add('hidden');
        userInput.value = '';
        userInput.blur();
        sendMessage(pattern);
        
        setTimeout(() => {
            isProcessing = false;
        }, 500);
    }
    
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    userInput.addEventListener('input', function() {
        const query = this.value.trim();
        
        if (query.length > 0) {
            const suggestions = filterPatterns(query);
            showSuggestions(suggestions);
        } else {
            hideSuggestions();
        }
    });
    
    userInput.addEventListener('blur', function() {
        setTimeout(() => {
            hideSuggestions();
        }, 200);
    });
    
    userInput.addEventListener('focus', function() {
        const query = this.value.trim();
        if (query.length > 0) {
            const suggestions = filterPatterns(query);
            showSuggestions(suggestions);
        }
    });
    
    userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
            e.preventDefault();
        }
    });

    function loadMessages() {
        fetch('/api/messages')
            .then(response => response.json())
            .then(data => {
                chatMessages.innerHTML = ''; 
                if (data.messages.length === 0 || (data.messages.length === 1 && data.messages[0].role === 'system')) {
                     addMessageToChat('assistant', 'Selamat datang! Saya adalah AI Assistant BRMP Penerapan. Tanyakan apa pun seputar BRMP.');
                } else {
                    data.messages.forEach(message => {
                        if (message.role !== 'system') {
                            addMessageToChat(message.role, message.content);
                        }
                    });
                }
                scrollToBottom();
            })
            .catch(error => {
                console.error('Error loading messages:', error);
                addMessageToChat('assistant', 'Selamat datang! Saya adalah AI Assistant BRMP Penerapan. Tanyakan apa pun seputar BRMP.');
            });
    }

    function formatMessageContent(content) {
        let formatted = content
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
        
        formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        formatted = formatted.replace(/__([^_]+)__/g, '<strong>$1</strong>');
        formatted = formatted.replace(/\*([^*]+)\*/g, '<em>$1</em>');
        formatted = formatted.replace(/_([^_]+)_/g, '<em>$1</em>');
        formatted = formatted.replace(/^•\s/gm, '&bull; ');
        formatted = formatted.replace(/\n•\s/g, '\n&bull; ');
        formatted = formatted.replace(/^-\s/gm, '&bull; ');
        formatted = formatted.replace(/\n-\s/g, '\n&bull; ');
        
        return formatted;
    }

    function addMessageToChat(role, content) {
        const messageDiv = document.createElement('div');
        const formattedContent = formatMessageContent(content);

        if (role === 'user') {
            messageDiv.className = 'flex justify-end mb-4 animate-fade-in';
            messageDiv.innerHTML = `
                <div class="bg-yellow-500 text-white rounded-t-xl rounded-bl-xl py-3 px-4 max-w-md shadow-md">
                    <div class="chat-message-content">${formattedContent}</div>
                </div>`;
        } else {
            messageDiv.className = 'flex justify-start mb-4 animate-fade-in';
            messageDiv.innerHTML = `
                <div class="bg-white text-gray-800 rounded-t-xl rounded-br-xl py-3 px-4 max-w-md shadow-md border border-gray-200">
                    <div class="chat-message-content">${formattedContent}</div>
                </div>`;
        }
        
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function sendMessage(message) {
        if (!message || message.trim() === '') return;

        addMessageToChat('user', message);

        const typingIndicator = document.createElement('div');
        typingIndicator.id = 'typing-indicator';
        typingIndicator.className = 'flex justify-start mb-4';
        typingIndicator.innerHTML = `
            <div class="bg-white text-gray-800 rounded-t-xl rounded-br-xl p-4 max-w-xs shadow-md border border-gray-200">
                <div class="typing-dots"><span></span><span></span><span></span></div>
            </div>`;
        chatMessages.appendChild(typingIndicator);
        scrollToBottom();

        fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            setTimeout(() => {
                document.getElementById('typing-indicator')?.remove();
                addMessageToChat('assistant', data.response);
            }, 1500);
        })
        .catch(error => {
            console.error('Error:', error);
            setTimeout(() => {
                document.getElementById('typing-indicator')?.remove();
                addMessageToChat('assistant', 'Maaf, terjadi kesalahan pada server. Silakan coba lagi nanti.');
            }, 1500);
        });
    }

    quickQuestionButtons.forEach(button => {
        button.addEventListener('click', () => {
            const question = button.textContent.trim();
            sendMessage(question);
        });
    });

    resetButton.addEventListener('click', function() {
        fetch('/api/reset', { method: 'POST' })
            .then(() => {
                chatMessages.innerHTML = '';
                addMessageToChat('assistant', 'Hai! Ada yang bisa saya bantu hari ini?');
                userInput.value = '';
                hideSuggestions();
            })
            .catch(error => console.error('Error resetting chat:', error));
    });

    function animasiKetikHapus(element, text, typeSpeed = 100, deleteSpeed = 50, pause = 2000) {
        let i = 0;

        function ketik() {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
                setTimeout(ketik, typeSpeed);
            } else {
                setTimeout(hapus, pause);
            }
        }

        function hapus() {
            if (i > 0) {
                element.textContent = text.substring(0, i - 1);
                i--;
                setTimeout(hapus, deleteSpeed);
            } else {
                setTimeout(ketik, 500);
            }
        }

        ketik();
    }

    if (welcomeHeading) {
        const welcomeText = "Halo, Selamat Datang!";
        animasiKetikHapus(welcomeHeading, welcomeText);
    }

    loadPatterns();
    loadMessages();
    if(userInput) userInput.focus();
});