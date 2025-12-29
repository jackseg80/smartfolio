/**
 * AI Chat Component - Global Reusable
 * Multi-provider AI assistant (Groq + Claude API)
 * Page-aware context injection
 */

export class AIChatComponent {
    constructor(options = {}) {
        this.page = options.page || 'unknown';
        this.contextBuilder = options.contextBuilder || (() => ({}));
        // ✅ USER ISOLATION: Use globalConfig (already isolated per user)
        this.provider = window.globalConfig?.get('aiProvider') || 'groq';
        this.includeDocs = window.globalConfig?.get('aiIncludeDocs') !== false;  // Default true
        this.messages = [];
        this.availableProviders = [];
        this.state = {
            isLoading: false,
            configured: false
        };
    }

    async sendMessage(message) {
        if (!message || !message.trim()) return;

        // Add user message to conversation
        this.messages.push({ role: 'user', content: message });
        this.addChatMessage('user', message);

        // Clear input
        const input = document.getElementById('aiChatInput');
        if (input) input.value = '';

        this.state.isLoading = true;
        this.updateUIState();

        try {
            // Build context using the provided builder
            const context = await this.contextBuilder();
            const fullContext = Object.assign({}, context, {
                page: this.page,
                timestamp: new Date().toISOString()
            });
            console.log('Sending AI chat message with context:', Object.keys(fullContext));

            const response = await fetch('/api/ai/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-User': this.getActiveUser()
                },
                body: JSON.stringify({
                    messages: this.messages,
                    context: fullContext,
                    provider: this.provider,
                    include_docs: this.includeDocs,
                    max_tokens: 2048,
                    temperature: 0.7
                })
            });

            const data = await response.json();

            if (!data.ok) {
                throw new Error(data.error || 'Unknown error');
            }

            // Add AI response to messages and UI
            this.messages.push({ role: 'assistant', content: data.message });
            this.addChatMessage('assistant', data.message, data.usage);

        } catch (error) {
            console.error('AI chat error:', error);
            this.addChatMessage('error', `Erreur: ${error.message}`);
        } finally {
            this.state.isLoading = false;
            this.updateUIState();
        }
    }

    async checkStatus() {
        try {
            const response = await fetch('/api/ai/providers', {
                headers: { 'X-User': this.getActiveUser() }
            });
            const data = await response.json();

            if (data.ok && data.providers) {
                this.availableProviders = data.providers;
                this.state.configured = data.providers.some(p => p.configured);
                this.updateProviderSelector();
            }
        } catch (error) {
            console.error('Failed to check AI status:', error);
        }
    }

    async loadQuickQuestions() {
        try {
            const response = await fetch(`/api/ai/quick-questions/${this.page}`);
            const data = await response.json();

            if (data.ok && data.questions) {
                this.renderQuickQuestions(data.questions);
            }
        } catch (error) {
            console.error('Failed to load quick questions:', error);
        }
    }

    addChatMessage(role, content, usage = null) {
        const messagesContainer = document.getElementById('aiChatMessages');
        if (!messagesContainer) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `ai-chat-message ai-chat-message-${role}`;

        if (role === 'user') {
            messageDiv.innerHTML = `
                <div class="ai-chat-message-content">
                    <strong>Vous:</strong>
                    <div>${this.escapeHtml(content)}</div>
                </div>
            `;
        } else if (role === 'assistant') {
            const formattedContent = this.formatMarkdown(content);
            const usageInfo = usage ? `<small class="ai-chat-usage">✓ ${usage.total_tokens} tokens utilisés</small>` : '';

            messageDiv.innerHTML = `
                <div class="ai-chat-message-content">
                    <strong>AI:</strong>
                    <div>${formattedContent}</div>
                    ${usageInfo}
                </div>
            `;
        } else if (role === 'error') {
            messageDiv.innerHTML = `
                <div class="ai-chat-message-content ai-chat-error">
                    <strong>⚠️ Erreur:</strong>
                    <div>${this.escapeHtml(content)}</div>
                </div>
            `;
        }

        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    renderQuickQuestions(questions) {
        const container = document.getElementById('aiChatQuickQuestions');
        if (!container) return;

        container.innerHTML = '';

        questions.forEach(q => {
            const button = document.createElement('button');
            button.className = 'ai-chat-quick-question';
            button.textContent = q.label;
            button.onclick = () => {
                document.getElementById('aiChatInput').value = q.prompt;
                this.sendMessage(q.prompt);
            };
            container.appendChild(button);
        });
    }

    updateProviderSelector() {
        const selector = document.getElementById('aiProviderSelector');
        if (!selector) return;

        selector.innerHTML = '';

        this.availableProviders.forEach(provider => {
            const option = document.createElement('option');
            option.value = provider.id;
            option.textContent = `${provider.name} ${provider.free ? '(Gratuit)' : '(Premium)'}`;
            option.disabled = !provider.configured;

            if (!provider.configured) {
                option.textContent += ' - Non configuré';
            }

            if (provider.id === this.provider) {
                option.selected = true;
            }

            selector.appendChild(option);
        });

        selector.onchange = (e) => {
            this.switchProvider(e.target.value);
        };
    }

    switchProvider(newProvider) {
        this.provider = newProvider;
        // ✅ USER ISOLATION: Use globalConfig (already isolated per user)
        window.globalConfig?.set('aiProvider', newProvider);
        console.log(`AI provider switched to: ${newProvider}`);
    }

    updateUIState() {
        const input = document.getElementById('aiChatInput');
        const button = document.getElementById('aiChatSendButton');

        if (input) input.disabled = this.state.isLoading;
        if (button) button.disabled = this.state.isLoading;

        if (this.state.isLoading) {
            if (button) button.textContent = 'Envoi...';
        } else {
            if (button) button.textContent = 'Envoyer';
        }
    }

    formatMarkdown(text) {
        // Basic markdown support
        let formatted = text;

        // Bold: **text** or __text__
        formatted = formatted.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        formatted = formatted.replace(/__(.+?)__/g, '<strong>$1</strong>');

        // Italic: *text* or _text_
        formatted = formatted.replace(/\*(.+?)\*/g, '<em>$1</em>');
        formatted = formatted.replace(/_(.+?)_/g, '<em>$1</em>');

        // Code: `text`
        formatted = formatted.replace(/`(.+?)`/g, '<code>$1</code>');

        // Line breaks
        formatted = formatted.replace(/\n/g, '<br>');

        return formatted;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    getActiveUser() {
        return localStorage.getItem('activeUser') || 'demo';
    }

    destroy() {
        const modal = document.getElementById('aiChatModal');
        if (modal) {
            modal.remove();
        }
    }
}

// Export for global access
window.AIChatComponent = AIChatComponent;
