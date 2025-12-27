/**
 * AI Chat Initialization Helper
 * Handles modal opening/closing and component initialization
 */

import { AIChatComponent } from './ai-chat.js';
import { getContextBuilder } from './ai-chat-context-builders.js';

// Global AI Chat instance
let aiChatInstance = null;

/**
 * Initialize AI Chat for a specific page
 * @param {string} pageId - Page identifier (e.g., 'dashboard', 'saxo-dashboard')
 */
export function initAIChat(pageId = 'generic') {
    // Inject modal HTML if not present
    if (!document.getElementById('aiChatModal')) {
        fetch('/static/components/ai-chat-modal.html')
            .then(res => res.text())
            .then(html => {
                document.body.insertAdjacentHTML('beforeend', html);
            })
            .catch(error => {
                console.error('Failed to load AI chat modal:', error);
            });
    }

    // Create AI Chat instance
    const contextBuilder = getContextBuilder(pageId);
    aiChatInstance = new AIChatComponent({
        page: pageId,
        contextBuilder: contextBuilder
    });

    // Store in window for global access
    window.aiChat = {
        open: openAIChat,
        close: closeAIChat,
        sendMessage: (msg) => aiChatInstance.sendMessage(msg),
        instance: aiChatInstance
    };

    // Keyboard shortcut: Ctrl+K or Ctrl+Shift+A
    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey && e.key === 'k') || (e.ctrlKey && e.shiftKey && e.key === 'A')) {
            e.preventDefault();
            openAIChat();
        }
    });

    console.log(`AI Chat initialized for page: ${pageId}`);
}

/**
 * Open AI Chat modal
 */
export async function openAIChat() {
    const modal = document.getElementById('aiChatModal');
    if (!modal) {
        console.error('AI Chat modal not found');
        return;
    }

    // Show modal
    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden';

    // Load provider status and quick questions
    if (aiChatInstance) {
        await aiChatInstance.checkStatus();
        await aiChatInstance.loadQuickQuestions();
    }

    // Focus input
    setTimeout(() => {
        const input = document.getElementById('aiChatInput');
        if (input) input.focus();
    }, 300);
}

/**
 * Close AI Chat modal
 */
export function closeAIChat() {
    const modal = document.getElementById('aiChatModal');
    if (modal) {
        modal.style.display = 'none';
        document.body.style.overflow = '';
    }
}

/**
 * Add AI button to navigation (floating action button)
 */
export function addAIButtonToNav() {
    // Check if button already exists
    if (document.getElementById('aiFAB')) return;

    // Create floating action button
    const fab = document.createElement('button');
    fab.id = 'aiFAB';
    fab.className = 'ai-chat-fab';
    fab.title = 'Assistant IA (Ctrl+K)';
    fab.onclick = openAIChat;

    document.body.appendChild(fab);
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        addAIButtonToNav();
    });
} else {
    addAIButtonToNav();
}

// Export for manual initialization
window.initAIChat = initAIChat;
window.openAIChat = openAIChat;
window.closeAIChat = closeAIChat;
