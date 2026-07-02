'use client';

/**
 * page.tsx — Root layout
 * Lifts state:
 *  - selectedCollectionIds: which PDFs are active for chat queries
 *  - injectMessage: when set, chat.tsx auto-sends this AI message
 *    (used by Summary button → fetches /summary → injects result into chat)
 */
import * as React from 'react';
import CollectionSidebar from './components/collection-sidebar';
import ChatComponent from './components/chat';

export default function Home() {

  const [selectedCollectionIds, setSelectedCollectionIds] = React.useState<Set<string>>(new Set());

  // When an AI message is injected (e.g. from a summary), chat shows it immediately
  const [injectedAIMessage, setInjectedAIMessage] = React.useState<{ content: string; ts: number } | null>(null);

  const handleSummarize = async (collectionId: string) => {

    // Show a loading indicator by injecting a temporary message
    const loadingMsg = { content: '⏳ Generating summary...', ts: Date.now() };
    setInjectedAIMessage(loadingMsg);

    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_SERVER_URL || 'http://localhost:8000'}/summarize/${collectionId}`);
      const data = await res.json();
      setInjectedAIMessage({ content: data.summary || 'Could not generate summary.', ts: Date.now() });
    } catch {
      setInjectedAIMessage({ content: '⚠️ Failed to generate summary. Is the backend running?', ts: Date.now() });
    }
  };

  return (
    <div className="min-h-screen w-screen flex bg-slate-950 overflow-hidden">
      {/* Sidebar — fixed 280px */}
      <div className="w-[280px] flex-shrink-0 h-screen overflow-hidden">
        <CollectionSidebar
          selectedIds={selectedCollectionIds}
          onSelectionChange={setSelectedCollectionIds}
          onSummarize={handleSummarize}
        />
      </div>

      {/* Chat — fills remaining space */}
      <div className="flex-1 h-screen overflow-hidden border-l border-slate-800">
        <ChatComponent
          selectedCollectionIds={selectedCollectionIds}
          injectedAIMessage={injectedAIMessage}
        />
      </div>
    </div>
  );
}