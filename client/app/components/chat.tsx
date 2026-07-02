'use client';

/**
 * chat.tsx — Advanced Chat Component
 *
 * Features:
 * - STT: Web Speech API mic button → live transcription → auto-fills input
 * - TTS: Speaker button on each AI response → speechSynthesis reads it aloud
 * - Multi-collection: passes selected collection IDs to the /chat endpoint
 * - Rich source viewer: renders text/table/image chunks differently
 * - Query is shown as user bubble; AI response has source citation drawer
 */
import * as React from 'react';
import {
  Mic, MicOff, Send, Volume2, VolumeX, ChevronDown, ChevronUp,
  FileText, Table2, ImageIcon, Loader2, GitBranch, Trash2
} from 'lucide-react';

interface Doc {
  excerpt: string;           // backend: first 220 chars of chunk text
  page?: number;
  chunk_type?: 'text' | 'table' | 'image';  // backend: content type
  source?: string;
  collectionId?: string;
  rrfScore?: number;
  rerankScore?: number;
  from_graph?: boolean;      // backend: Graph RAG flag
}

interface IMessage {
  role: 'assistant' | 'user';
  content?: string;
  documents?: Doc[];
}


interface Props {
  selectedCollectionIds: Set<string>;
  /** When set by parent (e.g. from summary), auto-injects an AI message into chat */
  injectedAIMessage?: { content: string; ts: number } | null;
}

const ChatComponent: React.FC<Props> = ({ selectedCollectionIds, injectedAIMessage }) => {

  const [message, setMessage] = React.useState('');
  const [messages, setMessages] = React.useState<IMessage[]>([]);
  const [isLoading, setIsLoading] = React.useState(false);
  const [isListening, setIsListening] = React.useState(false);
  const [speakingIndex, setSpeakingIndex] = React.useState<number | null>(null);
  const [expandedSources, setExpandedSources] = React.useState<Set<number>>(new Set());
  const [interimTranscript, setInterimTranscript] = React.useState('');
  const [loadingStage, setLoadingStage] = React.useState('Searching...');

  const messagesEndRef = React.useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const speechRef = React.useRef<any>(null);
  const historyRef = React.useRef<{ role: string; content: string }[]>([]);

  // Load history from localStorage on mount
  React.useEffect(() => {
    const saved = localStorage.getItem('docuverse_chat_history');
    if (saved) {
      try {
        const parsed: IMessage[] = JSON.parse(saved);
        setMessages(parsed);
        // Sync historyRef for the LLM
        historyRef.current = parsed.map(m => ({ role: m.role, content: m.content || '' })).slice(-12);
      } catch (e) {
        console.error('Failed to parse chat history', e);
      }
    }
  }, []);

  // Save history to localStorage
  React.useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem('docuverse_chat_history', JSON.stringify(messages));
    }
  }, [messages]);

  // Stage-cycling loader — cycles through pipeline stage names while loading
  const LOADING_STAGES = [
    'HyDE: Generating hypothetical doc...',
    'Expanding query variants...',
    'Searching collections...',
    'Graph RAG: Expanding entity neighbors...',
    'Re-ranking chunks...',
    'Compressing context...',
    'Generating answer...',
  ];
  const stageIntervalRef = React.useRef<ReturnType<typeof setInterval> | null>(null);
  const stageIndexRef = React.useRef(0);

  // Auto-scroll to latest message
  React.useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Start / stop stage cycling when loading changes
  React.useEffect(() => {
    
    if (isLoading) {
      stageIndexRef.current = 0;
      setLoadingStage(LOADING_STAGES[0]);
      stageIntervalRef.current = setInterval(() => {
        stageIndexRef.current = (stageIndexRef.current + 1) % LOADING_STAGES.length;
        setLoadingStage(LOADING_STAGES[stageIndexRef.current]);
      }, 2800);
    } else {
      if (stageIntervalRef.current) clearInterval(stageIntervalRef.current);
    }
    return () => { if (stageIntervalRef.current) clearInterval(stageIntervalRef.current); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLoading]);

  // Inject AI messages from parent (e.g. document summaries)
  React.useEffect(() => {
    if (!injectedAIMessage) return;
    setMessages((prev) => {
      // Replace a loading placeholder if present, else append
      const last = prev[prev.length - 1];
      if (last?.role === 'assistant' && last.content?.startsWith('⏳')) {
        return [...prev.slice(0, -1), { role: 'assistant', content: injectedAIMessage.content }];
      }
      return [...prev, { role: 'assistant', content: injectedAIMessage.content }];
    });
  }, [injectedAIMessage]);

  // ── STT: Web Speech API ─────────────────────────────────────────
  const startListening = () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const SpeechRecognitionAPI = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SpeechRecognitionAPI) {
      alert('Speech recognition is not supported in this browser. Try Chrome.');
      return;
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const recognition: any = new SpeechRecognitionAPI();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = () => setIsListening(true);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    recognition.onresult = (event: any) => {
      let final = message;
      let interim = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          final += (final ? ' ' : '') + transcript;
        } else {
          interim += transcript;
        }
      }
      setMessage(final);
      setInterimTranscript(interim);
    };

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    recognition.onerror = (e: any) => {
      console.warn('STT error:', e.error);
      setIsListening(false);
      setInterimTranscript('');
    };

    recognition.onend = () => {
      setIsListening(false);
      setInterimTranscript('');
    };

    speechRef.current = recognition;
    recognition.start();
  };

  const stopListening = () => {
    speechRef.current?.stop();
    setIsListening(false);
    setInterimTranscript('');
  };

  const toggleListening = () => (isListening ? stopListening() : startListening());

  // ── TTS: Web SpeechSynthesis ────────────────────────────────────
  const speak = (text: string, index: number) => {
    if (speakingIndex === index) {
      window.speechSynthesis.cancel();
      setSpeakingIndex(null);
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.onend = () => setSpeakingIndex(null);
    utterance.onerror = () => setSpeakingIndex(null);
    setSpeakingIndex(index);
    window.speechSynthesis.speak(utterance);
  };

  // ── Send message ────────────────────────────────────────────────
  const handleSend = async () => {
    const query = message.trim();
    if (!query || isLoading) return;

    stopListening();
    setIsLoading(true);
    setMessages((prev) => [...prev, { role: 'user', content: query }]);
    setMessage('');
    setInterimTranscript('');

    try {
      const collectionsParam = Array.from(selectedCollectionIds).join(',');
      // Send conversation history so the LLM can handle follow-up questions
      const historyParam = encodeURIComponent(JSON.stringify(historyRef.current));
      const baseUrl = process.env.NEXT_PUBLIC_SERVER_URL || 'http://localhost:8000';
      const url = `${baseUrl}/chat?message=${encodeURIComponent(query)}&collections=${collectionsParam}&history=${historyParam}`;
      const res = await fetch(url);
      const data = await res.json();

      // backend returns: { message, docs: [{source, page, excerpt, from_graph}], meta }
      const assistantContent = data?.message || 'No response received.';

      // Append to history (keep last 10 messages = 5 exchanges)
      historyRef.current = [
        ...historyRef.current,
        { role: 'user', content: query },
        { role: 'assistant', content: assistantContent },
      ].slice(-10);

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: assistantContent,
          documents: (data?.docs || []) as Doc[],
        },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: '⚠️ Connection error. Is the backend running?' },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const clearHistory = () => {
    if (window.confirm('Clear all chat history?')) {
      setMessages([]);
      historyRef.current = [];
      localStorage.removeItem('docuverse_chat_history');
    }
  };

  const toggleSources = (index: number) => {
    setExpandedSources((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

  // ── Render a source doc based on chunk_type ─────────────────────
  const renderDoc = (doc: Doc, idx: number) => {
    const chunkType = doc.chunk_type || 'text';
    const typeIcon = chunkType === 'table'
      ? <Table2 size={11} className="text-amber-400" />
      : chunkType === 'image'
      ? <ImageIcon size={11} className="text-purple-400" />
      : <FileText size={11} className="text-slate-400" />;

    const typeBadge = chunkType !== 'text' ? (
      <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
        chunkType === 'table' ? 'bg-amber-900/40 text-amber-300' : 'bg-purple-900/40 text-purple-300'
      }`}>{chunkType}</span>
    ) : null;

    return (
      <div key={idx} className="bg-slate-800 rounded-lg p-3 border border-slate-700 text-xs">
        <div className="flex items-center gap-2 mb-2 flex-wrap">
          {typeIcon}
          <span className="text-slate-300 font-medium">{doc.source || 'Unknown'}</span>
          <span className="ml-auto text-slate-500">pg {doc.page ?? '?'}</span>
          {typeBadge}
          {doc.from_graph && (
            <span className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-cyan-900/40 text-cyan-300 border border-cyan-700/40">
              <GitBranch size={9} />
              Graph
            </span>
          )}
          {typeof doc.rerankScore === 'number' && (
            <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-indigo-900/30 text-indigo-300">
              ★ {doc.rerankScore.toFixed(0)}
            </span>
          )}
        </div>

        {chunkType === 'table' ? (
          <div className="overflow-x-auto">
            <table className="text-xs text-slate-300 border-collapse w-full">
              <tbody>
                {doc.excerpt.replace('[TABLE]','').replace('[/TABLE]','').trim().split('\n').map((row, ri) => {
                  if (row.startsWith('|---') || row.startsWith('| ---')) return null;
                  const cells = row.split('|').filter(c => c.trim() !== '');
                  if (!cells.length) return null;
                  const Tag = ri === 0 ? 'th' : 'td';
                  return (
                    <tr key={ri} className="border-b border-slate-700">
                      {cells.map((cell, ci) => (
                        <Tag key={ci} className="px-2 py-1 border border-slate-700 text-left">{cell.trim()}</Tag>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : chunkType === 'image' ? (
          <p className="text-slate-400 italic">{doc.excerpt.replace('[IMAGE DESCRIPTION]:', '').trim()}</p>
        ) : (
          <p className="text-slate-400 line-clamp-4 leading-relaxed">{doc.excerpt}</p>
        )}
      </div>
    );
  };

  return (
    <div className="flex flex-col h-screen bg-slate-950 font-sans">
      {/* Header bar */}
      <div className="px-6 py-3 border-b border-slate-800 flex items-center gap-3">
        <h1 className="text-white font-semibold text-sm">PDF Chat</h1>
        {/* Graph RAG mode badge */}
        <span className="flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-cyan-900/30 text-cyan-400 border border-cyan-700/40">
          <GitBranch size={9} />
          Graph RAG
        </span>
        <span className="ml-auto text-slate-500 text-xs">
          {selectedCollectionIds.size === 0
            ? 'No collections selected'
            : `${selectedCollectionIds.size} collection${selectedCollectionIds.size > 1 ? 's' : ''} active`}
        </span>
        {messages.length > 0 && (
          <button
            onClick={clearHistory}
            className="p-1.5 rounded-lg text-slate-500 hover:text-red-400 hover:bg-red-400/10 transition-colors"
            title="Clear Chat History"
          >
            <Trash2 size={16} />
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-center">
            <div className="w-16 h-16 rounded-2xl bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center mb-4">
              <FileText size={28} className="text-indigo-400" />
            </div>
            <p className="text-slate-400 text-sm font-medium">Ask anything about your PDFs</p>
            <p className="text-slate-600 text-xs mt-1">
              Select collections in the sidebar, then type or speak your question
            </p>
          </div>
        )}

        {messages.map((msg, index) => (
          <div key={index} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div
              className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                msg.role === 'user'
                  ? 'bg-indigo-600 text-white rounded-br-sm'
                  : 'bg-slate-800 text-slate-100 rounded-bl-sm border border-slate-700'
              }`}
            >
              {/* Message content */}
              <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>

              {/* AI message controls + sources */}
              {msg.role === 'assistant' && (
                <div className="mt-2 pt-2 border-t border-slate-700/50">
                  <div className="flex items-center gap-2">
                    {/* TTS button */}
                    <button
                      onClick={() => msg.content && speak(msg.content, index)}
                      className="flex items-center gap-1 text-xs text-slate-500 hover:text-indigo-400 transition-colors px-2 py-1 rounded-lg hover:bg-indigo-900/20"
                    >
                      {speakingIndex === index ? (
                        <><VolumeX size={13} /> Stop</>
                      ) : (
                        <><Volume2 size={13} /> Listen</>
                      )}
                    </button>

                    {/* Sources toggle */}
                    {msg.documents && msg.documents.length > 0 && (
                      <button
                        onClick={() => toggleSources(index)}
                        className="ml-auto flex items-center gap-1 text-xs text-slate-500 hover:text-slate-300 transition-colors px-2 py-1 rounded-lg hover:bg-slate-700/50"
                      >
                        {expandedSources.has(index) ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                        {msg.documents.length} source{msg.documents.length > 1 ? 's' : ''}
                      </button>
                    )}
                  </div>

                  {/* Expanded sources */}
                  {expandedSources.has(index) && msg.documents && (
                    <div className="mt-3 space-y-2">
                      {msg.documents.map((doc, di) => renderDoc(doc, di))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Loading indicator with stage cycling */}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-slate-800 border border-slate-700 rounded-2xl rounded-bl-sm px-4 py-3 flex items-center gap-2 max-w-[80%]">
              <Loader2 size={14} className="animate-spin text-indigo-400 flex-shrink-0" />
              <span className="text-sm text-slate-400">{loadingStage}</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="px-6 py-4 border-t border-slate-800 bg-slate-950">
        {/* Interim transcript preview */}
        {interimTranscript && (
          <p className="text-xs text-slate-500 italic mb-2 px-1">{interimTranscript}...</p>
        )}

        <div className="flex gap-3 items-end">
          {/* Mic button */}
          <button
            onClick={toggleListening}
            className={`flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-200 ${
              isListening
                ? 'bg-red-500 hover:bg-red-600 text-white shadow-lg shadow-red-500/30'
                : 'bg-slate-800 hover:bg-slate-700 text-slate-400 border border-slate-700'
            }`}
            title={isListening ? 'Stop listening' : 'Start voice input'}
          >
            {isListening ? <MicOff size={16} /> : <Mic size={16} />}
          </button>

          {/* Textarea */}
          <div className="flex-1 relative">
            <textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder={
                isListening
                  ? '🎤 Listening... speak your question'
                  : 'Ask a question about your PDFs...'
              }
              disabled={isLoading}
              rows={1}
              className="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/30 resize-none disabled:opacity-50 transition-colors"
              style={{ minHeight: '44px', maxHeight: '160px' }}
            />
          </div>

          {/* Send button */}
          <button
            onClick={handleSend}
            disabled={!message.trim() || isLoading}
            className="flex-shrink-0 w-10 h-10 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 disabled:text-slate-500 text-white flex items-center justify-center transition-all duration-200 shadow-lg shadow-indigo-500/20 disabled:shadow-none"
          >
            {isLoading ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatComponent;