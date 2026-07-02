'use client';

/**
 * collection-sidebar.tsx
 * Displays all uploaded PDF collections with:
 * - Status indicators (processing spinner / ready / error)
 * - Checkbox multi-select for querying against specific PDFs
 * - Upload form with display name input
 * - Delete per-collection
 * - Auto-polls every 4 seconds for status updates
 */
import * as React from 'react';
import { FileText, Trash2, Upload, Loader2, CheckCircle2, XCircle, RefreshCw, GitBranch } from 'lucide-react';

export interface Collection {
  id: string;
  displayName: string;
  filename: string;
  uploadedAt: string;
  status: 'processing' | 'ready' | 'error';
  chunkCount?: number;
}

interface Props {
  /** Currently selected collection IDs for querying */
  selectedIds: Set<string>;
  onSelectionChange: (ids: Set<string>) => void;
  /** Called with collectionId when user clicks Summarize */
  onSummarize: (collectionId: string) => void;
}

const CollectionSidebar: React.FC<Props> = ({ selectedIds, onSelectionChange, onSummarize }) => {
  const [collections, setCollections] = React.useState<Collection[]>([]);
  const [uploadStatus, setUploadStatus] = React.useState<'idle' | 'uploading'>('idle');
  const [displayName, setDisplayName] = React.useState('');
  const [dragOver, setDragOver] = React.useState(false);
  const [summarizingId, setSummarizingId] = React.useState<string | null>(null);
  // Graph entity counts per collectionId (from GET /graph)
  const [graphEntityCounts, setGraphEntityCounts] = React.useState<Record<string, number>>({});

  // Use a ref so fetchCollections never becomes stale but never re-subscribes
  const selectedIdsRef = React.useRef(selectedIds);
  React.useEffect(() => { selectedIdsRef.current = selectedIds; }, [selectedIds]);
  const onSelectionChangeRef = React.useRef(onSelectionChange);
  React.useEffect(() => { onSelectionChangeRef.current = onSelectionChange; }, [onSelectionChange]);

  const fetchCollections = React.useCallback(async () => {
    try {
      const [collectionsRes, graphRes] = await Promise.all([
        fetch(`${process.env.NEXT_PUBLIC_SERVER_URL || 'http://localhost:8000'}/collections`),
        fetch(`${process.env.NEXT_PUBLIC_SERVER_URL || 'http://localhost:8000'}/graph/stats`).catch(() => null),
      ]);
      const data: Collection[] = await collectionsRes.json();
      setCollections(data);

      // Parse graph entity counts
      if (graphRes && graphRes.ok) {
        const graphData = await graphRes.json();
        const counts: Record<string, number> = {};
        for (const [id, stats] of Object.entries(graphData.graphs || {})) {
          counts[id] = (stats as { entities: number }).entities;
        }
        setGraphEntityCounts(counts);
      }

      // Auto-select all ready collections when none are selected yet
      const readyIds = data.filter((c) => c.status === 'ready').map((c) => c.id);
      if (readyIds.length > 0 && selectedIdsRef.current.size === 0) {
        onSelectionChangeRef.current(new Set(readyIds));
      }
    } catch {
      // Server may not be running yet — silently wait
    }
  }, []);

  // Poll for status updates — faster when any PDF is still processing
  React.useEffect(() => {
    fetchCollections();
    const getId = () => {
      const hasProcessing = collections.some((c) => c.status === 'processing');
      return setInterval(fetchCollections, hasProcessing ? 2000 : 6000);
    };
    const id = getId();
    return () => clearInterval(id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [collections.some((c) => c.status === 'processing'), fetchCollections]);


  // ── Upload handler ─────────────────────────────────────────────
  const handleUpload = async (file: File) => {
    if (file.type !== 'application/pdf') {
      alert('Only PDF files are supported');
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      alert('File too large — max 50MB');
      return;
    }

    setUploadStatus('uploading');
    const formData = new FormData();
    formData.append('file', file);  // backend UploadFile param is named 'file'
    formData.append('displayName', displayName.trim() || file.name);

    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_SERVER_URL || 'http://localhost:8000'}/upload/pdf`, { method: 'POST', body: formData });
      if (!res.ok) throw new Error((await res.json()).detail || 'Upload failed');
      await fetchCollections();
      setDisplayName('');
    } catch (err) {
      alert(`Upload failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setUploadStatus('idle');
    }
  };

  const openFilePicker = () => {
    const el = document.createElement('input');
    el.type = 'file';
    el.accept = 'application/pdf';
    el.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) handleUpload(file);
    };
    el.click();
  };

  // ── Selection helpers ──────────────────────────────────────────
  const toggleSelected = (id: string) => {
    const next = new Set(selectedIds);
    if (next.has(id)) {
      next.delete(id);
    } else {
      next.add(id);
    }
    onSelectionChange(next);
  };

  const selectAll = () =>
    onSelectionChange(new Set(collections.filter((c) => c.status === 'ready').map((c) => c.id)));

  const clearAll = () => onSelectionChange(new Set());

  // ── Delete collection ─────────────────────────────────────────
  const deleteCollection = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('Delete this collection?')) return;
    await fetch(`${process.env.NEXT_PUBLIC_SERVER_URL || 'http://localhost:8000'}/collections/${id}`, { method: 'DELETE' });
    const next = new Set(selectedIds);
    next.delete(id);
    onSelectionChange(next);
    await fetchCollections();
  };

  // ── Summarize collection ──────────────────────────────────────
  const handleSummarize = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setSummarizingId(id);
    try {
      await onSummarize(id);
    } finally {
      setSummarizingId(null);
    }
  };

  // ── Status icon ───────────────────────────────────────────────
  const StatusIcon = ({ status }: { status: Collection['status'] }) => {
    if (status === 'processing') return <Loader2 size={14} className="animate-spin text-blue-400" />;
    if (status === 'ready') return <CheckCircle2 size={14} className="text-emerald-400" />;
    return <XCircle size={14} className="text-red-400" />;
  };

  return (
    <div className="h-full flex flex-col bg-slate-950 border-r border-slate-800 font-sans">
      {/* Header */}
      <div className="p-4 border-b border-slate-800">
        <h2 className="text-white font-semibold text-sm tracking-wide flex items-center gap-2">
          <FileText size={16} className="text-indigo-400" />
          PDF Collections
        </h2>
        <p className="text-slate-500 text-xs mt-1">
          {collections.filter((c) => c.status === 'ready').length} ready
          {collections.some((c) => c.status === 'processing') && (
            <span className="ml-2 text-blue-400 inline-flex items-center gap-1">
              <Loader2 size={10} className="animate-spin" /> processing
            </span>
          )}
        </p>
      </div>

      {/* Upload area */}
      <div className="p-4 border-b border-slate-800">
        <input
          value={displayName}
          onChange={(e) => setDisplayName(e.target.value)}
          placeholder="Collection name (optional)"
          className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 mb-2"
        />
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault(); setDragOver(false);
            const file = e.dataTransfer.files[0];
            if (file) handleUpload(file);
          }}
          onClick={uploadStatus === 'idle' ? openFilePicker : undefined}
          className={`border-2 border-dashed rounded-lg p-4 flex flex-col items-center gap-2 cursor-pointer transition-all duration-200 ${
            dragOver
              ? 'border-indigo-400 bg-indigo-950/30'
              : 'border-slate-700 hover:border-indigo-600 hover:bg-slate-900'
          }`}
        >
          {uploadStatus === 'uploading' ? (
            <Loader2 size={20} className="text-indigo-400 animate-spin" />
          ) : (
            <Upload size={20} className="text-slate-400" />
          )}
          <span className="text-xs text-slate-400">
            {uploadStatus === 'uploading' ? 'Uploading...' : 'Drop PDF or click to upload'}
          </span>
        </div>
      </div>

      {/* Select helpers */}
      {collections.length > 0 && (
        <div className="px-4 py-2 flex gap-2 border-b border-slate-800">
          <button onClick={selectAll} className="text-xs text-indigo-400 hover:text-indigo-300">
            All
          </button>
          <span className="text-slate-700">|</span>
          <button onClick={clearAll} className="text-xs text-slate-500 hover:text-slate-300">
            None
          </button>
          <span className="ml-auto text-xs text-slate-600">
            {selectedIds.size} selected
          </span>
        </div>
      )}

      {/* Collections list */}
      <div className="flex-1 overflow-y-auto">
        {collections.length === 0 ? (
          <div className="p-6 text-center text-slate-600 text-sm">
            No PDFs uploaded yet.<br />Upload one to get started.
          </div>
        ) : (
          collections.map((col) => (
            <div
              key={col.id}
              onClick={() => col.status === 'ready' && toggleSelected(col.id)}
              className={`px-4 py-3 border-b border-slate-800/50 flex items-start gap-3 transition-colors duration-150
                ${col.status === 'ready' ? 'cursor-pointer hover:bg-slate-900' : 'opacity-60'}
                ${selectedIds.has(col.id) ? 'bg-indigo-950/40 border-l-2 border-l-indigo-500' : ''}
              `}
            >
              {/* Checkbox */}
              <div className="mt-0.5 flex-shrink-0">
                {col.status === 'ready' ? (
                  <div className={`w-4 h-4 rounded border flex items-center justify-center transition-colors ${
                    selectedIds.has(col.id) ? 'bg-indigo-500 border-indigo-500' : 'border-slate-600'
                  }`}>
                    {selectedIds.has(col.id) && (
                      <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                      </svg>
                    )}
                  </div>
                ) : (
                  <StatusIcon status={col.status} />
                )}
              </div>

              {/* Info */}
              <div className="flex-1 min-w-0">
                <p className="text-sm text-white font-medium truncate">{col.displayName}</p>
                <p className="text-xs text-slate-500 truncate">{col.filename}</p>
                <div className="flex items-center gap-2 mt-1 flex-wrap">
                  <StatusIcon status={col.status} />
                  <span className="text-xs text-slate-600">
                    {col.status === 'processing' && 'Indexing...'}
                    {col.status === 'ready' && `${col.chunkCount ?? '?'} chunks`}
                    {col.status === 'error' && 'Failed'}
                  </span>
                  {/* Graph entity count badge */}
                  {col.status === 'ready' && graphEntityCounts[col.id] !== undefined && (
                    <span className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-cyan-900/30 text-cyan-400 border border-cyan-800/40">
                      <GitBranch size={9} />
                      {graphEntityCounts[col.id]} entities
                    </span>
                  )}
                </div>
              </div>

              {/* Summary button — only for ready collections */}
              {col.status === 'ready' && (
                <button
                  onClick={(e) => handleSummarize(col.id, e)}
                  disabled={summarizingId === col.id}
                  title="Summarize this document"
                  className="flex-shrink-0 p-1 rounded text-slate-600 hover:text-indigo-400 hover:bg-indigo-950/30 transition-colors disabled:opacity-40"
                >
                  {summarizingId === col.id
                    ? <Loader2 size={13} className="animate-spin" />
                    : <span className="text-[11px]">📋</span>}
                </button>
              )}

              {/* Delete */}
              <button
                onClick={(e) => deleteCollection(col.id, e)}
                className="flex-shrink-0 p-1 rounded text-slate-600 hover:text-red-400 hover:bg-red-950/30 transition-colors"
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))
        )}
      </div>

      {/* Refresh */}
      <div className="p-3 border-t border-slate-800">
        <button
          onClick={fetchCollections}
          className="w-full flex items-center justify-center gap-2 text-xs text-slate-500 hover:text-slate-300 py-1 rounded transition-colors"
        >
          <RefreshCw size={12} />
          Refresh
        </button>
      </div>
    </div>
  );
};

export default CollectionSidebar;
