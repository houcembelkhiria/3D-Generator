import React, { useEffect, useRef } from 'react';
import { ProcessLog } from '../types';
import { IconTerminal, IconTrash } from './Icons';

interface TerminalProps {
  logs: ProcessLog[];
  onClear?: () => void;
}

export const Terminal: React.FC<TerminalProps> = ({ logs, onClear }) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="terminal flex flex-col h-full overflow-hidden font-mono text-sm shadow-inner shadow-black/20">
      <div className="flex items-center justify-between px-4 py-2 bg-[var(--bg-tertiary)] border-b border-theme">
        <div className="flex items-center">
          <IconTerminal className="w-4 h-4 text-theme-muted mr-2" />
          <span className="text-theme-muted font-semibold text-xs uppercase tracking-wider">FastAPI / Celery Worker Logs</span>
        </div>
        {onClear && (
          <button
            onClick={onClear}
            className="flex items-center gap-1 px-2 py-1 text-xs text-theme-muted hover:text-red-400 hover:bg-[var(--bg-hover)] rounded transition-colors"
            title="Clear logs"
          >
            <IconTrash className="w-3 h-3" />
            Clear
          </button>
        )}
      </div>
      <div
        ref={scrollRef}
        className="flex-1 p-4 overflow-y-auto space-y-1 scrollbar-hide"
      >
        {logs.length === 0 && (
          <div className="text-theme-muted italic">Waiting for input stream...</div>
        )}
        {logs.map((log) => (
          <div key={log.id} className="flex gap-2">
            <span className="text-[var(--text-muted)] whitespace-nowrap">[{log.timestamp}]</span>
            <span className={`
              ${log.type === 'error' ? 'text-red-500' : ''}
              ${log.type === 'warning' ? 'text-[#FF8C66]' : ''}
              ${log.type === 'success' ? 'text-[#7C3AED]' : ''}
              ${log.type === 'info' ? 'text-theme-secondary' : ''}
            `}>
              {log.type === 'info' && 'INFO:'}
              {log.type === 'success' && 'OK:'}
              {log.type === 'warning' && 'WARN:'}
              {log.type === 'error' && 'ERR:'}
            </span>
            <span className="text-[var(--text-tertiary)] break-all">{log.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
