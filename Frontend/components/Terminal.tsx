import React, { useEffect, useRef } from 'react';
import { ProcessLog } from '../types';
import { IconTerminal } from './Icons';

interface TerminalProps {
  logs: ProcessLog[];
}

export const Terminal: React.FC<TerminalProps> = ({ logs }) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="flex flex-col h-full bg-[#050505] rounded-xl border border-zinc-800 overflow-hidden font-mono text-sm shadow-inner shadow-black/50">
      <div className="flex items-center px-4 py-2 bg-black border-b border-zinc-800">
        <IconTerminal className="w-4 h-4 text-zinc-500 mr-2" />
        <span className="text-zinc-500 font-semibold text-xs uppercase tracking-wider">FastAPI / Celery Worker Logs</span>
      </div>
      <div 
        ref={scrollRef}
        className="flex-1 p-4 overflow-y-auto space-y-1 scrollbar-hide"
      >
        {logs.length === 0 && (
          <div className="text-zinc-700 italic">Waiting for input stream...</div>
        )}
        {logs.map((log) => (
          <div key={log.id} className="flex gap-2">
            <span className="text-zinc-600 whitespace-nowrap">[{log.timestamp}]</span>
            <span className={`
              ${log.type === 'error' ? 'text-red-500' : ''}
              ${log.type === 'warning' ? 'text-[#FF8C66]' : ''}
              ${log.type === 'success' ? 'text-[#7C3AED]' : ''}
              ${log.type === 'info' ? 'text-zinc-300' : ''}
            `}>
              {log.type === 'info' && 'INFO:'}
              {log.type === 'success' && 'OK:'}
              {log.type === 'warning' && 'WARN:'}
              {log.type === 'error' && 'ERR:'}
            </span>
            <span className="text-zinc-400 break-all">{log.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
};