import { useEffect, useRef, useState } from 'react';
import { TrackerEvent, TrackerState } from '../types';
import { API_BASE } from '../api';
import { normaliseCeleryStage } from '../lib/pipelines';

export interface GenerationTracker {
  currentStage: string | null;
  worker: string | null;
  queue: string;
  events: TrackerEvent[];
  trackerState: TrackerState;
  error: string | null;
  result: any | null;
}

const EMPTY: GenerationTracker = {
  currentStage: null,
  worker: null,
  queue: '3d_generation',
  events: [],
  trackerState: 'idle',
  error: null,
  result: null,
};

/**
 * Tracks a running 3D-generation Celery task via WebSocket (with polling
 * fallback). Feed it the uid returned by /image-to-3d/async (or text/multiview).
 * Returns live stage info, worker metadata, and the final result dict.
 */
export function useGenerationTracker(uid: string | null): GenerationTracker {
  const [state, setState] = useState<GenerationTracker>(EMPTY);
  const wsRef = useRef<WebSocket | null>(null);
  const startRef = useRef<number>(0);
  const cancelledRef = useRef(false);

  useEffect(() => {
    if (!uid) {
      setState(EMPTY);
      return;
    }
    cancelledRef.current = false;
    startRef.current = Date.now();

    setState({
      currentStage: 'queued',
      worker: null,
      queue: '3d_generation',
      events: [{ ts: Date.now(), stage: 'queued', pct: 0, message: 'Task submitted to Redis broker' }],
      trackerState: 'queued',
      error: null,
      result: null,
    });

    const pushEvent = (prog: any) => {
      const raw = prog.stage as string;
      const stage = normaliseCeleryStage(raw);
      const event: TrackerEvent = {
        ts: prog.ts ? prog.ts * 1000 : Date.now(),
        stage,
        pct: prog.pct ?? 0,
        worker: prog.worker,
        queue: prog.queue,
      };
      setState(prev => {
        const newEvents = [...prev.events, event].slice(-50);
        const trackerState: TrackerState =
          stage === 'completed' ? 'completed'
          : stage === 'failed'  ? 'failed'
          : stage === 'cancelled' ? 'failed'
          : 'running';
        return {
          currentStage: stage,
          worker: prog.worker ?? prev.worker,
          queue: prog.queue ?? prev.queue,
          events: newEvents,
          trackerState,
          error: trackerState === 'failed' ? (prog.error ?? 'Task failed') : null,
          result: trackerState === 'completed' ? prog : null,
        };
      });
    };

    const wsBase = API_BASE.replace(/^http/, 'ws');
    const ws = new WebSocket(`${wsBase}/api/v1/ws/generation/${uid}`);
    wsRef.current = ws;

    ws.onmessage = (ev) => {
      if (cancelledRef.current) return;
      try { pushEvent(JSON.parse(ev.data)); } catch {}
    };

    ws.onerror = () => {
      if (cancelledRef.current) return;
      // Fallback: polling
      ws.close();
      const poll = async () => {
        while (!cancelledRef.current) {
          await new Promise(r => setTimeout(r, 2000));
          if (cancelledRef.current) return;
          try {
            const res = await fetch(`${API_BASE}/api/v1/generation-status/${uid}`);
            if (!res.ok) continue;
            const data = await res.json();
            pushEvent({ stage: data.status, pct: data.pct ?? 0, ...data });
            if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') return;
          } catch {}
        }
      };
      poll();
    };

    ws.onclose = () => {};

    return () => {
      cancelledRef.current = true;
      ws.close();
      wsRef.current = null;
    };
  }, [uid]);

  return state;
}
