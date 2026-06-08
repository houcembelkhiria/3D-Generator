import { useEffect, useMemo, useRef, useState } from 'react';
import { TrackerEvent, TrackerState } from '../types';
import { useTaskPolling } from './useTaskPolling';
import { normaliseNodeToStep } from '../lib/pipelines';
import { API_BASE } from '../api';

export interface LangGraphTracker {
  currentStage: string | null;
  worker: string | null;
  queue: string;
  events: TrackerEvent[];
  trackerState: TrackerState;
  error: string | null;
  result: any | null;
}

const EMPTY: LangGraphTracker = {
  currentStage: null,
  worker: null,
  queue: 'document_processing',
  events: [],
  trackerState: 'idle',
  error: null,
  result: null,
};

/**
 * Tracks a LangGraph run_pipeline Celery task via HTTP polling.
 * Builds an event log from node_history diffs so the ExecutionTracker
 * receives the same {events, currentStage} shape as useGenerationTracker.
 */
export function useLangGraphTracker(taskId: string | null): LangGraphTracker {
  const poll = useTaskPolling(taskId, API_BASE);
  const prevHistoryLen = useRef(0);
  const [events, setEvents] = useState<TrackerEvent[]>([]);
  const startRef = useRef<number>(0);

  useEffect(() => {
    if (!taskId) {
      setEvents([]);
      prevHistoryLen.current = 0;
      startRef.current = 0;
      return;
    }
    startRef.current = Date.now();
    setEvents([{ ts: Date.now(), stage: 'queued', pct: 0, message: 'Task submitted to Redis broker' }]);
    prevHistoryLen.current = 0;
  }, [taskId]);

  useEffect(() => {
    if (!taskId || poll.status === 'idle') return;

    const meta = poll.meta as any;
    const nodeHistory: string[] = meta?.node_history ?? [];

    // Append new nodes as events
    const newNodes = nodeHistory.slice(prevHistoryLen.current);
    if (newNodes.length > 0) {
      const now = meta?.ts ? meta.ts * 1000 : Date.now();
      const newEvents: TrackerEvent[] = newNodes.map((node, i) => ({
        ts: now + i,
        stage: normaliseNodeToStep(node),
        pct: 0,
        worker: meta?.worker,
        queue: meta?.queue,
        message: node,
      }));
      setEvents(prev => [...prev, ...newEvents].slice(-50));
      prevHistoryLen.current = nodeHistory.length;
    }

    // Completion / failure
    if (poll.status === 'completed' || poll.status === 'failed') {
      const stage = poll.status === 'completed' ? 'store_result' : 'failed';
      const event: TrackerEvent = { ts: Date.now(), stage, pct: 100, worker: meta?.worker, queue: meta?.queue };
      setEvents(prev => {
        const last = prev[prev.length - 1];
        if (last?.stage === stage) return prev;
        return [...prev, event].slice(-50);
      });
    }
  }, [taskId, poll.status, poll.meta]);

  return useMemo((): LangGraphTracker => {
    if (!taskId) return EMPTY;

    const meta = poll.meta as any;
    const currentNode: string | undefined = meta?.current_node;
    const currentStage = currentNode ? normaliseNodeToStep(currentNode) : null;

    const trackerState: TrackerState =
      poll.status === 'completed' ? 'completed'
      : poll.status === 'failed'  ? 'failed'
      : poll.status === 'idle'    ? 'idle'
      : poll.status === 'pending' ? 'queued'
      : 'running';

    return {
      currentStage: trackerState === 'completed' ? 'store_result' : currentStage,
      worker: meta?.worker ?? null,
      queue: meta?.queue ?? 'document_processing',
      events,
      trackerState,
      error: poll.error,
      result: poll.result,
    };
  }, [taskId, poll, events]);
}
