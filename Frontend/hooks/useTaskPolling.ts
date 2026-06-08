import { useEffect, useRef, useState } from 'react';

/**
 * Poll a Celery task endpoint with exponential-backoff cadence and surface
 * per-node LangGraph progress (current_node, recent_errors) in addition to
 * the basic status. Replaces the 1-second-interval polling loops scattered
 * across TextTo3D / ImageTo3D / MultiViewTo3D / TextExtractor.
 *
 * Cadence (was hammering FastAPI at 1 req/s for 20-min tasks):
 *   0-30s    → 1.5s interval
 *   30s-2min → 3s interval
 *   2-10min  → 5s interval
 *   10min+   → 8s interval
 */

export type TaskStatus = 'idle' | 'pending' | 'processing' | 'completed' | 'failed';

export interface TaskMeta {
  status?: string;
  current_node?: string;
  current_step?: string;
  recent_errors?: string[];
  thread_id?: string;
  resumed?: boolean;
}

export interface TaskPollResult<T = any> {
  status: TaskStatus;
  meta: TaskMeta;
  result: T | null;
  error: string | null;
  elapsedMs: number;
}

function intervalForElapsed(ms: number): number {
  if (ms < 30_000) return 1500;
  if (ms < 120_000) return 3000;
  if (ms < 600_000) return 5000;
  return 8000;
}

/**
 * Poll a task. Returns the latest poll result; updates as the task progresses.
 * Pass `null` taskId to disable polling.
 *
 * @param taskId  the Celery task UUID
 * @param apiBase API base URL (e.g. http://localhost:8000)
 * @param maxRunMs total time budget before giving up (default 25min — covers
 *                 20-min mesh gen + slack). Polling continues if backend still
 *                 reports "processing" past this — does NOT auto-error.
 */
export function useTaskPolling<T = any>(
  taskId: string | null,
  apiBase: string,
  maxRunMs: number = 25 * 60 * 1000,
): TaskPollResult<T> {
  const [state, setState] = useState<TaskPollResult<T>>({
    status: 'idle',
    meta: {},
    result: null,
    error: null,
    elapsedMs: 0,
  });
  const startRef = useRef<number>(0);
  const timeoutRef = useRef<number | null>(null);

  useEffect(() => {
    if (!taskId) {
      setState({ status: 'idle', meta: {}, result: null, error: null, elapsedMs: 0 });
      return;
    }
    startRef.current = Date.now();
    let cancelled = false;

    const tick = async () => {
      if (cancelled) return;
      const elapsedMs = Date.now() - startRef.current;
      try {
        const res = await fetch(`${apiBase}/api/v1/task/${taskId}`);
        if (!res.ok) {
          throw new Error(`Poll failed: HTTP ${res.status}`);
        }
        const payload = await res.json();
        const status = payload.status as TaskStatus;
        // Per-node progress lives in payload.result.{current_node, recent_errors}
        // while the task is processing; final model_info lives there once completed.
        const meta: TaskMeta = (status === 'processing' && payload.result)
          ? payload.result
          : {};
        const result = (status === 'completed') ? (payload.result as T) : null;
        const error = (status === 'failed')
          ? (payload.result?.error ?? 'Task failed')
          : null;

        if (cancelled) return;
        setState({ status, meta, result, error, elapsedMs });

        if (status === 'completed' || status === 'failed') {
          return; // stop polling
        }
        // Soft cap: keep polling past maxRunMs but warn via meta
        const nextInterval = intervalForElapsed(elapsedMs);
        timeoutRef.current = window.setTimeout(tick, nextInterval) as unknown as number;
      } catch (e: any) {
        if (cancelled) return;
        setState(prev => ({ ...prev, error: e.message, elapsedMs }));
        // Don't terminate on transient errors — back off and retry
        const nextInterval = Math.min(intervalForElapsed(elapsedMs) * 2, 15000);
        timeoutRef.current = window.setTimeout(tick, nextInterval) as unknown as number;
      }
    };

    setState({ status: 'pending', meta: {}, result: null, error: null, elapsedMs: 0 });
    tick();

    return () => {
      cancelled = true;
      if (timeoutRef.current !== null) {
        window.clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };
  }, [taskId, apiBase, maxRunMs]);

  return state;
}

/**
 * Format a node name from the streaming pipeline for display.
 * E.g. "spec_extraction:extract_spec_llm" → "Extracting spec"
 */
export function formatNodeName(node?: string): string {
  if (!node) return 'Running pipeline';
  const last = node.split(':').pop() || node;
  const pretty = last
    .replace(/_/g, ' ')
    .replace(/\bnode\b/i, '')
    .trim();
  return pretty.charAt(0).toUpperCase() + pretty.slice(1);
}
