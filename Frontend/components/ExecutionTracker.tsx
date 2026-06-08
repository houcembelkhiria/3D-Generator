import React, { useEffect, useRef, useState } from 'react';
import { TrackerEvent, TrackerState, TrackerStep } from '../types';

interface ExecutionTrackerProps {
  taskId: string | null;
  queue: string;
  worker?: string | null;
  steps: TrackerStep[];
  currentStepId: string | null;
  events: TrackerEvent[];
  elapsedSec: number;
  state: TrackerState;
  error?: string | null;
  title?: string;
}

function fmt(ms: number): string {
  const s = Math.floor((Date.now() - ms) / 1000);
  const m = Math.floor(s / 60);
  return m > 0 ? `${m}m ${s % 60}s ago` : `${s}s ago`;
}

function fmtElapsed(sec: number): string {
  const m = Math.floor(sec / 60);
  return m > 0 ? `${m}m ${sec % 60}s` : `${sec}s`;
}

/** Big monospace clock (H:MM:SS over an hour, MM:SS under). Used in the
 *  header for at-a-glance "how long has this been running". */
function fmtClock(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  const pad = (n: number) => String(n).padStart(2, '0');
  return h > 0 ? `${h}:${pad(m)}:${pad(r)}` : `${pad(m)}:${pad(r)}`;
}

/** Per-step duration in seconds derived from event timestamps. Returns
 *  [completedDurations, activeDurationSec, activeStepId]. Active step's
 *  duration grows in real time via the ticker; completed steps are frozen
 *  at (nextStepFirstSeen - thisStepFirstSeen). */
function computeStepDurations(
  steps: TrackerStep[],
  events: TrackerEvent[],
  currentStepId: string | null,
  state: TrackerState,
  nowMs: number,
): Record<string, number> {
  const firstSeen: Record<string, number> = {};
  for (const ev of events) {
    if (firstSeen[ev.stage] === undefined) firstSeen[ev.stage] = ev.ts;
  }
  const durations: Record<string, number> = {};
  for (let i = 0; i < steps.length; i++) {
    const step = steps[i];
    const start = firstSeen[step.id];
    if (start === undefined) continue;
    let end: number | undefined;
    for (let j = i + 1; j < steps.length; j++) {
      const next = firstSeen[steps[j].id];
      if (next !== undefined) { end = next; break; }
    }
    if (end === undefined) {
      // Step is current or last. Three cases:
      //  - actively running -> count up to "now"
      //  - completed/failed -> freeze at this step's first-event timestamp
      //    (no end => duration shows as "0s" instead of growing unbounded
      //    while the user stares at the post-run UI)
      //  - idle / no events -> drop (no duration shown)
      if (step.id === currentStepId && (state === 'running' || state === 'queued')) {
        end = nowMs;
      } else if (step.id === currentStepId) {
        // Run finished with this step as the "current" one — find the very
        // last event for this stage so the duration is bounded.
        let lastForStep = start;
        for (const ev of events) {
          if (ev.stage === step.id && ev.ts > lastForStep) lastForStep = ev.ts;
        }
        end = lastForStep;
      } else {
        continue;
      }
    }
    durations[step.id] = Math.max(0, Math.floor((end - start) / 1000));
  }
  return durations;
}

function stepStatus(step: TrackerStep, currentStepId: string | null, steps: TrackerStep[], state: TrackerState) {
  if (!currentStepId) return state === 'queued' && step.id === 'queued' ? 'active' : 'pending';
  const currentIdx = steps.findIndex(s => s.id === currentStepId);
  const stepIdx = steps.findIndex(s => s.id === step.id);
  if (state === 'failed' && stepIdx === currentIdx) return 'failed';
  if (state === 'completed') return 'done';
  if (stepIdx < currentIdx) return 'done';
  if (stepIdx === currentIdx) return 'active';
  return 'pending';
}

export const ExecutionTracker: React.FC<ExecutionTrackerProps> = ({
  taskId, queue, worker, steps, currentStepId, events, elapsedSec, state, error, title,
}) => {
  const logRef = useRef<HTMLDivElement>(null);
  const [copied, setCopied] = useState(false);
  const [, tick] = useState(0);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [events.length]);

  // Re-render timestamps. While running, tick every 1s so the active-step
  // duration counts up in real time; once idle/done, slow down to 5s — that's
  // only used by event "X ago" timestamps so 5s is plenty.
  useEffect(() => {
    const isRunningNow = state === 'running' || state === 'queued';
    const id = setInterval(() => tick(n => n + 1), isRunningNow ? 1000 : 5000);
    return () => clearInterval(id);
  }, [state]);

  const stepDurations = computeStepDurations(steps, events, currentStepId, state, Date.now());

  const copy = () => {
    if (taskId) {
      navigator.clipboard.writeText(taskId).then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      });
    }
  };

  const isRunning = state === 'running' || state === 'queued';

  return (
    <div className="flex flex-col gap-3 h-full">
      {/* Header: Redis + task metadata */}
      <div className="rounded-xl border border-theme bg-[var(--bg-tertiary)] px-4 py-3 flex flex-col gap-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-2">
            {/* Redis indicator */}
            <span className="flex items-center gap-1.5 text-xs font-mono font-bold">
              <span className={`w-2 h-2 rounded-full ${isRunning ? 'bg-[#4ade80] animate-pulse' : state === 'completed' ? 'bg-[#4ade80]' : state === 'failed' ? 'bg-red-400' : 'bg-gray-500'}`} />
              <span className="text-[#4ade80]">Redis</span>
            </span>
            <span className="text-theme-muted text-xs">broker</span>
            {queue === 'document_processing' && (
              <>
                <span className="text-theme-muted text-xs">·</span>
                <span className="flex items-center gap-1.5 text-xs font-mono font-bold px-2 py-0.5 rounded-full bg-[#0ea5e9]/10 border border-[#0ea5e9]/30 text-[#38bdf8]">
                  ⬡ LangGraph
                </span>
              </>
            )}
          </div>

          {/* Large elapsed-time readout: front-and-centre so the user can
              see at a glance how long the run has been processing.
              MM:SS under 1h, H:MM:SS once we cross an hour. Pulses while
              running, locks once completed/failed. */}
          <div className="flex items-center gap-3 ml-auto">
            <div
              className={`font-mono tabular-nums text-2xl font-bold tracking-tight leading-none ${
                isRunning ? 'text-[#FF8C66]' : state === 'completed' ? 'text-[#4ade80]' : state === 'failed' ? 'text-red-400' : 'text-theme-muted'
              }`}
              title="Total elapsed time since the task was submitted"
              aria-label={`Elapsed time ${fmtClock(elapsedSec)}`}
            >
              {fmtClock(elapsedSec)}
            </div>
            {/* Task ID */}
            {taskId && (
              <button
                onClick={copy}
                title="Copy task ID"
                className="flex items-center gap-1.5 font-mono text-xs bg-[var(--bg-input)] border border-theme rounded-lg px-2 py-1 text-theme-secondary hover:text-theme-primary hover:border-[#FF8C66] transition-colors"
              >
                <span className="text-theme-muted">task:</span>
                <span>{taskId.slice(0, 8)}…</span>
                <span className="text-theme-muted text-[10px]">{copied ? '✓' : '📋'}</span>
              </button>
            )}
          </div>
        </div>

        <div className="flex flex-wrap gap-2 text-[10px] font-mono">
          <span className="px-2 py-0.5 rounded-full bg-[#7C3AED]/20 text-[#a78bfa] border border-[#7C3AED]/30">
            queue: {queue}
          </span>
          {worker && (
            <span className="px-2 py-0.5 rounded-full bg-[#FF8C66]/10 text-[#FF8C66] border border-[#FF8C66]/20">
              worker: {worker.replace(/^celery@/, '')}
            </span>
          )}
        </div>

        {title && <p className="text-[10px] text-theme-muted">{title}</p>}
      </div>

      {/* Pipeline steps strip */}
      <div className="rounded-xl border border-theme bg-[var(--bg-tertiary)] px-4 py-4">
        <p className="text-[10px] font-semibold text-theme-muted uppercase tracking-widest mb-4">
          {queue === 'document_processing' ? 'LangGraph nodes' : 'Celery pipeline'}
        </p>
        <div className="flex items-start justify-between relative">
          {/* connector line */}
          <div className="absolute top-4 left-0 right-0 h-px bg-[var(--bg-input)] mx-4" />

          {steps.map((step, i) => {
            const status = stepStatus(step, currentStepId, steps, state);
            return (
              <div key={step.id} className="flex flex-col items-center gap-1 relative flex-1" style={{ zIndex: 1 }}>
                {/* Node circle */}
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-sm border-2 transition-all duration-300 ${
                    status === 'done'    ? 'bg-[#4ade80]/20 border-[#4ade80] text-[#4ade80]'
                    : status === 'active'  ? 'bg-[#FF8C66]/20 border-[#FF8C66] text-[#FF8C66] animate-pulse'
                    : status === 'failed'  ? 'bg-red-500/20 border-red-400 text-red-400'
                    : 'bg-[var(--bg-input)] border-theme text-theme-muted'
                  }`}
                >
                  {status === 'done'   ? '✓'
                  : status === 'failed' ? '✕'
                  : step.icon}
                </div>
                {/* Label */}
                <span className={`text-[9px] font-mono text-center leading-tight max-w-[52px] ${
                  status === 'active' ? 'text-[#FF8C66] font-bold'
                  : status === 'done' ? 'text-[#4ade80]'
                  : status === 'failed' ? 'text-red-400'
                  : 'text-theme-muted'
                }`}>
                  {step.label}
                </span>
                {/* Per-step duration: active step counts up live, completed
                    steps freeze at (next-step-start - this-step-start). Lets
                    the user see *which* step is consuming wall-clock time. */}
                {stepDurations[step.id] !== undefined && (
                  <span className={`text-[9px] font-mono tabular-nums ${
                    status === 'active' ? 'text-[#FF8C66]'
                    : status === 'done' ? 'text-theme-muted'
                    : 'text-theme-muted'
                  }`}>
                    {fmtElapsed(stepDurations[step.id])}
                  </span>
                )}
              </div>
            );
          })}
        </div>

        {error && (
          <p className="mt-3 text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
            {error}
          </p>
        )}
      </div>

      {/* Event log */}
      <div className="flex-1 rounded-xl border border-theme bg-[var(--bg-tertiary)] flex flex-col min-h-0">
        <div className="flex items-center justify-between px-4 py-2 border-b border-theme">
          <span className="text-[10px] font-semibold text-theme-muted uppercase tracking-widest">Events</span>
          <span className={`text-[10px] font-mono ${isRunning ? 'text-[#FF8C66] animate-pulse' : 'text-theme-muted'}`}>
            {isRunning ? '● LIVE' : state === 'completed' ? '✓ DONE' : state === 'failed' ? '✕ FAILED' : ''}
          </span>
        </div>
        <div ref={logRef} className="flex-1 overflow-y-auto px-3 py-2 space-y-1 font-mono text-[11px]" style={{ maxHeight: '220px' }}>
          {events.length === 0 ? (
            <p className="text-theme-muted py-4 text-center text-xs">Waiting for task…</p>
          ) : (
            events.map((ev, i) => (
              <div key={i} className="flex items-baseline gap-2 py-0.5">
                <span className="text-theme-muted shrink-0 text-[10px]">
                  {new Date(ev.ts).toLocaleTimeString('en-GB', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                </span>
                <span className={`shrink-0 font-semibold ${
                  ev.stage === 'completed' || ev.stage === 'store_result' ? 'text-[#4ade80]'
                  : ev.stage === 'failed' ? 'text-red-400'
                  : ev.stage === 'queued' ? 'text-[#60a5fa]'
                  : 'text-[#FF8C66]'
                }`}>
                  {ev.stage}
                </span>
                {ev.pct > 0 && ev.pct < 100 && (
                  <span className="text-theme-muted">{ev.pct}%</span>
                )}
                {ev.worker && (
                  <span className="text-theme-muted text-[10px] truncate">
                    via {ev.worker.replace(/^celery@/, '')}
                  </span>
                )}
                {ev.message && ev.message !== ev.stage && (
                  <span className="text-theme-muted text-[10px] truncate">{ev.message}</span>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};
