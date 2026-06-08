import { TrackerStep } from '../types';

export const CELERY_3D_STEPS: TrackerStep[] = [
  { id: 'queued',           label: 'Queued',      icon: '📬', description: 'Task submitted to Redis broker' },
  { id: 'received',         label: 'Worker',      icon: '⚙',  description: 'Celery worker received task' },
  { id: 'loading_model',    label: 'Load model',  icon: '🧠', description: 'Loading Hunyuan3D pipeline' },
  { id: 'generating_shape', label: 'Generate',    icon: '🗿', description: 'Running diffusion model' },
  { id: 'saving',           label: 'Saving',      icon: '💾', description: 'Writing output file' },
  { id: 'completed',        label: 'Done',        icon: '✓',  description: 'Generation complete' },
];

export const LANGGRAPH_STEPS: TrackerStep[] = [
  { id: 'parse_document',           label: 'Parse',       icon: '📄', description: 'Parsing PDF/EML document' },
  { id: 'validate_parsed_document', label: 'Validate',    icon: '✔',  description: 'Validating parsed content' },
  { id: 'spec_extraction',          label: 'LLM extract', icon: '🧠', description: 'Extracting 3D spec (LLM)' },
  { id: 'mesh_generation',          label: 'Generate 3D', icon: '🗿', description: 'Generating 3D mesh (Celery)' },
  { id: 'store_result',             label: 'Store',       icon: '💾', description: 'Persisting result' },
];

// Map a raw Celery stage name (from WS) to the matching CELERY_3D_STEPS id.
// WS emits: "queued" (from WS handler), "received", "loading_model",
// "generating_shape", "saving", "completed"
// The mapping is 1:1 — just return as-is, but normalise legacy names.
export function normaliseCeleryStage(stage: string): string {
  const map: Record<string, string> = {
    started: 'received',
    generating: 'generating_shape',
  };
  return map[stage] ?? stage;
}

// Map a LangGraph current_node string to a LANGGRAPH_STEPS id.
// Subgraph nodes arrive as "spec_extraction:extract_spec_llm" — use the prefix.
export function normaliseNodeToStep(node: string): string {
  const top = node.split(':')[0];
  // validate_parsed_document comes through as-is, others map directly
  return top;
}
