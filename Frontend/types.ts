export enum PipelineStep {
  IDLE = 'IDLE',
  INGESTION = 'INGESTION',
  EXTRACTION = 'EXTRACTION',
  GENERATION = 'GENERATION',
  MCP_DISPATCH = 'MCP_DISPATCH',
  COMPLETED = 'COMPLETED',
  ERROR = 'ERROR'
}

export enum GenerationMethod {
  VISUAL = 'VISUAL', // Text-to-3D (TripoSR/Shap-E)
  PROCEDURAL = 'PROCEDURAL' // Code LLM (Qwen 2.5) -> C#
}

export interface UnityTransform {
  position: { x: number; y: number; z: number };
  rotation: { x: number; y: number; z: number };
  scale: { x: number; y: number; z: number };
}

export interface UnityMaterial {
  color: string;
  texture?: string;
  metallic?: number;
  smoothness?: number;
}

export interface AssetMetadata {
  name: string;
  type: 'Prop' | 'Structure' | 'Character' | 'Vehicle';
  transform: UnityTransform;
  physics: {
    mass: number;
    isKinematic: boolean;
    colliderType: 'Box' | 'Sphere' | 'Mesh' | 'Capsule';
  };
  material: UnityMaterial;
  generationMethod: GenerationMethod;
}

export interface ProcessLog {
  id: string;
  timestamp: string;
  step: PipelineStep;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
}

export interface SystemStatus {
  device: string;
  ramUsage: number;
  ramTotal: number;
  vramUsage: number;
  vramTotal: number;
  hunyuan3dReady: boolean;
  hasTexgen: boolean;
  hasT2i: boolean;
  hasMv: boolean;
}
// ---- Hunyuan3D API Types ----

export interface GeneratedModel {
  id: string;
  previewUrl: string;
  downloadUrl: string;
  format: string;
  source: 'image-to-3d' | 'text-to-3d' | 'multiview-to-3d';
  prompt?: string;
  createdAt: string;
  fromCache?: boolean;
  attempt?: number;
  generationTime?: number;
  faceCount?: number;
  fileSizeMb?: number;
  hasTexture?: boolean;
}

export type AppView = 'agent' | 'files' | 'settings' | 'image-to-3d' | 'text-to-3d' | 'multiview-to-3d' | 'gallery';
