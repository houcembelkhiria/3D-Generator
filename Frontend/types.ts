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
  gpuVramUsage: number; // in GB
  gpuTotalVram: number; // in GB
  redisConnected: boolean;
  unityMcpConnected: boolean;
  activeWorkers: number;
}