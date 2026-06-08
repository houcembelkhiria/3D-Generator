import React, { useState, useEffect, useCallback, useRef } from 'react';
import { PipelineStep, GenerationMethod, AssetMetadata, ProcessLog, SystemStatus, AppView, GeneratedModel } from './types';
import { PipelineVisualizer } from './components/PipelineVisualizer';
import { Terminal } from './components/Terminal';
import { StatusBadge } from './components/StatusBadge';
import { Sidebar } from './components/Sidebar';
import { FilesTreatment } from './components/FilesTreatment';
import { ThemeToggle } from './components/ThemeToggle';
import { ImageTo3D } from './components/ImageTo3D';
import { TextTo3D } from './components/TextTo3D';
import { MultiViewTo3D } from './components/MultiViewTo3D';
import { ModelGallery } from './components/ModelGallery';
import { API_BASE } from './api';
import { IconUpload, IconBox, IconCpu, IconDatabase, IconSettings, IconActivity, IconCheckCircle, IconMessageSquare, IconPaperclip, IconX, IconFileText } from './components/Icons';

// Mock Data for "Step B: Extraction"
const MOCK_EXTRACTED_METADATA_PROCEDURAL: AssetMetadata = {
  name: "SciFi_Crate_01",
  type: "Prop",
  transform: { position: { x: 0, y: 1.5, z: 5 }, rotation: { x: 0, y: 45, z: 0 }, scale: { x: 1, y: 1, z: 1 } },
  physics: { mass: 50, isKinematic: false, colliderType: "Box" },
  material: { color: "#3B82F6", metallic: 0.8, smoothness: 0.4 },
  generationMethod: GenerationMethod.PROCEDURAL
};

const MOCK_EXTRACTED_METADATA_VISUAL: AssetMetadata = {
  name: "Alien_Tree_Organic",
  type: "Prop",
  transform: { position: { x: 5, y: 0, z: 5 }, rotation: { x: 0, y: 0, z: 0 }, scale: { x: 2, y: 2, z: 2 } },
  physics: { mass: 100, isKinematic: true, colliderType: "Mesh" },
  material: { color: "#10B981", metallic: 0.1, smoothness: 0.2 },
  generationMethod: GenerationMethod.VISUAL
};

export default function App() {
  const [activeView, setActiveView] = useState<AppView>('agent');
  const [generatedModels, setGeneratedModels] = useState<GeneratedModel[]>([]);
  // Keep generation views mounted once visited so in-progress jobs survive tab switches
  const [mountedViews, setMountedViews] = useState<Set<AppView>>(() => new Set([activeView]));
  const [spawnMsg, setSpawnMsg] = useState<string | null>(null);
  const spawnTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [currentStep, setCurrentStep] = useState<PipelineStep>(PipelineStep.IDLE);
  const [logs, setLogs] = useState<ProcessLog[]>([]);
  const [metadata, setMetadata] = useState<AssetMetadata | null>(null);
  const [promptText, setPromptText] = useState('');
  const [files, setFiles] = useState<string[]>([]);
  const [extractedText, setExtractedText] = useState<string>('');
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    device: 'cpu',
    ramUsage: 0,
    ramTotal: 0,
    vramUsage: 0,
    vramTotal: 0,
    hunyuan3dReady: false,
    hasTexgen: false,
    hasT2i: false,
    hasMv: false,
  });

  const addLog = useCallback((message: string, type: ProcessLog['type'] = 'info') => {
    const newLog: ProcessLog = {
      id: Math.random().toString(36).substring(7),
      timestamp: new Date().toLocaleTimeString('fr-FR', { hour12: false, fractionalSecondDigits: 2 } as any),
      step: currentStep,
      message,
      type
    };
    const MAX_LOGS = 500;
    setLogs(prev => {
      const updated = [...prev, newLog];
      return updated.length > MAX_LOGS ? updated.slice(-MAX_LOGS) : updated;
    });
  }, [currentStep]);

  const handleTextExtracted = useCallback((text: string) => {
    setExtractedText(text);
    addLog(`Text extracted from file. Length: ${text.length} characters.`, 'success');
  }, [addLog]);

  const clearLogs = useCallback(() => {
    setLogs([]);
    addLog("Logs cleared", 'info');
  }, [addLog]);

  // Poll real system stats from Backend
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/system-stats`);
        if (res.ok) {
          const data = await res.json();
          setSystemStatus({
            device: data.device,
            ramUsage: data.ram_used_gb,
            ramTotal: data.ram_total_gb,
            vramUsage: data.vram_used_gb,
            vramTotal: data.vram_total_gb,
            hunyuan3dReady: data.hunyuan3d_ready,
            hasTexgen: data.has_texgen,
            hasT2i: data.has_t2i,
            hasMv: data.has_mv,
          });
        }
      } catch { /* Backend not reachable */ }
    };
    fetchStats();
    const interval = setInterval(fetchStats, 10000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    setMountedViews(prev => { prev.add(activeView); return new Set(prev); });
  }, [activeView]);

  // Load gallery from disk (the actual source of truth — generated/3d_outputs/*.glb).
  // We also probe the vector cache for richer metadata (prompt, source) and merge
  // it on top, but disk is the canonical list. Disk listing is loaded fresh on
  // every call, so newly-added GLBs (including from this session) are picked up.
  const fetchGallery = useCallback(async () => {
    let diskModels: GeneratedModel[] = [];
    try {
      const diskRes = await fetch(`${API_BASE}/api/v1/generated-models`);
      if (diskRes.ok) {
        const diskData = await diskRes.json();
        diskModels = (diskData.models ?? []).map((m: any) => ({
          id: m.uid ?? m.filename ?? crypto.randomUUID(),
          previewUrl: m.preview_url?.startsWith('http')
            ? m.preview_url
            : `${API_BASE}${m.preview_url ?? ''}`,
          downloadUrl: m.download_url?.startsWith('http')
            ? m.download_url
            : `${API_BASE}${m.download_url ?? ''}`,
          format: m.format || (m.filename?.split('.').pop() ?? 'glb'),
          source: (m.source as GeneratedModel['source']) ?? 'image-to-3d',
          prompt: m.prompt || undefined,
          createdAt: m.created_at ?? (typeof m.created === 'number'
            ? new Date(m.created * 1000).toISOString()
            : new Date().toISOString()),
          generationTime: m.generation_time ?? undefined,
          faceCount: m.face_count ?? undefined,
          fileSizeMb: m.file_size_mb ?? undefined,
        }));
      }
    } catch { /* backend unreachable; treat disk as empty */ }

    // Best-effort enrichment from the vector cache (prompt, source, generation_time).
    // The cache may be empty or stale — that's fine, disk is still the truth.
    const meta = new Map<string, Partial<GeneratedModel>>();
    try {
      const cacheRes = await fetch(`${API_BASE}/api/v1/cache-stats`);
      if (cacheRes.ok) {
        const cacheData = await cacheRes.json();
        for (const m of (cacheData.models ?? [])) {
          // Cache id is an opaque string; line it up with disk by matching the
          // GLB filename that's embedded in preview_url (last path segment minus extension).
          const url: string = m.previewUrl ?? '';
          const stem = url.split('/').pop()?.replace(/\.glb$/i, '');
          if (!stem) continue;
          meta.set(stem, {
            source: m.source,
            prompt: m.prompt,
            attempt: m.attempt,
            generationTime: m.generationTime,
            fromCache: true,
          });
        }
      }
    } catch { /* enrichment is optional */ }

    if (diskModels.length === 0 && meta.size > 0) {
      // No disk files but cache has entries — fall back to cache so the user
      // at least sees something. URLs may be broken, but better than empty.
      const fromCacheOnly = Array.from(meta.entries()).map(([stem, m]) => ({
        id: stem,
        previewUrl: `${API_BASE}/api/v1/outputs/${stem}.glb`,
        downloadUrl: `${API_BASE}/api/v1/outputs/${stem}.glb`,
        format: 'glb',
        source: (m.source as GeneratedModel['source']) ?? 'image-to-3d',
        createdAt: new Date().toISOString(),
        ...m,
      } as GeneratedModel));
      setGeneratedModels(fromCacheOnly);
      return;
    }

    const merged = diskModels.map(d => {
      const enrich = meta.get(d.id) ?? {};
      return { ...d, ...enrich };
    });
    setGeneratedModels(prev => {
      // Preserve any in-session items that aren't on disk yet (very fresh generations
      // where the export thread hasn't flushed). Match by id; disk wins on conflict.
      const onDisk = new Set(merged.map(m => m.id));
      const inSessionOnly = prev.filter(m => !onDisk.has(m.id));
      return [...inSessionOnly, ...merged];
    });
  }, []);

  // Initial fetch on mount.
  useEffect(() => { fetchGallery(); }, [fetchGallery]);

  // Refetch when the backend transitions to ready — handles the page-load-before-
  // backend-ready race that previously left the gallery stuck empty for the session.
  const wasReadyRef = useRef(false);
  useEffect(() => {
    if (systemStatus.hunyuan3dReady && !wasReadyRef.current) {
      wasReadyRef.current = true;
      fetchGallery();
    }
  }, [systemStatus.hunyuan3dReady, fetchGallery]);

  const addMockFile = () => {
    const mockFiles = ["specifications_v1.pdf", "asset_reference.jpg", "mechanics.xml"];
    const randomFile = mockFiles[Math.floor(Math.random() * mockFiles.length)];
    if (!files.includes(randomFile)) {
      setFiles([...files, randomFile]);
    }
  };

  const removeFile = (index: number) => {
    const newFiles = [...files];
    newFiles.splice(index, 1);
    setFiles(newFiles);
  };

  const handleProcess = (method: GenerationMethod) => {
    if (currentStep !== PipelineStep.IDLE && currentStep !== PipelineStep.COMPLETED && currentStep !== PipelineStep.ERROR) return;

    // Check if we have extracted text from PDF or user prompt
    const hasValidInput = extractedText.trim() || promptText.trim() || files.length > 0;

    if (!hasValidInput) {
      addLog("Erreur: Veuillez entrer un prompt, extraire du texte d'un PDF ou attacher un fichier.", 'error');
      return;
    }

    // Reset
    setCurrentStep(PipelineStep.INGESTION);
    setLogs([]);
    setMetadata(null);
    setSystemStatus(prev => ({ ...prev, activeWorkers: 1 }));

    addLog(`🚀 Architecture PFE chargée. Pipeline démarré pour méthode: ${method}`, 'success');

    // Ingestion Logic - prioritize extracted text from PDF
    if (extractedText.trim()) {
      addLog(`Ingestion de texte extrait d'un PDF (${extractedText.length} caractères)...`, 'info');
      addLog("Contenu: \"" + extractedText.substring(0, 40) + (extractedText.length > 40 ? '...' : '') + "\"", 'info');
    } else if (files.length > 0) {
       addLog(`Ingestion de ${files.length} fichier(s) attaché(s)...`, 'info');
       files.forEach(f => addLog(`Lecture: ${f}`, 'info'));
    }

    if (promptText.trim() && !extractedText.trim()) {
       addLog("Ingestion du prompt textuel utilisateur...", 'info');
       addLog(`Contenu: "${promptText.substring(0, 40)}${promptText.length > 40 ? '...' : ''}"`, 'info');
    }

    // STEP A: Ingestion
    setTimeout(() => {
      addLog("Bibliothèque 'unstructured' activée.", 'info');
      addLog("Nettoyage des données & parsing sémantique.", 'info');

      setCurrentStep(PipelineStep.EXTRACTION);

      // STEP B: Brain (LLM)
      setTimeout(() => {
        addLog("Chargement context Llama 3 8B (vLLM local)...", 'warning');
        addLog(`VRAM Spike détecté: ${(systemStatus.vramUsage + 4).toFixed(1)}GB`, 'warning');
        addLog("Extraction entités nommées (NER) en cours...", 'info');
        addLog("Validation Pydantic schema 'UnityMetadata'...", 'info');
        addLog("JSON Metadata généré avec succès.", 'success');

        const extracted = method === GenerationMethod.PROCEDURAL ? MOCK_EXTRACTED_METADATA_PROCEDURAL : MOCK_EXTRACTED_METADATA_VISUAL;
        setMetadata(extracted);

        setCurrentStep(PipelineStep.GENERATION);

        // STEP C: Generation
        setTimeout(() => {
          if (method === GenerationMethod.PROCEDURAL) {
            addLog("Mode Procédural détecté (Objets techniques).", 'info');
            addLog("Appel Qwen 2.5 Coder...", 'info');
            addLog("Génération script C# 'RuntimeSpawner.cs'...", 'info');
            addLog("Compilation C# syntax check OK.", 'success');
          } else {
            addLog("Mode Visuel détecté (Objets organiques).", 'info');
            addLog("Appel TripoSR Text-to-3D...", 'info');
            addLog("Génération maillage .GLB en cours (Celery Worker 4)...", 'info');
            addLog("Texture baking terminé.", 'success');
          }

          setCurrentStep(PipelineStep.MCP_DISPATCH);

          // STEP D: MCP
          setTimeout(() => {
            addLog("Connexion Client MCP (Python) → Serveur MCP (Unity Editor).", 'warning');
            addLog(`Envoi commande: call_tool("SpawnAsset", { name: "${extracted.name}" })`, 'info');
            addLog("Unity Server a acquitté la réception.", 'success');
            addLog("Objet instancié dans la scène Active.", 'success');

            setCurrentStep(PipelineStep.COMPLETED);
            setSystemStatus(prev => ({ ...prev, activeWorkers: 0 }));
            addLog("Pipeline terminé. En attente de nouvelle tâche.", 'info');
          }, 2500);

        }, 3000);

      }, 2500);

    }, 2000);
  };



  const getPageTitle = () => {
    switch(activeView) {
      case 'agent': return 'Agent Operations';
      case 'files': return 'Files Treatment';
      case 'settings': return 'System Configuration';
      case 'image-to-3d': return 'Image to 3D';
      case 'text-to-3d': return 'Text to 3D';
      case 'multiview-to-3d': return 'Multi-View to 3D';
      case 'gallery': return 'Model Gallery';
      default: return 'Agent Operations';
    }
  };

  const handleModelGenerated = useCallback((model: GeneratedModel) => {
    setGeneratedModels(prev => {
      // Avoid duplicates if cache already has this model
      const filtered = prev.filter(m => m.id !== model.id);
      return [model, ...filtered];
    });
  }, []);

  const handleModelRemove = useCallback((id: string) => {
    setGeneratedModels(prev => prev.filter(m => m.id !== id));
    fetch(`${API_BASE}/api/v1/models/${id}`, { method: 'DELETE' }).catch(() => {});
  }, []);

  const isProcessing = currentStep !== PipelineStep.IDLE && currentStep !== PipelineStep.COMPLETED && currentStep !== PipelineStep.ERROR;

  return (
    <div className="min-h-screen flex bg-theme-primary text-theme-primary overflow-hidden">
      <Sidebar activeView={activeView} onViewChange={setActiveView} />

      <div className="flex-1 flex flex-col h-screen overflow-hidden">
        {/* HEADER */}
        <header className="flex flex-col md:flex-row justify-between items-start md:items-center bg-theme-secondary p-6 border-b border-theme backdrop-blur-sm z-10">
          <div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-[#FF8C66] via-[#FF5F6D] to-[#7C3AED] bg-clip-text text-transparent">
              {getPageTitle()}
            </h1>
            <p className="text-theme-muted mt-1 font-mono text-xs">
              Microservice d'IA Agentique &rarr; Unity MCP
            </p>
          </div>
          <div className="flex flex-wrap gap-3 mt-4 md:mt-0 items-center">
            <ThemeToggle />
            <StatusBadge
              label="RAM"
              status={systemStatus.ramUsage > systemStatus.ramTotal * 0.85 ? 'busy' : 'online'}
              value={`${systemStatus.ramUsage} / ${systemStatus.ramTotal} GB`}
            />
            <StatusBadge
              label={systemStatus.device === 'mps' ? 'MPS' : systemStatus.device === 'cuda' ? 'VRAM' : 'CPU'}
              status={systemStatus.vramUsage > systemStatus.vramTotal * 0.85 ? 'busy' : 'online'}
              value={systemStatus.vramTotal > 0 ? `${systemStatus.vramUsage} / ${systemStatus.vramTotal} GB` : 'N/A'}
            />
            <StatusBadge
              label="Hunyuan3D"
              status={systemStatus.hunyuan3dReady ? 'online' : 'offline'}
              value={systemStatus.hunyuan3dReady ? 'READY' : 'LOADING'}
            />
          </div>
        </header>

        {/* CONTENT SCROLLABLE AREA */}
        <main className="flex-1 overflow-y-auto p-6 scrollbar-hide bg-theme-primary">

          {/* VIEW: AGENT */}
          {activeView === 'agent' && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 pb-6">
              {/* LEFT COLUMN: CONTROL & INPUT */}
              <section className="lg:col-span-3 space-y-6">
                <div className="card p-6 h-full flex flex-col">
                  <h2 className="text-lg font-bold text-heading mb-4 flex items-center">
                    <IconMessageSquare className="mr-2 text-[#FF8C66]" /> Input Request
                  </h2>

                  <div className="flex-1 flex flex-col justify-end">
                    <p className="text-muted text-sm mb-4">
                      Describe the asset you want to generate or attach technical specifications (PDF, EML).
                    </p>

                    {/* Chat Input Container */}
                    <div className={`input-container transition-all ${isProcessing ? 'opacity-50' : 'focus-within:ring-1 focus-within:ring-[#FF8C66] focus-within:border-[#FF8C66]'}`}>

                        {/* Attached Files Display */}
                        {files.length > 0 && (
                          <div className="px-3 pt-3 flex flex-wrap gap-2">
                            {files.map((f, i) => (
                              <div key={i} className="flex items-center gap-2 bg-theme-input px-2 py-1 rounded-md text-xs text-theme-secondary border border-theme-secondary animate-fadeIn">
                                <IconFileText className="w-3 h-3 text-[#7C3AED]" />
                                <span className="max-w-[150px] truncate">{f}</span>
                                <button onClick={() => removeFile(i)} className="hover:text-red-400 transition-colors" disabled={isProcessing}>
                                  <IconX className="w-3 h-3"/>
                                </button>
                              </div>
                            ))}
                          </div>
                        )}

                        <textarea
                          className="w-full bg-transparent border-none text-sm text-theme-secondary placeholder-text-muted focus:ring-0 p-3 resize-none"
                          rows={4}
                          placeholder="Type your prompt here... (e.g. 'A futuristic vending machine')"
                          value={promptText}
                          onChange={(e) => setPromptText(e.target.value)}
                          disabled={isProcessing}
                        />

                        {/* Toolbar */}
                        <div className="flex items-center justify-between px-2 pb-2 border-t border-theme pt-2 mx-2">
                          <div className="flex gap-1">
                              <button
                                onClick={addMockFile}
                                disabled={isProcessing}
                                className="p-2 text-theme-muted hover:text-theme-secondary hover:bg-theme-input rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                                title="Attach File"
                              >
                                <IconPaperclip className="w-5 h-5" />
                              </button>
                          </div>

                          <div className="flex gap-2">
                              <button
                                onClick={() => handleProcess(GenerationMethod.PROCEDURAL)}
                                disabled={isProcessing}
                                className="px-3 py-1.5 bg-[#FF8C66] hover:bg-[#ff7a4d] disabled:opacity-50 disabled:cursor-not-allowed text-black rounded-lg text-xs font-bold transition-all shadow-lg shadow-[#FF8C66]/10 flex items-center gap-1"
                              >
                                <IconCpu className="w-3 h-3" />
                                Code
                              </button>
                              <button
                                onClick={() => handleProcess(GenerationMethod.VISUAL)}
                                disabled={isProcessing}
                                className="px-3 py-1.5 bg-[#7C3AED] hover:bg-[#6d28d9] disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg text-xs font-bold transition-all shadow-lg shadow-[#7C3AED]/10 flex items-center gap-1"
                              >
                                <IconBox className="w-3 h-3" />
                                Visual
                              </button>
                          </div>
                        </div>
                    </div>

                    <div className="mt-4 text-[10px] text-theme-muted font-mono text-center">
                      AI Model: Llama-3-8B-Instruct (Local)
                    </div>
                  </div>
                </div>
              </section>

              {/* MIDDLE COLUMN: PIPELINE VIZ */}
              <section className="lg:col-span-5 space-y-6 flex flex-col">
                <div className="card p-6">
                  <h2 className="text-lg font-bold text-heading mb-6 flex items-center justify-between">
                    <span>Pipeline Status</span>
                    {currentStep !== PipelineStep.IDLE && currentStep !== PipelineStep.COMPLETED && (
                      <span className="text-xs font-mono text-[#FF8C66] animate-pulse">PROCESSING...</span>
                    )}
                  </h2>
                  <PipelineVisualizer currentStep={currentStep} generationMethod={metadata?.generationMethod} />
                </div>

                <div className="flex-1 min-h-[300px]">
                  <Terminal logs={logs} onClear={clearLogs} />
                </div>
              </section>

              {/* RIGHT COLUMN: PREVIEW */}
              <section className="lg:col-span-4 space-y-6">
                <div className="card p-6 h-full flex flex-col">
                  <h2 className="text-lg font-bold text-heading mb-4 flex items-center">
                    <IconDatabase className="mr-2 text-[#7C3AED]" /> Extracted Metadata
                  </h2>

                  {metadata ? (
                    <div className="flex-1 space-y-4">
                      <div className="bg-theme-input p-4 rounded-xl border border-theme font-mono text-xs text-[#7C3AED] overflow-x-auto">
                          <pre>{JSON.stringify(metadata, null, 2)}</pre>
                      </div>

                      <div className="bg-theme-input p-4 rounded-xl border border-theme flex flex-col items-center">
                          <div className="text-xs text-muted mb-2 w-full text-left uppercase tracking-wider">Asset Preview</div>
                          <div className="w-full aspect-square bg-theme-secondary rounded-lg flex items-center justify-center border border-theme relative overflow-hidden group">
                            <img
                              src={`https://picsum.photos/400/400?random=${metadata.name}`}
                              alt="Asset Preview"
                              className="opacity-50 grayscale group-hover:grayscale-0 transition-all duration-500"
                            />
                            <div className="absolute inset-0 flex items-center justify-center">
                                <span className="px-3 py-1 bg-theme-primary/80 text-theme-primary text-xs rounded border border-theme backdrop-blur-md">
                                  {metadata.generationMethod === GenerationMethod.PROCEDURAL ? 'C# SCRIPT GENERATED' : '.GLB MESH GENERATED'}
                                </span>
                            </div>
                          </div>
                      </div>
                    </div>
                  ) : (
                    <div className="flex-1 flex flex-col items-center justify-center text-muted">
                      <IconDatabase className="w-16 h-16 mb-4 opacity-20" />
                      <p>No metadata extracted yet.</p>
                      <p className="text-sm">Waiting for pipeline Step B...</p>
                    </div>
                  )}
                </div>
              </section>
            </div>
          )}

          {/* VIEW: FILES TREATMENT */}
          {activeView === 'files' && (
            <FilesTreatment onTextExtracted={handleTextExtracted} onModelGenerated={handleModelGenerated} />
          )}

          {/* VIEW: SETTINGS */}
          {activeView === 'settings' && (
            <div className="max-w-4xl mx-auto space-y-8">

              {/* SYSTEM RESOURCES CARD */}
              <div className="card overflow-hidden">
                <div className="p-6 border-b border-theme">
                  <h2 className="text-xl font-bold text-heading flex items-center">
                    <IconSettings className="mr-3 text-[#FF8C66]" /> System Resources
                  </h2>
                  <p className="text-body mt-1">Manage local computation resources and limits.</p>
                </div>
                <div className="p-8 space-y-8">
                  <div className="space-y-4">
                      <div className="flex justify-between items-end">
                        <label className="text-sm font-semibold text-theme-secondary">Total GPU VRAM Limit</label>
                        <span className="text-2xl font-mono text-[#FF8C66] font-bold">{systemStatus.vramTotal} GB</span>
                      </div>
                      <input
                          type="range"
                          min="8" max="48" step="4"
                          value={systemStatus.vramTotal}
                          onChange={() => {}}
                          className="w-full h-3 bg-theme-input rounded-lg appearance-none cursor-pointer accent-[#FF8C66]"
                      />
                      <div className="flex justify-between text-xs text-muted font-mono">
                        <span>8 GB (Minimum)</span>
                        <span>48 GB (Dual A10)</span>
                      </div>
                      <p className="text-xs text-muted bg-[#FF8C66]/10 p-3 rounded border border-[#FF8C66]/20">
                        ⚠️ Increasing VRAM limit allows larger LLM context windows (e.g. Llama 3 70B) but requires hardware support.
                      </p>
                  </div>
                </div>
              </div>

              {/* CONNECTIVITY CARD */}
              <div className="card overflow-hidden">
                <div className="p-6 border-b border-theme">
                   <h2 className="text-xl font-bold text-heading flex items-center">
                    <IconActivity className="mr-3 text-[#FF8C66]" /> Service Connectivity
                  </h2>
                   <p className="text-body mt-1">Live status of backend capabilities.</p>
                </div>
                <div className="p-8 grid grid-cols-1 md:grid-cols-2 gap-8">
                   {/* Texture Generation */}
                   <div className="toggle-card">
                      <div className="flex items-center justify-between">
                          <span className="font-semibold text-heading">Texture Generation</span>
                          <div
                              className={`w-14 h-8 rounded-full p-1 transition-colors duration-300 ease-in-out ${systemStatus.hasTexgen ? 'bg-[#7C3AED]' : 'bg-[var(--bg-input)]'}`}
                              aria-label="Texture generation status"
                          >
                              <div className={`w-6 h-6 rounded-full bg-white shadow-sm transition-transform duration-300 ${systemStatus.hasTexgen ? 'translate-x-6' : 'translate-x-0'}`} />
                          </div>
                      </div>
                      <p className="text-xs text-muted">
                        Handles asynchronous job processing for 3D generation (TripoSR/Shap-E). Essential for non-blocking API responses.
                      </p>
                      <div className="flex items-center text-xs">
                        Status:
                        <span className={`ml-2 px-2 py-0.5 rounded ${systemStatus.hasTexgen ? 'bg-[#7C3AED]/20 text-[#7C3AED]' : 'bg-red-500/20 text-red-400'}`}>
                           {systemStatus.hasTexgen ? 'ENABLED' : 'DISABLED'}
                        </span>
                      </div>
                   </div>

                   {/* Multi-View */}
                   <div className="toggle-card">
                      <div className="flex items-center justify-between">
                          <span className="font-semibold text-heading">Multi-View Mode</span>
                          <div
                              className={`w-14 h-8 rounded-full p-1 transition-colors duration-300 ease-in-out ${systemStatus.hasMv ? 'bg-[#7C3AED]' : 'bg-[var(--bg-input)]'}`}
                              aria-label="Multi-view mode status"
                          >
                              <div className={`w-6 h-6 rounded-full bg-white shadow-sm transition-transform duration-300 ${systemStatus.hasMv ? 'translate-x-6' : 'translate-x-0'}`} />
                          </div>
                      </div>
                      <p className="text-xs text-muted">
                        Model Context Protocol connection to Unity Editor. Enables direct scene manipulation tools (Spawn, Transform).
                      </p>
                      <div className="flex items-center text-xs">
                        Status:
                        <span className={`ml-2 px-2 py-0.5 rounded ${systemStatus.hasMv ? 'bg-[#7C3AED]/20 text-[#7C3AED]' : 'bg-red-500/20 text-red-400'}`}>
                           {systemStatus.hasMv ? 'ENABLED' : 'DISABLED'}
                        </span>
                      </div>
                   </div>
                </div>
              </div>

            </div>
          )}


          {/* VIEW: IMAGE TO 3D — kept mounted after first visit so jobs survive tab switches */}
          {mountedViews.has('image-to-3d') && (
            <div className={activeView === 'image-to-3d' ? '' : 'hidden'}>
              <ImageTo3D onModelGenerated={handleModelGenerated} />
            </div>
          )}

          {/* VIEW: TEXT TO 3D */}
          {mountedViews.has('text-to-3d') && (
            <div className={activeView === 'text-to-3d' ? '' : 'hidden'}>
              <TextTo3D onModelGenerated={handleModelGenerated} />
            </div>
          )}

          {/* VIEW: MULTI-VIEW TO 3D */}
          {mountedViews.has('multiview-to-3d') && (
            <div className={activeView === 'multiview-to-3d' ? '' : 'hidden'}>
              <MultiViewTo3D onModelGenerated={handleModelGenerated} />
            </div>
          )}

          {/* VIEW: MODEL GALLERY */}
          {activeView === 'gallery' && (
            <ModelGallery
              models={generatedModels}
              onRemove={handleModelRemove}
              onSpawn={() => {
                if (spawnTimerRef.current) clearTimeout(spawnTimerRef.current);
                setSpawnMsg('Sent to Unity — the model is downloading and importing. This can take 10–30 s; check the Unity Console if it doesn\'t appear.');
                spawnTimerRef.current = setTimeout(() => setSpawnMsg(null), 12000);
              }}
            />
          )}

        </main>

      {/* Global spawn toast — persists across tab switches */}
      {spawnMsg && (
        <div className="fixed top-6 right-6 z-50 flex items-start gap-3 max-w-sm w-full px-4 py-3 rounded-xl border border-sky-500/40 bg-[var(--bg-card)] shadow-2xl text-sky-300 text-sm">
          <svg className="animate-spin h-4 w-4 shrink-0 mt-0.5" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z"/>
          </svg>
          <span className="flex-1 leading-snug">{spawnMsg}</span>
          <button type="button" onClick={() => setSpawnMsg(null)} className="text-sky-400 hover:text-sky-200 shrink-0 text-xs mt-0.5">✕</button>
        </div>
      )}
      </div>
    </div>
  );
}
