import React, { useState, useEffect, useCallback } from 'react';
import { PipelineStep, GenerationMethod, AssetMetadata, ProcessLog, SystemStatus } from './types';
import { PipelineVisualizer } from './components/PipelineVisualizer';
import { Terminal } from './components/Terminal';
import { StatusBadge } from './components/StatusBadge';
import { Sidebar } from './components/Sidebar';
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
  const [activeView, setActiveView] = useState<'dashboard' | 'settings'>('dashboard');
  const [currentStep, setCurrentStep] = useState<PipelineStep>(PipelineStep.IDLE);
  const [logs, setLogs] = useState<ProcessLog[]>([]);
  const [metadata, setMetadata] = useState<AssetMetadata | null>(null);
  const [promptText, setPromptText] = useState('');
  const [files, setFiles] = useState<string[]>([]);
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    gpuVramUsage: 14.2,
    gpuTotalVram: 24.0,
    redisConnected: true,
    unityMcpConnected: true,
    activeWorkers: 0
  });

  const addLog = useCallback((message: string, type: ProcessLog['type'] = 'info') => {
    const newLog: ProcessLog = {
      id: Math.random().toString(36).substring(7),
      timestamp: new Date().toLocaleTimeString('fr-FR', { hour12: false, fractionalSecondDigits: 2 } as any),
      step: currentStep,
      message,
      type
    };
    setLogs(prev => [...prev, newLog]);
  }, [currentStep]);

  // VRAM Fluctuation Simulation
  useEffect(() => {
    const interval = setInterval(() => {
      setSystemStatus(prev => ({
        ...prev,
        gpuVramUsage: Math.min(prev.gpuTotalVram, Math.max(8, prev.gpuVramUsage + (Math.random() - 0.5) * 0.5))
      }));
    }, 2000);
    return () => clearInterval(interval);
  }, []);

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

    if (!promptText.trim() && files.length === 0) {
      addLog("Erreur: Veuillez entrer un prompt ou attacher un fichier.", 'error');
      return;
    }

    // Reset
    setCurrentStep(PipelineStep.INGESTION);
    setLogs([]);
    setMetadata(null);
    setSystemStatus(prev => ({ ...prev, activeWorkers: 1 }));

    addLog(`🚀 Architecture PFE chargée. Pipeline démarré pour méthode: ${method}`, 'success');
    
    // Ingestion Logic
    if (files.length > 0) {
       addLog(`Ingestion de ${files.length} fichier(s) attaché(s)...`, 'info');
       files.forEach(f => addLog(`Lecture: ${f}`, 'info'));
    }
    
    if (promptText.trim()) {
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
        addLog(`VRAM Spike détecté: ${(systemStatus.gpuVramUsage + 4).toFixed(1)}GB`, 'warning');
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
            addLog("Connexion Client MCP (Python) -> Serveur MCP (Unity Editor).", 'warning');
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

  const updateSystemStatus = (key: keyof SystemStatus, value: any) => {
    setSystemStatus(prev => ({ ...prev, [key]: value }));
  };

  const getPageTitle = () => {
    switch(activeView) {
      case 'dashboard': return 'Operations Dashboard';
      case 'settings': return 'System Configuration';
      default: return 'Dashboard';
    }
  };

  const isProcessing = currentStep !== PipelineStep.IDLE && currentStep !== PipelineStep.COMPLETED && currentStep !== PipelineStep.ERROR;

  return (
    <div className="min-h-screen flex bg-[#020408] text-zinc-200 overflow-hidden">
      <Sidebar activeView={activeView} onViewChange={setActiveView} />

      <div className="flex-1 flex flex-col h-screen overflow-hidden">
        {/* HEADER */}
        <header className="flex flex-col md:flex-row justify-between items-start md:items-center bg-[#0A0A0A] p-6 border-b border-zinc-800 backdrop-blur-sm z-10">
          <div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-[#FF8C66] via-[#FF5F6D] to-[#7C3AED] bg-clip-text text-transparent">
              {getPageTitle()}
            </h1>
            <p className="text-zinc-400 mt-1 font-mono text-xs">
              Microservice d'IA Agentique &rarr; Unity MCP
            </p>
          </div>
          <div className="flex flex-wrap gap-3 mt-4 md:mt-0">
            <StatusBadge 
              label="GPU VRAM" 
              status={systemStatus.gpuVramUsage > 20 ? 'busy' : 'online'} 
              value={`${systemStatus.gpuVramUsage.toFixed(1)} / ${systemStatus.gpuTotalVram} GB`} 
            />
            <StatusBadge 
              label="Redis Queue" 
              status={systemStatus.redisConnected ? 'online' : 'offline'} 
            />
            <StatusBadge 
              label="Unity MCP" 
              status={systemStatus.unityMcpConnected ? 'online' : 'offline'} 
              value="CONNECTED" 
            />
          </div>
        </header>

        {/* CONTENT SCROLLABLE AREA */}
        <main className="flex-1 overflow-y-auto p-6 scrollbar-hide">
          
          {/* VIEW: DASHBOARD */}
          {activeView === 'dashboard' && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 pb-6">
              {/* LEFT COLUMN: CONTROL & INPUT */}
              <section className="lg:col-span-3 space-y-6">
                <div className="bg-[#0A0A0A] p-6 rounded-2xl border border-zinc-800 h-full flex flex-col">
                  <h2 className="text-lg font-bold text-white mb-4 flex items-center">
                    <IconMessageSquare className="mr-2 text-[#FF8C66]" /> Input Request
                  </h2>
                  
                  <div className="flex-1 flex flex-col justify-end">
                    <p className="text-zinc-500 text-sm mb-4">
                      Describe the asset you want to generate or attach technical specifications (PDF, EML).
                    </p>

                    {/* Chat Input Container */}
                    <div className={`bg-zinc-900/50 border rounded-xl transition-all ${isProcessing ? 'border-zinc-800 opacity-50' : 'border-zinc-700 focus-within:ring-1 focus-within:ring-[#FF8C66] focus-within:border-[#FF8C66]'}`}>
                        
                        {/* Attached Files Display */}
                        {files.length > 0 && (
                          <div className="px-3 pt-3 flex flex-wrap gap-2">
                            {files.map((f, i) => (
                              <div key={i} className="flex items-center gap-2 bg-zinc-800 px-2 py-1 rounded-md text-xs text-zinc-300 border border-zinc-700 animate-fadeIn">
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
                          className="w-full bg-transparent border-none text-sm text-zinc-200 placeholder-zinc-600 focus:ring-0 p-3 resize-none"
                          rows={4}
                          placeholder="Type your prompt here... (e.g. 'A futuristic vending machine')"
                          value={promptText}
                          onChange={(e) => setPromptText(e.target.value)}
                          disabled={isProcessing}
                        />
                        
                        {/* Toolbar */}
                        <div className="flex items-center justify-between px-2 pb-2 border-t border-zinc-800/50 pt-2 mx-2">
                          <div className="flex gap-1">
                              <button 
                                onClick={addMockFile}
                                disabled={isProcessing}
                                className="p-2 text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
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

                    <div className="mt-4 text-[10px] text-zinc-600 font-mono text-center">
                      AI Model: Llama-3-8B-Instruct (Local)
                    </div>
                  </div>
                </div>
              </section>

              {/* MIDDLE COLUMN: PIPELINE VIZ */}
              <section className="lg:col-span-5 space-y-6 flex flex-col">
                <div className="bg-[#0A0A0A] p-6 rounded-2xl border border-zinc-800">
                  <h2 className="text-lg font-bold text-white mb-6 flex items-center justify-between">
                    <span>Pipeline Status</span>
                    {currentStep !== PipelineStep.IDLE && currentStep !== PipelineStep.COMPLETED && (
                      <span className="text-xs font-mono text-[#FF8C66] animate-pulse">PROCESSING...</span>
                    )}
                  </h2>
                  <PipelineVisualizer currentStep={currentStep} generationMethod={metadata?.generationMethod} />
                </div>

                <div className="flex-1 min-h-[300px]">
                  <Terminal logs={logs} />
                </div>
              </section>

              {/* RIGHT COLUMN: PREVIEW */}
              <section className="lg:col-span-4 space-y-6">
                <div className="bg-[#0A0A0A] p-6 rounded-2xl border border-zinc-800 h-full flex flex-col">
                  <h2 className="text-lg font-bold text-white mb-4 flex items-center">
                    <IconDatabase className="mr-2 text-[#7C3AED]" /> Extracted Metadata
                  </h2>
                  
                  {metadata ? (
                    <div className="flex-1 space-y-4">
                      <div className="bg-black/40 p-4 rounded-xl border border-zinc-800 font-mono text-xs text-[#7C3AED] overflow-x-auto">
                          <pre>{JSON.stringify(metadata, null, 2)}</pre>
                      </div>
                      
                      <div className="bg-zinc-900/50 p-4 rounded-xl border border-zinc-800 flex flex-col items-center">
                          <div className="text-xs text-zinc-400 mb-2 w-full text-left uppercase tracking-wider">Asset Preview</div>
                          <div className="w-full aspect-square bg-[#050505] rounded-lg flex items-center justify-center border border-zinc-800 relative overflow-hidden group">
                            <img 
                              src={`https://picsum.photos/400/400?random=${metadata.name}`} 
                              alt="Asset Preview" 
                              className="opacity-50 grayscale group-hover:grayscale-0 transition-all duration-500"
                            />
                            <div className="absolute inset-0 flex items-center justify-center">
                                <span className="px-3 py-1 bg-black/80 text-white text-xs rounded border border-white/20 backdrop-blur-md">
                                  {metadata.generationMethod === GenerationMethod.PROCEDURAL ? 'C# SCRIPT GENERATED' : '.GLB MESH GENERATED'}
                                </span>
                            </div>
                          </div>
                      </div>
                    </div>
                  ) : (
                    <div className="flex-1 flex flex-col items-center justify-center text-zinc-700">
                      <IconDatabase className="w-16 h-16 mb-4 opacity-20" />
                      <p>No metadata extracted yet.</p>
                      <p className="text-sm">Waiting for pipeline Step B...</p>
                    </div>
                  )}
                </div>
              </section>
            </div>
          )}

          {/* VIEW: SETTINGS */}
          {activeView === 'settings' && (
            <div className="max-w-4xl mx-auto space-y-8">
              
              {/* SYSTEM RESOURCES CARD */}
              <div className="bg-[#0A0A0A] rounded-2xl border border-zinc-800 overflow-hidden">
                <div className="p-6 border-b border-zinc-800">
                  <h2 className="text-xl font-bold text-white flex items-center">
                    <IconSettings className="mr-3 text-[#FF8C66]" /> System Resources
                  </h2>
                  <p className="text-zinc-400 mt-1">Manage local computation resources and limits.</p>
                </div>
                <div className="p-8 space-y-8">
                  <div className="space-y-4">
                      <div className="flex justify-between items-end">
                        <label className="text-sm font-semibold text-zinc-300">Total GPU VRAM Limit</label>
                        <span className="text-2xl font-mono text-[#FF8C66] font-bold">{systemStatus.gpuTotalVram} GB</span>
                      </div>
                      <input 
                          type="range" 
                          min="8" max="48" step="4" 
                          value={systemStatus.gpuTotalVram}
                          onChange={(e) => updateSystemStatus('gpuTotalVram', parseFloat(e.target.value))}
                          className="w-full h-3 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-[#FF8C66]"
                      />
                      <div className="flex justify-between text-xs text-zinc-500 font-mono">
                        <span>8 GB (Minimum)</span>
                        <span>48 GB (Dual A10)</span>
                      </div>
                      <p className="text-xs text-zinc-500 bg-[#FF8C66]/10 p-3 rounded border border-[#FF8C66]/20">
                        ⚠️ Increasing VRAM limit allows larger LLM context windows (e.g. Llama 3 70B) but requires hardware support.
                      </p>
                  </div>
                </div>
              </div>

              {/* CONNECTIVITY CARD */}
              <div className="bg-[#0A0A0A] rounded-2xl border border-zinc-800 overflow-hidden">
                <div className="p-6 border-b border-zinc-800">
                   <h2 className="text-xl font-bold text-white flex items-center">
                    <IconActivity className="mr-3 text-[#7C3AED]" /> Service Connectivity
                  </h2>
                   <p className="text-zinc-400 mt-1">Toggle connections to external microservices.</p>
                </div>
                <div className="p-8 grid grid-cols-1 md:grid-cols-2 gap-8">
                   {/* Redis Toggle */}
                   <div className="flex flex-col space-y-4 p-4 rounded-xl bg-zinc-900/30 border border-zinc-800">
                      <div className="flex items-center justify-between">
                          <span className="font-semibold text-zinc-200">Redis Task Queue</span>
                          <button 
                              onClick={() => updateSystemStatus('redisConnected', !systemStatus.redisConnected)}
                              className={`w-14 h-8 rounded-full p-1 transition-colors duration-300 ease-in-out ${systemStatus.redisConnected ? 'bg-[#7C3AED]' : 'bg-zinc-800'}`}
                          >
                              <div className={`w-6 h-6 rounded-full bg-white shadow-sm transition-transform duration-300 ${systemStatus.redisConnected ? 'translate-x-6' : 'translate-x-0'}`} />
                          </button>
                      </div>
                      <p className="text-xs text-zinc-400">
                        Handles asynchronous job processing for 3D generation (TripoSR/Shap-E). Essential for non-blocking API responses.
                      </p>
                      <div className="flex items-center text-xs">
                        Status: 
                        <span className={`ml-2 px-2 py-0.5 rounded ${systemStatus.redisConnected ? 'bg-[#7C3AED]/20 text-[#7C3AED]' : 'bg-red-500/20 text-red-400'}`}>
                           {systemStatus.redisConnected ? 'ONLINE' : 'DISCONNECTED'}
                        </span>
                      </div>
                   </div>

                   {/* MCP Toggle */}
                   <div className="flex flex-col space-y-4 p-4 rounded-xl bg-zinc-900/30 border border-zinc-800">
                      <div className="flex items-center justify-between">
                          <span className="font-semibold text-zinc-200">Unity MCP Server</span>
                          <button 
                              onClick={() => updateSystemStatus('unityMcpConnected', !systemStatus.unityMcpConnected)}
                              className={`w-14 h-8 rounded-full p-1 transition-colors duration-300 ease-in-out ${systemStatus.unityMcpConnected ? 'bg-[#7C3AED]' : 'bg-zinc-800'}`}
                          >
                              <div className={`w-6 h-6 rounded-full bg-white shadow-sm transition-transform duration-300 ${systemStatus.unityMcpConnected ? 'translate-x-6' : 'translate-x-0'}`} />
                          </button>
                      </div>
                      <p className="text-xs text-zinc-400">
                        Model Context Protocol connection to Unity Editor. Enables direct scene manipulation tools (Spawn, Transform).
                      </p>
                      <div className="flex items-center text-xs">
                        Status: 
                        <span className={`ml-2 px-2 py-0.5 rounded ${systemStatus.unityMcpConnected ? 'bg-[#7C3AED]/20 text-[#7C3AED]' : 'bg-red-500/20 text-red-400'}`}>
                           {systemStatus.unityMcpConnected ? 'LISTENING' : 'OFFLINE'}
                        </span>
                      </div>
                   </div>
                </div>
              </div>

            </div>
          )}

        </main>
      </div>
    </div>
  );
}