import React from 'react';

const TechBadge = ({ children }: { children: React.ReactNode }) => (
  <span className="px-2 py-1 bg-zinc-800 border border-zinc-700 text-zinc-300 rounded text-xs font-mono mr-2 mb-2 inline-block">
    {children}
  </span>
);

export const ProjectSpecs = () => {
  return (
    <div className="max-w-6xl mx-auto space-y-8 pb-12">
      
      {/* HEADER SECTION */}
      <div className="bg-[#0A0A0A] rounded-2xl border border-zinc-800 p-8">
        <h1 className="text-3xl font-bold text-white mb-4 bg-gradient-to-r from-[#FF8C66] via-[#FF5F6D] to-[#7C3AED] bg-clip-text text-transparent">
          Microservice d'IA Agentique pour la Génération d'Assets 3D via MCP
        </h1>
        <p className="text-zinc-400 leading-relaxed text-lg">
          Transformation de la chaîne de production 3D par l'automatisation cognitive et le protocole MCP (Model Context Protocol).
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* LEFT COL: CONTEXT & STACK */}
        <div className="lg:col-span-1 space-y-8">
            {/* OBJECTIVES */}
            <div className="bg-[#0A0A0A] rounded-2xl border border-zinc-800 p-6">
                <h2 className="text-lg font-bold text-white mb-4 border-b border-zinc-800 pb-2">Objectifs du Microservice</h2>
                <ul className="space-y-3 text-sm text-zinc-300">
                    <li className="flex items-start">
                        <span className="text-[#FF8C66] mr-2">•</span>
                        Ingestion multi-formats (PDF, Email, CSV, XML).
                    </li>
                    <li className="flex items-start">
                        <span className="text-[#FF8C66] mr-2">•</span>
                        Nettoyage et Extraction de métadonnées Unity via LLM.
                    </li>
                    <li className="flex items-start">
                        <span className="text-[#FF8C66] mr-2">•</span>
                        Génération 3D Neuronale (TripoSR) ou Procédurale (Code).
                    </li>
                    <li className="flex items-start">
                        <span className="text-[#7C3AED] mr-2 font-bold">•</span>
                        Pilotage Unity via MCP (Model Context Protocol).
                    </li>
                </ul>
            </div>

            {/* TECH STACK */}
            <div className="bg-[#0A0A0A] rounded-2xl border border-zinc-800 p-6">
                <h2 className="text-lg font-bold text-white mb-4 border-b border-zinc-800 pb-2">Stack Technique</h2>
                <div className="space-y-4">
                    <div>
                        <div className="text-xs uppercase text-zinc-500 font-bold mb-2">Orchestration</div>
                        <div>
                            <TechBadge>FastAPI</TechBadge>
                            <TechBadge>Celery</TechBadge>
                            <TechBadge>Redis</TechBadge>
                        </div>
                    </div>
                    <div>
                        <div className="text-xs uppercase text-zinc-500 font-bold mb-2">Intelligence Artificielle</div>
                        <div>
                            <TechBadge>Llama 3 8B (vLLM)</TechBadge>
                            <TechBadge>TripoSR</TechBadge>
                            <TechBadge>Qwen 2.5 Coder</TechBadge>
                            <TechBadge>Unstructured</TechBadge>
                        </div>
                    </div>
                    <div>
                        <div className="text-xs uppercase text-zinc-500 font-bold mb-2">Unity / MCP</div>
                        <div>
                            <TechBadge>MCP Python SDK</TechBadge>
                            <TechBadge>Unity C# Bridge</TechBadge>
                            <TechBadge>glTFast</TechBadge>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        {/* RIGHT COL: ARCHITECTURE & BACKLOG */}
        <div className="lg:col-span-2 space-y-8">
            
            {/* ARCHITECTURE FLOW */}
            <div className="bg-[#0A0A0A] rounded-2xl border border-zinc-800 p-6">
                 <h2 className="text-lg font-bold text-white mb-4 border-b border-zinc-800 pb-2">Flux de Données "Agentique"</h2>
                 <div className="space-y-4 text-sm">
                     <div className="flex items-center p-3 bg-zinc-900/40 rounded-lg border border-zinc-800">
                        <div className="w-8 h-8 rounded-full bg-zinc-800 flex items-center justify-center text-white font-bold mr-4">1</div>
                        <div>
                            <strong className="text-[#FF8C66]">Réception & Analyse</strong>
                            <p className="text-zinc-400">Utilisateur envoie Email/PDF -> LLM extrait JSON (ex: type: chair, color: red).</p>
                        </div>
                     </div>
                     <div className="flex items-center p-3 bg-zinc-900/40 rounded-lg border border-zinc-800">
                        <div className="w-8 h-8 rounded-full bg-zinc-800 flex items-center justify-center text-white font-bold mr-4">2</div>
                        <div>
                            <strong className="text-[#FF8C66]">Génération</strong>
                            <p className="text-zinc-400">Microservice crée le fichier .glb ou le script .cs.</p>
                        </div>
                     </div>
                     <div className="flex items-center p-3 bg-[#7C3AED]/10 rounded-lg border border-[#7C3AED]/30">
                        <div className="w-8 h-8 rounded-full bg-[#7C3AED] flex items-center justify-center text-white font-bold mr-4">3</div>
                        <div>
                            <strong className="text-[#7C3AED]">Action MCP (Client -> Serveur)</strong>
                            <p className="text-zinc-300">
                                <code>call_tool("ImportAsset", path)</code> puis <code>call_tool("SetTransform", pos)</code>.
                            </p>
                        </div>
                     </div>
                     <div className="flex items-center p-3 bg-zinc-900/40 rounded-lg border border-zinc-800">
                        <div className="w-8 h-8 rounded-full bg-zinc-800 flex items-center justify-center text-white font-bold mr-4">4</div>
                        <div>
                            <strong className="text-[#FF8C66]">Exécution Unity</strong>
                            <p className="text-zinc-400">Unity reçoit les commandes et met à jour la scène en temps réel.</p>
                        </div>
                     </div>
                 </div>
            </div>

            {/* BACKLOG AGILE */}
            <div className="bg-[#0A0A0A] rounded-2xl border border-zinc-800 p-6">
                <h2 className="text-lg font-bold text-white mb-6 flex items-center justify-between">
                    <span>Backlog Agile (18 Semaines)</span>
                    <span className="text-xs bg-[#7C3AED]/20 text-[#7C3AED] px-2 py-1 rounded">PFE Roadmap</span>
                </h2>
                
                <div className="space-y-6 relative border-l-2 border-zinc-800 ml-3 pl-8">
                    
                    {/* Phase 1 */}
                    <div className="relative">
                        <div className="absolute -left-[41px] bg-zinc-900 border-4 border-[#0A0A0A] w-6 h-6 rounded-full"></div>
                        <h3 className="text-md font-bold text-zinc-200">Phase 1 : Infrastructure & Ingestion</h3>
                        <span className="text-xs font-mono text-zinc-500 mb-2 block">Sprints 1-2</span>
                        <ul className="list-disc list-inside text-sm text-zinc-400 space-y-1">
                            <li>Setup Docker-Compose (FastAPI, Redis, Celery).</li>
                            <li>Pipeline <code>unstructured</code> pour parser PDF et Emails.</li>
                            <li>Filtres Regex pour nettoyer le "bruit".</li>
                        </ul>
                    </div>

                    {/* Phase 2 */}
                    <div className="relative">
                        <div className="absolute -left-[41px] bg-zinc-900 border-4 border-[#0A0A0A] w-6 h-6 rounded-full"></div>
                        <h3 className="text-md font-bold text-zinc-200">Phase 2 : Le Cerveau - LLM & Extraction</h3>
                        <span className="text-xs font-mono text-zinc-500 mb-2 block">Sprints 3-4</span>
                        <ul className="list-disc list-inside text-sm text-zinc-400 space-y-1">
                            <li>Intégration Llama-3-8B-Instruct (via llama-cpp).</li>
                            <li>Prompt Système pour Schema Enforcement (JSON).</li>
                            <li>Validation Pydantic des métadonnées.</li>
                        </ul>
                    </div>

                    {/* Phase 3 */}
                    <div className="relative">
                        <div className="absolute -left-[41px] bg-zinc-900 border-4 border-[#0A0A0A] w-6 h-6 rounded-full"></div>
                        <h3 className="text-md font-bold text-zinc-200">Phase 3 : Le Constructeur - Génération 3D</h3>
                        <span className="text-xs font-mono text-zinc-500 mb-2 block">Sprints 5-6</span>
                        <ul className="list-disc list-inside text-sm text-zinc-400 space-y-1">
                            <li>Intégration TripoSR (Text-to-Mesh).</li>
                            <li>Génération scripts C# (Qwen 2.5 Coder).</li>
                            <li>Conversion OBJ vers GLB.</li>
                        </ul>
                    </div>

                    {/* Phase 4 */}
                    <div className="relative">
                        <div className="absolute -left-[41px] bg-[#7C3AED] border-4 border-[#0A0A0A] w-6 h-6 rounded-full shadow-[0_0_10px_rgba(124,58,237,0.5)]"></div>
                        <h3 className="text-md font-bold text-[#7C3AED]">Phase 4 : Serveur MCP Unity</h3>
                        <span className="text-xs font-mono text-[#7C3AED] mb-2 block">Sprints 7-8 (Focus Actuel)</span>
                        <ul className="list-disc list-inside text-sm text-zinc-300 space-y-1">
                            <li>Implémenter Client MCP (Python SDK).</li>
                            <li>Développer Serveur MCP C# dans Unity.</li>
                            <li>Outils: <code>SpawnGLB</code>, <code>CreatePrimitive</code>.</li>
                            <li>Orchestration Celery -> MCP Call.</li>
                        </ul>
                    </div>

                    {/* Phase 5 */}
                    <div className="relative">
                        <div className="absolute -left-[41px] bg-zinc-900 border-4 border-[#0A0A0A] w-6 h-6 rounded-full"></div>
                        <h3 className="text-md font-bold text-zinc-200">Phase 5 : UX & Finalisation</h3>
                        <span className="text-xs font-mono text-zinc-500 mb-2 block">Sprints 9+</span>
                        <ul className="list-disc list-inside text-sm text-zinc-400 space-y-1">
                            <li>Fenêtre "Moniteur IA" dans Unity.</li>
                            <li>Test E2E : Email -> Unity sans clic.</li>
                            <li>Optimisation VRAM.</li>
                        </ul>
                    </div>

                </div>
            </div>

        </div>

      </div>
    </div>
  );
};