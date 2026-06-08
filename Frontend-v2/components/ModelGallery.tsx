import React, { useCallback, useEffect, useRef, useState } from 'react';
import { GeneratedModel } from '../types';
import { ModelViewer3D } from './ModelViewer3D';
import { IconBox, IconTrash, IconX } from './Icons';
import { API_BASE } from '../api';

type UnityScene = 'new' | 'existing';

interface LauncherStatus {
  supported: boolean;
  installed: boolean;
  registered: boolean;
  repo_match?: boolean;
  reason?: string;
  app_path?: string;
  repo_root?: string;
}

const fireUnityUri = (model: GeneratedModel, scene: UnityScene) => {
  const raw = model.downloadUrl;
  const abs = /^https?:\/\//i.test(raw)
    ? raw
    : `${API_BASE}${raw.startsWith('/') ? '' : '/'}${raw}`;
  const q = new URLSearchParams({
    url: abs,
    scene,
    id: model.id,
    name: (model.prompt ?? model.id).slice(0, 60),
  });
  window.location.href = `unity3dgen://spawn?${q.toString()}`;
};

const fetchStatus = async (): Promise<LauncherStatus> => {
  const r = await fetch(`${API_BASE}/api/v1/unity/launcher-status`);
  if (!r.ok) throw new Error(`launcher-status ${r.status}`);
  return r.json();
};

const installLauncher = async (): Promise<LauncherStatus> => {
  const r = await fetch(`${API_BASE}/api/v1/unity/register-launcher`, { method: 'POST' });
  if (!r.ok) {
    const body = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
    throw new Error(body.detail ?? `HTTP ${r.status}`);
  }
  return r.json();
};

interface ModelGalleryProps {
  models: GeneratedModel[];
  onRemove?: (id: string) => void;
}

const sourceLabels: Record<GeneratedModel['source'], string> = {
  'image-to-3d': 'Image',
  'text-to-3d': 'Text',
  'multiview-to-3d': 'Multi-View',
};

const sourceColors: Record<GeneratedModel['source'], string> = {
  'image-to-3d': 'bg-[#FF8C66]/20 text-[#FF8C66]',
  'text-to-3d': 'bg-[#7C3AED]/20 text-[#7C3AED]',
  'multiview-to-3d': 'bg-emerald-500/20 text-emerald-400',
};

export const ModelGallery: React.FC<ModelGalleryProps> = ({ models, onRemove }) => {
  const [selectedModel, setSelectedModel] = useState<GeneratedModel | null>(null);
  const [unityMenuFor, setUnityMenuFor] = useState<string | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);

  const [status, setStatus] = useState<LauncherStatus | null>(null);
  const [installing, setInstalling] = useState(false);
  const [installError, setInstallError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchStatus()
      .then((s) => { if (!cancelled) setStatus(s); })
      .catch(() => { if (!cancelled) setStatus({ supported: false, installed: false, registered: false, reason: 'Backend unreachable' }); });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!unityMenuFor) return;
    const onDocClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setUnityMenuFor(null);
      }
    };
    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, [unityMenuFor]);

  const install = useCallback(async () => {
    setInstalling(true);
    setInstallError(null);
    try {
      const s = await installLauncher();
      setStatus(s);
      return s;
    } catch (e: any) {
      setInstallError(e?.message ?? String(e));
      return null;
    } finally {
      setInstalling(false);
    }
  }, []);

  const openInUnity = useCallback(async (model: GeneratedModel, scene: UnityScene) => {
    let s = status;
    if (!s) {
      try { s = await fetchStatus(); setStatus(s); } catch { /* fall through */ }
    }
    if (s && !s.supported) {
      setInstallError(s.reason ?? 'Unity launcher is not supported on this host.');
      return;
    }
    if (!s || !s.registered || s.repo_match === false) {
      const installed = await install();
      if (!installed || !installed.registered) return;
    }
    fireUnityUri(model, scene);
  }, [status, install]);

  const needsInstall =
    status !== null && status.supported && (!status.registered || status.repo_match === false);

  if (models.length === 0) {
    return (
      <div className="max-w-5xl mx-auto">
        <div className="card p-6">
          <h2 className="text-xl font-bold text-heading mb-6 flex items-center">
            <IconBox className="mr-3 text-[#FF8C66]" /> Model Gallery
          </h2>
          <div className="flex flex-col items-center justify-center py-16 text-theme-muted">
            <IconBox className="w-20 h-20 mb-4 opacity-20" />
            <p className="text-lg font-medium">No models generated yet</p>
            <p className="text-sm mt-1">Use Image-to-3D, Text-to-3D, or Multi-View to generate models</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="card p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-heading flex items-center">
            <IconBox className="mr-3 text-[#FF8C66]" /> Model Gallery
          </h2>
          <span className="text-sm text-theme-muted">{models.length} model{models.length !== 1 ? 's' : ''}</span>
        </div>

        {needsInstall && (
          <div className="mb-4 flex items-center justify-between gap-3 px-3 py-2 rounded-lg border border-amber-500/40 bg-amber-500/10 text-amber-300 text-sm">
            <span>
              Unity launcher {status?.installed ? 'is out of date (repo path changed)' : 'is not installed yet'}. Install it to enable <b>Open in Unity</b>.
            </span>
            <button
              type="button"
              disabled={installing}
              onClick={install}
              className="text-xs px-3 py-1.5 rounded bg-amber-500 hover:bg-amber-400 text-black font-semibold disabled:opacity-50"
            >
              {installing ? 'Installing…' : 'Install now'}
            </button>
          </div>
        )}

        {status && !status.supported && (
          <div className="mb-4 px-3 py-2 rounded-lg border border-slate-500/40 bg-slate-500/10 text-slate-300 text-xs">
            Unity integration unavailable: {status.reason}
          </div>
        )}

        {installError && (
          <div className="mb-4 flex items-center justify-between gap-3 px-3 py-2 rounded-lg border border-red-500/40 bg-red-500/10 text-red-300 text-xs">
            <span>Install failed: {installError}</span>
            <button
              type="button"
              onClick={() => setInstallError(null)}
              className="text-red-300 hover:text-red-100"
            >
              dismiss
            </button>
          </div>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {models.map(model => (
            <div
              key={model.id}
              className="bg-[var(--bg-tertiary)] rounded-xl border border-theme hover:border-[var(--border-secondary)] transition-all cursor-pointer group overflow-hidden"
              onClick={() => setSelectedModel(model)}
            >
              {/* Thumbnail via model-viewer */}
              <div className="aspect-square bg-[var(--bg-input)] relative overflow-hidden">
                {/* @ts-ignore */}
                <model-viewer
                  src={model.previewUrl}
                  auto-rotate
                  camera-controls
                  style={{ width: '100%', height: '100%', backgroundColor: 'transparent' }}
                  interaction-prompt="none"
                />
              </div>

              <div className="p-3 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${sourceColors[model.source]}`}>
                      {sourceLabels[model.source]}
                    </span>
                    {model.fromCache && (
                      <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-amber-500/20 text-amber-400">
                        Cached
                      </span>
                    )}
                    {model.attempt && model.attempt > 1 && (
                      <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-emerald-500/20 text-emerald-400">
                        v{model.attempt}
                      </span>
                    )}
                    {model.generationTime != null && (
                      <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-blue-500/20 text-blue-400">
                        {model.generationTime}s
                      </span>
                    )}
                  </div>
                  <span className="text-xs text-theme-muted uppercase">{model.format}</span>
                </div>

                {model.prompt && (
                  <p className="text-xs text-theme-secondary truncate">{model.prompt}</p>
                )}

                <div className="flex items-center justify-between">
                  <span className="text-xs text-theme-muted">
                    {new Date(model.createdAt).toLocaleTimeString()}
                    {model.generationTime != null && ` · ${model.generationTime}s`}
                  </span>
                  <div className="flex gap-1 items-center">
                    {status?.supported !== false && (
                      <div className="relative">
                        <button
                          type="button"
                          disabled={installing}
                          onClick={(e) => {
                            e.stopPropagation();
                            setUnityMenuFor(unityMenuFor === model.id ? null : model.id);
                          }}
                          className="text-xs px-2 py-1 bg-[#111827] hover:bg-black text-white rounded transition-colors border border-[#374151] disabled:opacity-50"
                          title="Open in Unity Editor"
                        >
                          {installing ? '…' : 'Unity ▾'}
                        </button>
                        {unityMenuFor === model.id && (
                          <div
                            ref={menuRef}
                            onClick={(e) => e.stopPropagation()}
                            className="absolute right-0 bottom-full mb-1 z-20 bg-[var(--bg-card)] border border-theme rounded-md shadow-lg text-xs overflow-hidden min-w-[140px]"
                          >
                            <button
                              type="button"
                              className="block w-full text-left px-3 py-2 hover:bg-[var(--bg-tertiary)] text-theme-primary"
                              onClick={(e) => {
                                e.stopPropagation();
                                setUnityMenuFor(null);
                                openInUnity(model, 'new');
                              }}
                            >
                              New scene
                            </button>
                            <button
                              type="button"
                              className="block w-full text-left px-3 py-2 hover:bg-[var(--bg-tertiary)] text-theme-primary border-t border-theme"
                              onClick={(e) => {
                                e.stopPropagation();
                                setUnityMenuFor(null);
                                openInUnity(model, 'existing');
                              }}
                            >
                              Current scene
                            </button>
                          </div>
                        )}
                      </div>
                    )}
                    <a
                      href={model.downloadUrl}
                      download
                      onClick={(e) => e.stopPropagation()}
                      className="text-xs px-2 py-1 bg-[#7C3AED] hover:bg-[#6d28d9] text-white rounded transition-colors"
                    >
                      Download
                    </a>
                    {onRemove && (
                      <button
                        onClick={(e) => { e.stopPropagation(); onRemove(model.id); }}
                        className="p-1 text-theme-muted hover:text-red-400 transition-colors"
                      >
                        <IconTrash className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Full-screen viewer modal */}
      {selectedModel && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm" onClick={() => setSelectedModel(null)}>
          <div className="bg-[var(--bg-card)] rounded-2xl border border-theme shadow-2xl w-full max-w-3xl m-4 overflow-hidden" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between p-4 border-b border-theme">
              <div className="flex items-center gap-3">
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${sourceColors[selectedModel.source]}`}>
                  {sourceLabels[selectedModel.source]}
                </span>
                {selectedModel.prompt && <span className="text-sm text-theme-secondary">{selectedModel.prompt}</span>}
              </div>
              <button onClick={() => setSelectedModel(null)} className="p-2 text-theme-muted hover:text-theme-primary transition-colors">
                <IconX className="w-5 h-5" />
              </button>
            </div>
            <div className="p-4">
              <ModelViewer3D src={selectedModel.previewUrl} className="!aspect-[4/3]" />
            </div>
            <div className="flex flex-col gap-3 p-4 border-t border-theme">
              <a
                href={selectedModel.downloadUrl}
                download
                className="text-center px-4 py-2 bg-[#7C3AED] hover:bg-[#6d28d9] text-white rounded-xl font-bold transition-all"
              >
                Download {selectedModel.format.toUpperCase()}
              </a>
              {status?.supported !== false && (
                <div className="flex gap-3">
                  <button
                    type="button"
                    disabled={installing}
                    onClick={() => openInUnity(selectedModel, 'new')}
                    className="flex-1 px-4 py-2 bg-[#111827] hover:bg-black text-white rounded-xl font-bold transition-all border border-[#374151] disabled:opacity-50"
                    title="Launch Unity Editor with a fresh empty scene and spawn this model"
                  >
                    {installing ? 'Setting up…' : 'Open in Unity · New scene'}
                  </button>
                  <button
                    type="button"
                    disabled={installing}
                    onClick={() => openInUnity(selectedModel, 'existing')}
                    className="flex-1 px-4 py-2 bg-[#111827] hover:bg-black text-white rounded-xl font-bold transition-all border border-[#374151] disabled:opacity-50"
                    title="Spawn this model into the currently open scene (or launch Unity if closed)"
                  >
                    {installing ? 'Setting up…' : 'Open in Unity · Current scene'}
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
