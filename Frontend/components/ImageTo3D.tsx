import React, { useState, useRef, useCallback, useEffect } from 'react';
import { GeneratedModel } from '../types';
import { ModelViewer3D } from './ModelViewer3D';
import { IconUpload, IconLoader, IconBox } from './Icons';
import { API_BASE } from '../api';

interface ImageTo3DProps {
  onModelGenerated?: (model: GeneratedModel) => void;
}

const PRESETS = {
  fast:     { steps: 5,  octreeResolution: 64,  faceCount: 10000 },
  balanced: { steps: 10, octreeResolution: 128, faceCount: 20000 },
  quality:  { steps: 20, octreeResolution: 192, faceCount: 40000 },
} as const;

export const ImageTo3D: React.FC<ImageTo3DProps> = ({ onModelGenerated }) => {
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageB64, setImageB64] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ previewUrl: string; downloadUrl: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Basic params
  const [texture, setTexture] = useState(false);
  const [steps, setSteps] = useState(10);
  const [outputType, setOutputType] = useState('glb');
  // Advanced params
  const [seed, setSeed] = useState(1234);
  const [guidanceScale, setGuidanceScale] = useState(5.0);
  const [octreeResolution, setOctreeResolution] = useState(128);
  const [numChunks, setNumChunks] = useState(50000);
  const [faceCount, setFaceCount] = useState(20000);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [activePreset, setActivePreset] = useState<'fast' | 'balanced' | 'quality' | null>('balanced');
  // UI
  const [elapsed, setElapsed] = useState(0);
  const [generationTime, setGenerationTime] = useState<number | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const currentUidRef = useRef<string | null>(null);
  const cancelledRef = useRef(false);

  useEffect(() => {
    if (!loading) return;
    const t0 = Date.now();
    setElapsed(0);
    const tick = () => setElapsed(Math.round((Date.now() - t0) / 1000));
    const id = setInterval(tick, 1000);
    const onVisible = () => { if (!document.hidden) tick(); };
    document.addEventListener('visibilitychange', onVisible);
    return () => { clearInterval(id); document.removeEventListener('visibilitychange', onVisible); };
  }, [loading]);

  const applyPreset = (preset: 'fast' | 'balanced' | 'quality') => {
    const p = PRESETS[preset];
    setSteps(p.steps);
    setOctreeResolution(p.octreeResolution);
    setFaceCount(p.faceCount);
    setActivePreset(preset);
  };

  const handleFile = useCallback((file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      setImagePreview(dataUrl);
      setImageB64(dataUrl.split(',')[1]);
      setResult(null);
      setError(null);
    };
    reader.readAsDataURL(file);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const cancel = async () => {
    cancelledRef.current = true;
    const uid = currentUidRef.current;
    if (uid) {
      try { await fetch(`${API_BASE}/api/v1/generation/${uid}`, { method: 'DELETE' }); } catch {}
    }
    setLoading(false);
  };

  const generate = async () => {
    if (!imageB64) return;
    cancelledRef.current = false;
    setLoading(true);
    setError(null);
    setResult(null);
    setGenerationTime(null);
    try {
      const submitRes = await fetch(`${API_BASE}/api/v1/image-to-3d/async`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: imageB64,
          seed,
          num_inference_steps: steps,
          guidance_scale: guidanceScale,
          octree_resolution: octreeResolution,
          num_chunks: numChunks,
          texture,
          face_count: faceCount,
          type: outputType,
        }),
      });
      if (!submitRes.ok) {
        const err = await submitRes.json().catch(() => ({}));
        let msg: string;
        if (Array.isArray(err?.detail)) {
          msg = err.detail.map((d: any) => d?.msg ?? JSON.stringify(d)).join('; ');
        } else if (typeof err?.detail === 'string') {
          msg = err.detail;
        } else if (err?.message) {
          msg = err.message;
        } else {
          msg = `HTTP ${submitRes.status}`;
        }
        throw new Error(msg);
      }
      const { uid } = await submitRes.json();
      currentUidRef.current = uid;

      // WebSocket with polling fallback
      let data: any = null;
      const wsBase = API_BASE.replace(/^http/, 'ws');
      try {
        await new Promise<void>((resolve, reject) => {
          const ws = new WebSocket(`${wsBase}/api/v1/ws/generation/${uid}`);
          ws.onmessage = (ev) => {
            const prog = JSON.parse(ev.data);
            if (prog.stage === 'completed') { data = { status: 'completed', ...prog }; ws.close(); resolve(); }
            else if (prog.stage === 'failed') { ws.close(); reject(new Error(prog.error || 'Generation failed')); }
            else if (prog.stage === 'cancelled') { ws.close(); resolve(); }
          };
          ws.onerror = () => reject(new Error('ws error'));
        });
      } catch {
        // fallback: polling
        while (true) {
          await new Promise(r => setTimeout(r, 2000));
          if (cancelledRef.current) return;
          const pollRes = await fetch(`${API_BASE}/api/v1/generation-status/${uid}`);
          if (!pollRes.ok) throw new Error(`Poll failed: HTTP ${pollRes.status}`);
          const poll = await pollRes.json();
          if (poll.status === 'completed') { data = poll; break; }
          if (poll.status === 'failed') throw new Error(poll.error || 'Generation failed');
          if (poll.status === 'cancelled') return;
        }
      }

      if (cancelledRef.current) return;
      if (!data) return;
      setResult({ previewUrl: `${API_BASE}${data.preview_url}`, downloadUrl: `${API_BASE}${data.download_url}` });
      setGenerationTime(data.generation_time ?? null);
      onModelGenerated?.({
        id: data.uid ?? crypto.randomUUID(),
        previewUrl: `${API_BASE}${data.preview_url}`,
        downloadUrl: `${API_BASE}${data.download_url}`,
        format: data.format,
        source: 'image-to-3d',
        createdAt: new Date().toISOString(),
        fromCache: data.from_cache ?? false,
        generationTime: data.generation_time,
        faceCount: data.face_count,
        fileSizeMb: data.file_size_mb,
      });
    } catch (e: any) {
      if (!cancelledRef.current) setError(e.message);
    } finally {
      setLoading(false);
      currentUidRef.current = null;
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div className="card p-6">
        <h2 className="text-xl font-bold text-heading mb-2 flex items-center">
          <IconBox className="mr-3 text-[#FF8C66]" /> Image to 3D
        </h2>
        <p className="text-body text-sm mb-6">Upload an image and generate a 3D model from it.</p>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left */}
          <div className="space-y-4">
            <div
              className="dropzone flex flex-col items-center justify-center p-8 cursor-pointer min-h-[240px]"
              onClick={() => fileRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
            >
              {imagePreview ? (
                <img src={imagePreview} alt="Preview" className="max-h-56 rounded-lg object-contain" />
              ) : (
                <>
                  <IconUpload className="w-12 h-12 text-theme-muted mb-4" />
                  <p className="text-theme-muted text-sm">Drop an image here or click to browse</p>
                  <p className="text-theme-muted text-xs mt-1">PNG, JPG — transparent background recommended</p>
                </>
              )}
            </div>
            <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />

            {/* Presets */}
            <div className="flex gap-2">
              {(['fast', 'balanced', 'quality'] as const).map(p => (
                <button
                  key={p}
                  type="button"
                  onClick={() => applyPreset(p)}
                  disabled={loading}
                  className={`flex-1 px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
                    activePreset === p
                      ? 'bg-[#FF8C66] border-[#FF8C66] text-black'
                      : 'bg-transparent border-theme text-theme-secondary hover:border-[#FF8C66] hover:text-[#FF8C66]'
                  }`}
                >
                  {p === 'fast' ? '⚡ Fast' : p === 'balanced' ? '⚖ Balanced' : '✦ Quality'}
                </button>
              ))}
            </div>

            {/* Basic options */}
            <div className="flex flex-wrap items-center gap-3">
              <label className="flex items-center gap-2 text-sm text-theme-secondary">
                <input type="checkbox" checked={texture} onChange={(e) => setTexture(e.target.checked)} className="accent-[#7C3AED]" />
                Texture
              </label>
              <label className="flex items-center gap-2 text-sm text-theme-secondary">
                Format:
                <select value={outputType} onChange={(e) => setOutputType(e.target.value)} className="bg-[var(--bg-input)] border border-theme rounded px-2 py-1 text-sm text-theme-primary">
                  <option value="glb">GLB</option><option value="obj">OBJ</option>
                  <option value="ply">PLY</option><option value="stl">STL</option>
                </select>
              </label>
              <label className="flex items-center gap-2 text-sm text-theme-secondary">
                Steps:
                <input type="number" value={steps} onChange={(e) => { setSteps(Number(e.target.value)); setActivePreset(null); }} min={1} max={100}
                  className="w-16 bg-[var(--bg-input)] border border-theme rounded px-2 py-1 text-sm text-theme-primary" />
              </label>
            </div>

            {/* Advanced toggle */}
            <button type="button" onClick={() => setShowAdvanced(v => !v)}
              className="text-xs text-theme-muted hover:text-theme-secondary flex items-center gap-1 transition-colors">
              ⚙ Advanced {showAdvanced ? '▲' : '▼'}
            </button>

            {showAdvanced && (
              <div className="grid grid-cols-2 gap-3 p-3 bg-[var(--bg-tertiary)] rounded-xl border border-theme">
                {/* Seed */}
                <label className="flex flex-col gap-1 text-xs text-theme-secondary">
                  Seed
                  <div className="flex gap-1">
                    <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))}
                      className="flex-1 min-w-0 bg-[var(--bg-input)] border border-theme rounded px-2 py-1 text-sm text-theme-primary" />
                    <button type="button" onClick={() => setSeed(Math.floor(Math.random() * 99999))}
                      title="Random seed"
                      className="px-2 py-1 bg-[var(--bg-input)] border border-theme rounded hover:bg-[var(--bg-card)] transition-colors text-sm">⟳</button>
                  </div>
                </label>
                {/* Guidance Scale */}
                <label className="flex flex-col gap-1 text-xs text-theme-secondary">
                  <span>Guidance Scale <span className="font-mono text-theme-primary">{guidanceScale.toFixed(1)}</span></span>
                  <input type="range" value={guidanceScale} onChange={(e) => setGuidanceScale(Number(e.target.value))}
                    min={0} max={20} step={0.5} className="accent-[#FF8C66] mt-2" />
                </label>
                {/* Octree Resolution */}
                <label className="flex flex-col gap-1 text-xs text-theme-secondary">
                  Mesh Resolution
                  <select value={octreeResolution} onChange={(e) => { setOctreeResolution(Number(e.target.value)); setActivePreset(null); }}
                    className="bg-[var(--bg-input)] border border-theme rounded px-2 py-1 text-sm text-theme-primary">
                    <option value={64}>64 — fastest</option>
                    <option value={128}>128 — default</option>
                    <option value={160}>160 — detailed</option>
                    <option value={192}>192 — high quality</option>
                  </select>
                </label>
                {/* Num Chunks */}
                <label className="flex flex-col gap-1 text-xs text-theme-secondary">
                  Mesh Chunks
                  <select value={numChunks} onChange={(e) => setNumChunks(Number(e.target.value))}
                    className="bg-[var(--bg-input)] border border-theme rounded px-2 py-1 text-sm text-theme-primary">
                    <option value={2000}>2 000 — low RAM</option>
                    <option value={8000}>8 000</option>
                    <option value={32000}>32 000</option>
                    <option value={50000}>50 000 — default</option>
                    <option value={100000}>100 000 — fast</option>
                    <option value={200000}>200 000 — fastest</option>
                  </select>
                </label>
                {/* Face Count (only relevant with texture) */}
                {texture && (
                  <label className="flex flex-col gap-1 text-xs text-theme-secondary col-span-2">
                    Max Face Count
                    <input type="number" value={faceCount} onChange={(e) => { setFaceCount(Number(e.target.value)); setActivePreset(null); }}
                      min={1000} step={1000}
                      className="bg-[var(--bg-input)] border border-theme rounded px-2 py-1 text-sm text-theme-primary" />
                  </label>
                )}
              </div>
            )}

            <button
              type="button"
              onClick={() => { setSteps(1); setOctreeResolution(64); setNumChunks(200000); setShowAdvanced(true); setActivePreset(null); }}
              className="w-full px-3 py-1.5 border border-[#FF8C66]/50 hover:border-[#FF8C66] text-[#FF8C66] text-xs font-bold rounded-xl transition-all"
              disabled={loading}
            >
              ⚡ Draft Mode — fastest (steps=1, res=64, chunks=200k)
            </button>

            {loading ? (
              <div className="flex gap-2">
                <button type="button" onClick={cancel}
                  className="flex-1 px-4 py-3 border border-red-500/60 hover:border-red-500 text-red-400 hover:text-red-300 rounded-xl font-bold transition-all">
                  ✕ Cancel
                </button>
                <div className="flex-1 px-4 py-3 bg-[#FF8C66]/40 text-black/60 rounded-xl font-bold flex items-center justify-center gap-2 cursor-not-allowed">
                  <IconLoader className="w-5 h-5 animate-spin" /> {elapsed}s
                </div>
              </div>
            ) : (
              <button onClick={generate} disabled={!imageB64 || loading}
                className="w-full px-4 py-3 bg-[#FF8C66] hover:bg-[#ff7a4d] disabled:opacity-50 disabled:cursor-not-allowed text-black rounded-xl font-bold transition-all flex items-center justify-center gap-2">
                Generate 3D Model
              </button>
            )}

            {error && <p className="text-red-400 text-sm bg-red-500/10 border border-red-500/20 p-3 rounded-lg">{error}</p>}
          </div>

          {/* Right: Result */}
          <div className="space-y-4">
            {result ? (
              <>
                <ModelViewer3D src={result.previewUrl} />
                {generationTime != null && (
                  <div className="text-center text-sm text-theme-muted font-mono">Generated in {generationTime}s</div>
                )}
                <a href={result.downloadUrl} download
                  className="block w-full text-center px-4 py-3 bg-[#7C3AED] hover:bg-[#6d28d9] text-white rounded-xl font-bold transition-all">
                  {`Download ${outputType.toUpperCase()}`}
                </a>
              </>
            ) : (
              <div className="flex flex-col items-center justify-center min-h-[300px] bg-[var(--bg-tertiary)] rounded-xl border border-theme text-theme-muted">
                <IconBox className="w-16 h-16 mb-4 opacity-20" />
                <p>3D preview will appear here</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
