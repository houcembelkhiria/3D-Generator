import React, { useState, useRef, useCallback, useEffect } from 'react';
import { GeneratedModel } from '../types';
import { ModelViewer3D } from './ModelViewer3D';
import { IconUpload, IconLoader, IconBox } from './Icons';
import { API_BASE } from '../api';

const VIEWS = ['front', 'back', 'left', 'right'] as const;
type ViewName = typeof VIEWS[number];

interface MultiViewTo3DProps {
  onModelGenerated?: (model: GeneratedModel) => void;
}

export const MultiViewTo3D: React.FC<MultiViewTo3DProps> = ({ onModelGenerated }) => {
  const [images, setImages] = useState<Record<ViewName, { preview: string; b64: string } | null>>({
    front: null, back: null, left: null, right: null,
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ previewUrl: string; downloadUrl: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Basic params
  const [texture, setTexture] = useState(false);
  const [steps, setSteps] = useState(5);
  const [outputType, setOutputType] = useState('glb');
  // Advanced params
  const [seed, setSeed] = useState(1234);
  const [guidanceScale, setGuidanceScale] = useState(5.0);
  const [octreeResolution, setOctreeResolution] = useState(128);
  const [numChunks, setNumChunks] = useState(50000);
  const [faceCount, setFaceCount] = useState(20000);
  const [showAdvanced, setShowAdvanced] = useState(false);
  // UI
  const [elapsed, setElapsed] = useState(0);
  const [generationTime, setGenerationTime] = useState<number | null>(null);
  const fileRefs = useRef<Record<ViewName, HTMLInputElement | null>>({ front: null, back: null, left: null, right: null });
  const currentUidRef = useRef<string | null>(null);
  const cancelledRef = useRef(false);

  const handleFile = useCallback((view: ViewName, file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      setImages(prev => ({ ...prev, [view]: { preview: dataUrl, b64: dataUrl.split(',')[1] } }));
      setResult(null);
      setError(null);
    };
    reader.readAsDataURL(file);
  }, []);

  useEffect(() => {
    if (!loading) return;
    const t0 = Date.now();
    setElapsed(0);
    const tick = () => setElapsed(Math.round((Date.now() - t0) / 1000));
    const id = setInterval(tick, 1000);
    // Immediately correct the counter when the user returns to this browser tab
    const onVisible = () => { if (!document.hidden) tick(); };
    document.addEventListener('visibilitychange', onVisible);
    return () => { clearInterval(id); document.removeEventListener('visibilitychange', onVisible); };
  }, [loading]);

  const cancel = async () => {
    cancelledRef.current = true;
    const uid = currentUidRef.current;
    if (uid) {
      try { await fetch(`${API_BASE}/api/v1/generation/${uid}`, { method: 'DELETE' }); } catch {}
    }
    setLoading(false);
  };

  const generate = async () => {
    if (!images.front) return;
    cancelledRef.current = false;
    setLoading(true);
    setError(null);
    setResult(null);
    setGenerationTime(null);
    try {
      const payload: Record<string, any> = {
        front: images.front.b64,
        seed,
        num_inference_steps: steps,
        guidance_scale: guidanceScale,
        octree_resolution: octreeResolution,
        num_chunks: numChunks,
        texture,
        face_count: faceCount,
        type: outputType,
      };
      if (images.back) payload.back = images.back.b64;
      if (images.left) payload.left = images.left.b64;
      if (images.right) payload.right = images.right.b64;

      const submitRes = await fetch(`${API_BASE}/api/v1/multiview-to-3d/async`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
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
      let data: any = null;
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
      if (cancelledRef.current) return;
      setResult({ previewUrl: `${API_BASE}${data.preview_url}`, downloadUrl: `${API_BASE}${data.download_url}` });
      setGenerationTime(data.generation_time ?? null);
      onModelGenerated?.({
        id: crypto.randomUUID(),
        previewUrl: `${API_BASE}${data.preview_url}`,
        downloadUrl: `${API_BASE}${data.download_url}`,
        format: data.format,
        source: 'multiview-to-3d',
        createdAt: new Date().toISOString(),
        fromCache: data.from_cache ?? false,
        generationTime: data.generation_time,
      });
    } catch (e: any) {
      if (!cancelledRef.current) setError(e.message);
    } finally {
      setLoading(false);
      currentUidRef.current = null;
    }
  };

  const viewCount = VIEWS.filter(v => images[v]).length;

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div className="card p-6">
        <h2 className="text-xl font-bold text-heading mb-2 flex items-center">
          <IconBox className="mr-3 text-[#FF8C66]" /> Multi-View to 3D
        </h2>
        <p className="text-body text-sm mb-6">Upload 1–4 views of an object (front required) for higher-quality 3D reconstruction.</p>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left */}
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              {VIEWS.map(view => (
                <div key={view} className="space-y-1">
                  <span className="text-xs font-medium text-theme-secondary uppercase tracking-wider">
                    {view} {view === 'front' && <span className="text-[#FF8C66]">*</span>}
                  </span>
                  <div
                    className="dropzone flex flex-col items-center justify-center p-4 cursor-pointer aspect-square"
                    onClick={() => fileRefs.current[view]?.click()}
                    onDrop={(e) => { e.preventDefault(); e.dataTransfer.files[0] && handleFile(view, e.dataTransfer.files[0]); }}
                    onDragOver={(e) => e.preventDefault()}
                  >
                    {images[view] ? (
                      <img src={images[view]!.preview} alt={view} className="max-h-full rounded object-contain" />
                    ) : (
                      <IconUpload className="w-8 h-8 text-theme-muted" />
                    )}
                  </div>
                  <input
                    ref={el => { fileRefs.current[view] = el; }}
                    type="file" accept="image/*" className="hidden"
                    onChange={(e) => e.target.files?.[0] && handleFile(view, e.target.files[0])}
                  />
                </div>
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
                <input type="number" value={steps} onChange={(e) => setSteps(Number(e.target.value))} min={1} max={100}
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
                <label className="flex flex-col gap-1 text-xs text-theme-secondary">
                  Seed
                  <div className="flex gap-1">
                    <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))}
                      className="flex-1 min-w-0 bg-[var(--bg-input)] border border-theme rounded px-2 py-1 text-sm text-theme-primary" />
                    <button type="button" onClick={() => setSeed(Math.floor(Math.random() * 99999))} title="Random seed"
                      className="px-2 py-1 bg-[var(--bg-input)] border border-theme rounded hover:bg-[var(--bg-card)] transition-colors text-sm">⟳</button>
                  </div>
                </label>
                <label className="flex flex-col gap-1 text-xs text-theme-secondary">
                  <span>Guidance Scale <span className="font-mono text-theme-primary">{guidanceScale.toFixed(1)}</span></span>
                  <input type="range" value={guidanceScale} onChange={(e) => setGuidanceScale(Number(e.target.value))}
                    min={0} max={20} step={0.5} className="accent-[#FF8C66] mt-2" />
                </label>
                <label className="flex flex-col gap-1 text-xs text-theme-secondary">
                  Mesh Resolution
                  <select value={octreeResolution} onChange={(e) => setOctreeResolution(Number(e.target.value))}
                    className="bg-[var(--bg-input)] border border-theme rounded px-2 py-1 text-sm text-theme-primary">
                    <option value={64}>64 — fastest</option>
                    <option value={128}>128 — default</option>
                    <option value={160}>160 — detailed</option>
                    <option value={192}>192 — high quality</option>
                  </select>
                </label>
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
                {texture && (
                  <label className="flex flex-col gap-1 text-xs text-theme-secondary col-span-2">
                    Max Face Count
                    <input type="number" value={faceCount} onChange={(e) => setFaceCount(Number(e.target.value))}
                      min={1000} step={1000}
                      className="bg-[var(--bg-input)] border border-theme rounded px-2 py-1 text-sm text-theme-primary" />
                  </label>
                )}
              </div>
            )}

            <button
              type="button"
              onClick={() => { setSteps(1); setOctreeResolution(64); setNumChunks(200000); setShowAdvanced(true); }}
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
              <button onClick={generate} disabled={!images.front || loading}
                className="w-full px-4 py-3 bg-[#FF8C66] hover:bg-[#ff7a4d] disabled:opacity-50 disabled:cursor-not-allowed text-black rounded-xl font-bold transition-all flex items-center justify-center gap-2">
                {`Generate from ${viewCount} view${viewCount !== 1 ? 's' : ''}`}
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
