import React, { useState, useRef, useCallback } from 'react';
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
  const [texture, setTexture] = useState(false);
  const [steps, setSteps] = useState(5);
  const [outputType, setOutputType] = useState('glb');
  const fileRefs = useRef<Record<ViewName, HTMLInputElement | null>>({ front: null, back: null, left: null, right: null });

  const handleFile = useCallback((view: ViewName, file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      setImages(prev => ({
        ...prev,
        [view]: { preview: dataUrl, b64: dataUrl.split(',')[1] },
      }));
      setResult(null);
      setError(null);
    };
    reader.readAsDataURL(file);
  }, []);

  const generate = async () => {
    if (!images.front) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const payload: Record<string, any> = {
        front: images.front.b64,
        num_inference_steps: steps,
        guidance_scale: 5.0,
        octree_resolution: 128,
        num_chunks: 8000,
        texture,
        type: outputType,
      };
      if (images.back) payload.back = images.back.b64;
      if (images.left) payload.left = images.left.b64;
      if (images.right) payload.right = images.right.b64;

      const res = await fetch(`${API_BASE}/api/v1/multiview-to-3d`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setResult({
        previewUrl: `${API_BASE}${data.preview_url}`,
        downloadUrl: `${API_BASE}${data.download_url}`,
      });

      onModelGenerated?.({
        id: crypto.randomUUID(),
        previewUrl: `${API_BASE}${data.preview_url}`,
        downloadUrl: `${API_BASE}${data.download_url}`,
        format: data.format,
        source: 'multiview-to-3d',
        createdAt: new Date().toISOString(),
        fromCache: data.from_cache ?? false,
      });
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const viewCount = VIEWS.filter(v => images[v]).length;

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div className="card p-6">
        <h2 className="text-xl font-bold text-heading mb-2 flex items-center">
          <IconBox className="mr-3 text-[#FF8C66]" /> Multi-View to 3D
        </h2>
        <p className="text-body text-sm mb-6">Upload 1-4 views of an object (front required) for higher-quality 3D reconstruction.</p>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left: Upload views */}
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
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => e.target.files?.[0] && handleFile(view, e.target.files[0])}
                  />
                </div>
              ))}
            </div>

            {/* Options */}
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 text-sm text-theme-secondary">
                <input type="checkbox" checked={texture} onChange={(e) => setTexture(e.target.checked)} className="accent-[#7C3AED]" />
                Generate texture
              </label>

              <label className="flex items-center gap-2 text-sm text-theme-secondary">
                Format:
                <select value={outputType} onChange={(e) => setOutputType(e.target.value)} className="bg-[var(--bg-input)] border border-theme rounded px-2 py-1 text-sm text-theme-primary">
                  <option value="glb">GLB</option>
                  <option value="obj">OBJ</option>
                  <option value="ply">PLY</option>
                  <option value="stl">STL</option>
                </select>
              </label>
              <label className="flex items-center gap-2 text-sm text-theme-secondary">
                Steps:
                <input type="number" value={steps} onChange={(e) => setSteps(Number(e.target.value))} min={1} max={100} className="w-16 bg-[var(--bg-input)] border border-theme rounded px-2 py-1 text-sm text-theme-primary" />
              </label>
            </div>

            <button
              onClick={generate}
              disabled={!images.front || loading}
              className="w-full px-4 py-3 bg-[#FF8C66] hover:bg-[#ff7a4d] disabled:opacity-50 disabled:cursor-not-allowed text-black rounded-xl font-bold transition-all flex items-center justify-center gap-2"
            >
              {loading ? (
                <><IconLoader className="w-5 h-5 animate-spin" /> Generating from {viewCount} view{viewCount > 1 ? 's' : ''}...</>
              ) : (
                `Generate from ${viewCount} view${viewCount > 1 ? 's' : ''}`
              )}
            </button>

            {error && <p className="text-red-400 text-sm bg-red-500/10 border border-red-500/20 p-3 rounded-lg">{error}</p>}
          </div>

          {/* Right: Result */}
          <div className="space-y-4">
            {result ? (
              <>
                <ModelViewer3D src={result.previewUrl} />
                <a
                  href={result.downloadUrl}
                  download
                  className="block w-full text-center px-4 py-3 bg-[#7C3AED] hover:bg-[#6d28d9] text-white rounded-xl font-bold transition-all"
                >
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
