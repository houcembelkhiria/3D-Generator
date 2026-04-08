import React, { useState, useRef, useCallback } from 'react';
import { GeneratedModel } from '../types';
import { ModelViewer3D } from './ModelViewer3D';
import { IconUpload, IconLoader, IconBox } from './Icons';
import { API_BASE } from '../api';


interface ImageTo3DProps {
  onModelGenerated?: (model: GeneratedModel) => void;
}

export const ImageTo3D: React.FC<ImageTo3DProps> = ({ onModelGenerated }) => {
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageB64, setImageB64] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ previewUrl: string; downloadUrl: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [texture, setTexture] = useState(false);
  const [steps, setSteps] = useState(5);
  const [outputType, setOutputType] = useState('glb');
  const fileRef = useRef<HTMLInputElement>(null);

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

  const generate = async () => {
    if (!imageB64) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/api/v1/image-to-3d`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: imageB64,
          num_inference_steps: steps,
          guidance_scale: 5.0,
          octree_resolution: 128,
          num_chunks: 8000,
          texture,
          type: outputType,
        }),
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
        source: 'image-to-3d',
        createdAt: new Date().toISOString(),
        fromCache: data.from_cache ?? false,
      });
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
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
          {/* Left: Upload */}
          <div className="space-y-4">
            <div
              className="dropzone flex flex-col items-center justify-center p-8 cursor-pointer min-h-[300px]"
              onClick={() => fileRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
            >
              {imagePreview ? (
                <img src={imagePreview} alt="Preview" className="max-h-64 rounded-lg object-contain" />
              ) : (
                <>
                  <IconUpload className="w-12 h-12 text-theme-muted mb-4" />
                  <p className="text-theme-muted text-sm">Drop an image here or click to browse</p>
                  <p className="text-theme-muted text-xs mt-1">PNG, JPG — transparent background recommended</p>
                </>
              )}
            </div>
            <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />

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
              disabled={!imageB64 || loading}
              className="w-full px-4 py-3 bg-[#FF8C66] hover:bg-[#ff7a4d] disabled:opacity-50 disabled:cursor-not-allowed text-black rounded-xl font-bold transition-all flex items-center justify-center gap-2"
            >
              {loading ? <><IconLoader className="w-5 h-5 animate-spin" /> Generating...</> : 'Generate 3D Model'}
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
